import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_FILE   = Path("data/eval_contrastive_gpt4.1m_to_llama3.1-8b.txt")
CACHE_FILE  = Path("data/logprob_cache.pkl")
OUTPUT_DIR  = Path("figures")
RESULTS_DIR = Path("results")

# ── Descriptor Vocabulary ─────────────────────────────────────────────────────
# Goldberg (1992) TDA: 20 adjectives per Big-Five facet (+ and − poles).
BIG5 = {
    "EXT": [
        "talkative","bold","assertive","extraverted","energetic",
        "outgoing","sociable","lively","adventurous","enthusiastic",
        "withdrawn","quiet","reserved","timid","inhibited",
        "shy","silent","introverted","submissive","unadventurous",
    ],
    "AGR": [
        "kind","cooperative","sympathetic","warm","helpful",
        "friendly","trustful","generous","agreeable","gentle",
        "harsh","uncooperative","unsympathetic","cold","unhelpful",
        "unfriendly","distrustful","stingy","disagreeable","unkind",
    ],
    "CON": [
        "organized","efficient","systematic","thorough","careful",
        "reliable","dependable","precise","diligent","orderly",
        "careless","disorganized","inefficient","haphazard","sloppy",
        "unreliable","undependable","imprecise","negligent","disorderly",
    ],
    "NEU": [
        "anxious","nervous","tense","moody","temperamental",
        "insecure","unstable","fearful","emotional","worrying",
        "relaxed","calm","stable","secure","confident",
        "easygoing","untroubled","composed","serene","balanced",
    ],
    "OPN": [
        "creative","imaginative","insightful","artistic","curious",
        "intellectual","inventive","perceptive","thoughtful","philosophical",
        "unimaginative","uncreative","incurious","conventional","uninventive",
        "imperceptive","shallow","literal","narrow","rigid",
    ],
}

# Style descriptors that map to model-identity dimensions rather than
# human personality (hedging, formality, verbosity, certainty, warmth…).
STYLE = [
    "verbose","concise","terse","elaborate","succinct",
    "hedging","definitive","tentative","cautious","direct",
    "formal","informal","professional","casual","polished",
    "structured","flowing","fragmented","methodical","systematic",
    "confident","uncertain","authoritative","hesitant","decisive",
    "empathetic","detached","supportive","clinical","neutral",
]

ALL_ADJ = [a for adjs in BIG5.values() for a in adjs] + STYLE

FACET_OF = {a: f for f, adjs in BIG5.items() for a in adjs}
for a in STYLE:
    FACET_OF[a] = "STY"

FACET_COLORS = {
    "EXT": "#e74c3c", "AGR": "#2ecc71", "CON": "#3498db",
    "NEU": "#f39c12", "OPN": "#9b59b6", "STY": "#95a5a6",
}

CONDITIONS = ["source_response", "disguised_response", "target_response"]
COND_COLORS = {
    "source_response":    "#e74c3c",
    "disguised_response": "#f39c12",
    "target_response":    "#2ecc71",
}
COND_LABELS = {
    "source_response":    "Source (GPT-4.1-mini)",
    "disguised_response": "Disguised",
    "target_response":    "Target (Llama-3.1-8B)",
}

TRUNCATE = 600  # chars — keeps prompts short and cost predictable


# Data Loading 
def load_data(path: Path, max_rows: int) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[:max_rows]


# Matrix: Embedding-cosine (default, no API key needed)
def matrix_embeddings(rows: list[dict], adj_list: list[str]) -> dict[str, np.ndarray]:
    """
    X[i,j] = log( (cosine_sim(embed(text_i), embed(adj_j)) + 1) / 2 )
    as a log-probability proxy. Uses all-MiniLM-L6-v2 locally.
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("  Encoding adjectives...")
    adj_embs = model.encode(adj_list, normalize_embeddings=True, show_progress_bar=False)

    matrices = {}
    for cond in CONDITIONS:
        texts = [r[cond][:TRUNCATE] for r in rows]
        print(f"  Encoding {cond} ({len(texts)} texts)...")
        t_embs = model.encode(texts, normalize_embeddings=True,
                              batch_size=64, show_progress_bar=True)
        sim = t_embs @ adj_embs.T                              # N × D in [−1, 1]
        matrices[cond] = np.log(np.clip((sim + 1) / 2, 1e-9, 1.0))
    return matrices


# Matrix: OpenAI log-probs
def _vocab_token_ids(adj_list: list[str]) -> dict[str, int]:
    """Return {adj: token_id} for adjectives that are a single token."""
    import tiktoken
    enc = tiktoken.get_encoding("o200k_base")  # gpt-4o / gpt-4.1 family
    out = {}
    for adj in adj_list:
        toks = enc.encode(" " + adj)   # leading space for mid-sentence encoding
        if len(toks) == 1:
            out[adj] = toks[0]
        else:
            toks2 = enc.encode(adj)
            if len(toks2) == 1:
                out[adj] = toks2[0]
    return out


def _query_logprobs(client, text: str, tok_map: dict[str, int]) -> dict[str, float]:
    """
    One API call → {adj: log_prob} for all single-token adjectives.
    logit_bias +100 forces the model to choose only from vocab tokens,
    so top_logprobs=20 covers the full (vocab-restricted) distribution.
    """
    # Split vocab into chunks of 20 (OpenAI logit_bias key limit is large,
    # but top_logprobs only returns 20 per call — so chunk to cover all vocab).
    adj_items = list(tok_map.items())
    results: dict[str, float] = {}

    for chunk_start in range(0, len(adj_items), 20):
        chunk = dict(adj_items[chunk_start:chunk_start + 20])
        logit_bias = {str(tid): 100 for tid in chunk.values()}
        prompt = (
            "A single adjective that best describes the writing style and "
            f"personality of the author of the following text is '____'.\n\n"
            f"Text: {text[:TRUNCATE]}"
        )
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1,
            logprobs=True,
            top_logprobs=20,
            logit_bias=logit_bias,
        )
        top = resp.choices[0].logprobs.content[0].top_logprobs
        for t in top:
            adj_clean = t.token.strip(" '\"").lower()
            if adj_clean in chunk:
                results[adj_clean] = t.logprob

    floor = -12.0
    return {adj: results.get(adj, floor) for adj in tok_map}


def matrix_logprob(rows: list[dict], adj_list: list[str], workers: int = 1) -> tuple[dict, list]:
    import time
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from openai import OpenAI

    tok_map = _vocab_token_ids(adj_list)
    single_adjs = list(tok_map.keys())
    N = len(rows)
    D = len(single_adjs)
    n_chunks = max(1, (D + 19) // 20)   # API calls per (row, condition)
    total_cells = N * len(CONDITIONS)

    print(f"  {len(single_adjs)}/{len(adj_list)} adjectives are single-token.")
    print(f"  {n_chunks} API call(s) per text  ×  {total_cells} texts = {total_cells * n_chunks} calls max")
    print(f"  Workers: {workers}")

    cache: dict = {}
    lock = threading.Lock()

    if CACHE_FILE.exists():
        with open(CACHE_FILE, "rb") as f:
            cache = pickle.load(f)
        cached = sum(1 for i, row in enumerate(rows)
                     for cond in CONDITIONS
                     if (cond, i, row[cond][:80]) in cache)
        print(f"  Cache: {cached}/{total_cells} already done "
              f"({cached * n_chunks} API calls saved)")

    # Each worker gets its own client (OpenAI client is not thread-safe to share)
    def make_client():
        return OpenAI()

    completed = [0]
    api_calls  = [0]
    t0 = time.time()

    def _flush_cache():
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(cache, f)

    def _print_progress():
        done    = completed[0]
        elapsed = time.time() - t0
        calls   = api_calls[0]
        rate    = calls / elapsed if elapsed > 0 and calls > 0 else 0
        remaining = (total_cells - done) * n_chunks
        eta_s   = remaining / rate if rate > 0 else 0
        eta_str = f"{eta_s/60:.1f}min" if eta_s > 90 else f"{eta_s:.0f}s"
        print(
            f"  {done:>3}/{total_cells} texts  |  {calls} API calls  |  ETA {eta_str}          ",
            end="\r", flush=True,
        )

    def fetch(i: int, cond: str) -> tuple[int, str, dict]:
        key = (cond, i, rows[i][cond][:80])
        with lock:
            if key in cache:
                return i, cond, cache[key]
        result = _query_logprobs(make_client(), rows[i][cond], tok_map)
        with lock:
            cache[key] = result
            api_calls[0] += n_chunks
            completed[0] += 1
            if completed[0] % 10 == 0:
                _flush_cache()
            _print_progress()
        return i, cond, result

    # Build list of work items that aren't already cached
    todo = [
        (i, cond)
        for i in range(N)
        for cond in CONDITIONS
        if (cond, i, rows[i][cond][:80]) not in cache
    ]
    # Bump completed counter for already-cached items so ETA is accurate
    completed[0] = total_cells - len(todo)
    _print_progress()

    matrices = {c: np.zeros((N, D)) for c in CONDITIONS}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch, i, cond): (i, cond) for i, cond in todo}
        for fut in as_completed(futures):
            i, cond, lp = fut.result()
            for j, adj in enumerate(single_adjs):
                matrices[cond][i, j] = lp.get(adj, -12.0)

    # Fill cached items that weren't in todo
    for i, row in enumerate(rows):
        for cond in CONDITIONS:
            key = (cond, i, row[cond][:80])
            if key in cache:
                lp = cache[key]
                for j, adj in enumerate(single_adjs):
                    matrices[cond][i, j] = lp.get(adj, -12.0)

    print()  # newline after \r
    _flush_cache()
    print(f"  Done. Cache saved to {CACHE_FILE}")

    return matrices, single_adjs


# SVD Factorization
def factorize_joint(matrices: dict[str, np.ndarray], k: int):
    """
    Stack all conditions → (3N × D) matrix, mean-center, SVD.
    Returns per-condition score matrices (N × k), singular values, and V (D × k).
    Scores are scaled by singular values (i.e. X ≈ U·Σ·Vᵀ, we return U·Σ).
    """
    X = np.vstack([matrices[c] for c in CONDITIONS])   # 3N × D
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    N = len(matrices[CONDITIONS[0]])
    scores = U * S                                     # 3N × full_k (whitened)
    U_by_cond = {c: scores[i*N:(i+1)*N, :k] for i, c in enumerate(CONDITIONS)}
    return U_by_cond, S, Vt[:k].T                     # V: D × k


# ── Plots ─────────────────────────────────────────────────────────────────────
def _confidence_ellipse(ax, xs: np.ndarray, ys: np.ndarray, color: str, n_std: float = 1.5):
    cov = np.cov(xs, ys)
    if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
        return
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * n_std * np.sqrt(np.maximum(vals, 0))
    ax.add_patch(Ellipse(
        xy=(xs.mean(), ys.mean()), width=w, height=h, angle=theta,
        edgecolor=color, facecolor="none", lw=2, alpha=0.8,
    ))


def plot_scatter(U_by_cond: dict, out_path: Path):
    """2-D scatter: PC1 vs PC2 and PC1 vs PC3, with 1.5-sigma ellipses."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (px, py) in zip(axes, [(0, 1), (0, 2)]):
        for cond in CONDITIONS:
            sc = U_by_cond[cond]
            xs, ys = sc[:, px], sc[:, py]
            c = COND_COLORS[cond]
            ax.scatter(xs, ys, c=c, alpha=0.45, s=18, label=COND_LABELS[cond])
            _confidence_ellipse(ax, xs, ys, c)
        ax.set_xlabel(f"PC{px+1}")
        ax.set_ylabel(f"PC{py+1}")
        ax.set_title(f"PC{px+1} vs PC{py+1}")
        if (px, py) == (0, 1):
            ax.legend(fontsize=8)
    fig.suptitle("Latent Style Space — Source / Disguised / Target", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path}")


def plot_violin(U_by_cond: dict, out_path: Path, k: int = 4):
    """Per-PC violin plots showing score distributions for each condition."""
    k = min(k, next(iter(U_by_cond.values())).shape[1])
    fig, axes = plt.subplots(1, k, figsize=(4 * k, 4), sharey=False)
    if k == 1:
        axes = [axes]
    for pc_idx, ax in enumerate(axes):
        data = [U_by_cond[c][:, pc_idx] for c in CONDITIONS]
        vp = ax.violinplot(data, positions=range(len(CONDITIONS)),
                           showmedians=True, showextrema=False)
        for body, cond in zip(vp["bodies"], CONDITIONS):
            body.set_facecolor(COND_COLORS[cond])
            body.set_alpha(0.7)
        vp["cmedians"].set_color("white")
        ax.set_xticks(range(len(CONDITIONS)))
        ax.set_xticklabels(["Source", "Disguised", "Target"], fontsize=8)
        ax.set_title(f"PC{pc_idx+1}")
        if pc_idx == 0:
            ax.set_ylabel("Score")
    fig.suptitle("PC Score Distributions by Condition", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path}")


def plot_scree(S: np.ndarray, out_path: Path):
    k = min(20, len(S))
    var = (S ** 2) / (S ** 2).sum()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(1, k + 1), var[:k] * 100, color="#3498db", alpha=0.8)
    ax2 = ax.twinx()
    ax2.plot(range(1, k + 1), np.cumsum(var[:k]) * 100,
             "o-", color="#e74c3c", markersize=4, label="Cumulative")
    ax.set_xlabel("Component")
    ax.set_ylabel("Variance explained (%)")
    ax2.set_ylabel("Cumulative (%)")
    ax.set_title("Scree Plot")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path}")


def plot_loadings(V: np.ndarray, adj_list: list[str], out_path: Path, k: int = 4):
    """Heatmap of adjective loadings on top-k PCs, colour-coded by facet."""
    k = min(k, V.shape[1])
    D = len(adj_list)
    fig, ax = plt.subplots(figsize=(k * 1.6 + 2, D * 0.16 + 1))
    im = ax.imshow(V[:, :k], aspect="auto", cmap="RdBu_r", vmin=-0.15, vmax=0.15)
    ax.set_xticks(range(k))
    ax.set_xticklabels([f"PC{i+1}" for i in range(k)], fontsize=9)
    ax.set_yticks(range(D))
    ax.set_yticklabels(adj_list, fontsize=6)
    for ytick, adj in zip(ax.get_yticklabels(), adj_list):
        ytick.set_color(FACET_COLORS.get(FACET_OF.get(adj, "STY"), "black"))
    plt.colorbar(im, ax=ax, shrink=0.5, label="Loading")
    ax.set_title("Adjective Loadings on Top PCs  (colour = Big-Five facet)")

    # Add a compact legend for facet colours
    handles = [
        plt.Line2D([0], [0], color=c, lw=3, label=f)
        for f, c in FACET_COLORS.items()
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=6,
              bbox_to_anchor=(1.18, 0), framealpha=0.7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path}")


def plot_persistence(U_by_cond: dict, out_path: Path, k: int = 5,
                     title_tag: str = "") -> tuple[float, np.ndarray]:
    """
    Arena-style radar chart of disguise effectiveness per latent PC.

    Each spoke runs 0 → CLAMP %.
      • 0 %  = source signature fully retained   (blue dashed ring)
      • 100 % = complete shift to target          (green dashed ring)
    Values that exceed CLAMP are clamped to the rim and annotated.

    Returns (persistence_score, frac_array).
    """
    CLAMP = 150

    k = min(k, next(iter(U_by_cond.values())).shape[1])
    mu = {c: U_by_cond[c].mean(axis=0)[:k] for c in CONDITIONS}
    direction    = mu["target_response"]    - mu["source_response"]
    displacement = mu["disguised_response"] - mu["source_response"]

    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(np.abs(direction) > 1e-9, displacement / direction, np.nan)

    persistence = float(1.0 - np.nanmean(frac))

    PC_LABELS = {
        0: "PC1\n(STY: neutral, formal)",
        1: "PC2\n(STY: formal−)",
        2: "PC3\n(EXT: reserved)",
        3: "PC4\n(CON: professional)",
        4: "PC5\n(STY: clinical)",
    }
    labels = [PC_LABELS.get(i, f"PC{i+1}") for i in range(k)]

    angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, k, endpoint=False)
    angles_closed = np.append(angles, angles[0])

    def spoke_coords(values_pct):
        v = np.clip(np.nan_to_num(values_pct, nan=0.0), -30, CLAMP)
        r = v / CLAMP
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        return np.append(x, x[0]), np.append(y, y[0])

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(aspect="equal"))
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)
    ax.axis("off")
    fig.patch.set_facecolor("#F7F5F0")
    ax.set_facecolor("#F7F5F0")

    for pct in [0, 25, 50, 75, 100, CLAMP]:
        r = pct / CLAMP
        ring_x = r * np.cos(angles_closed)
        ring_y = r * np.sin(angles_closed)
        lw = 1.2 if pct in (0, 100) else 0.5
        ls = "-" if pct in (0, 100) else "--"
        color = "#BBBBBB" if pct not in (0, 100) else "#CCCCCC"
        ax.plot(ring_x, ring_y, lw=lw, ls=ls, color=color, zorder=1)
        if pct in (0, 25, 50, 75, 100):
            lx = (r + 0.04) * np.cos(angles[0] + 0.08)
            ly = (r + 0.04) * np.sin(angles[0] + 0.08)
            ax.text(lx, ly, f"{pct}%", fontsize=8, color="#999999",
                    ha="left", va="center", zorder=3)

    for ang in angles:
        ax.plot([0, np.cos(ang)], [0, np.sin(ang)],
                lw=0.7, color="#CCCCCC", zorder=1)

    for ref_pct, cond_key in [(0, "source_response"), (100, "target_response")]:
        rx, ry = spoke_coords(np.full(k, float(ref_pct)))
        ax.fill(rx, ry, alpha=0.07, color=COND_COLORS[cond_key], zorder=3)
        ax.plot(rx, ry, lw=1.6, ls="--", color=COND_COLORS[cond_key], alpha=0.7, zorder=3)

    dis_pct = frac * 100
    dis_x, dis_y = spoke_coords(dis_pct)
    ax.fill(dis_x, dis_y, alpha=0.22, color=COND_COLORS["disguised_response"], zorder=4)
    ax.plot(dis_x, dis_y, lw=2.5, color=COND_COLORS["disguised_response"], zorder=5)
    ax.scatter(dis_x[:-1], dis_y[:-1], s=55, color=COND_COLORS["disguised_response"],
               zorder=6, edgecolors="white", linewidths=1.2)

    for i, val in enumerate(dis_pct):
        if np.isnan(val) or val <= CLAMP:
            continue
        rim_x, rim_y = np.cos(angles[i]), np.sin(angles[i])
        ax.annotate(
            f"{val:.0f}%",
            xy=(rim_x, rim_y),
            xytext=(rim_x * 1.18, rim_y * 1.18),
            fontsize=9, fontweight="bold",
            color=COND_COLORS["disguised_response"],
            ha="center", va="center",
            arrowprops=dict(arrowstyle="->",
                            color=COND_COLORS["disguised_response"], lw=1.4),
            zorder=7,
        )

    label_r = 1.22
    for ang, lbl in zip(angles, labels):
        lx, ly = label_r * np.cos(ang), label_r * np.sin(ang)
        ha = "center"
        if np.cos(ang) > 0.3:
            ha = "left"
        elif np.cos(ang) < -0.3:
            ha = "right"
        ax.text(lx, ly, lbl, fontsize=9.5, color="#3A3A3A",
                ha=ha, va="center", multialignment="center", zorder=8)

    ax.text(0, 1.42, "Disguise effectiveness per latent dimension",
            fontsize=13, fontweight="bold", color="#1A1A1A",
            ha="center", va="top", zorder=8)
    subtitle = f"{title_tag}  |  persistence: {persistence * 100:.1f}%" if title_tag \
               else f"persistence: {persistence * 100:.1f}%"
    ax.text(0, 1.33, subtitle, fontsize=9, color="#777777",
            ha="center", va="top", zorder=8)

    legend_items = [
        mpatches.Patch(facecolor=COND_COLORS[c], alpha=0.75, label=COND_LABELS[c])
        for c in CONDITIONS
    ]
    ax.legend(handles=legend_items, loc="lower center",
              bbox_to_anchor=(0.5, -0.06), ncol=3,
              fontsize=9, frameon=False,
              handlelength=1.4, handleheight=0.9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  saved {out_path}")
    return persistence, frac


def plot_per_sample_trajectory(U_by_cond: dict, out_path: Path, pc_idx: int = 0):
    """
    Per-sample PC score, normalised so source mean = 0, target mean = 1.
    Reveals how many individual prompts resist/overshoot disguise.
    """
    src = U_by_cond["source_response"][:, pc_idx]
    dis = U_by_cond["disguised_response"][:, pc_idx]
    tgt = U_by_cond["target_response"][:, pc_idx]

    src_m, tgt_m = src.mean(), tgt.mean()
    span = tgt_m - src_m if abs(tgt_m - src_m) > 1e-9 else 1.0
    src_n = (src - src_m) / span
    dis_n = (dis - src_m) / span
    tgt_n = (tgt - src_m) / span

    order = np.argsort(dis_n)
    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(order))
    ax.fill_between(x, src_n[order], tgt_n[order], alpha=0.08, color="grey")
    ax.plot(x, src_n[order], color=COND_COLORS["source_response"],    alpha=0.5, lw=0.7, label="Source")
    ax.plot(x, dis_n[order], color=COND_COLORS["disguised_response"],  alpha=0.9, lw=1.0, label="Disguised")
    ax.plot(x, tgt_n[order], color=COND_COLORS["target_response"],    alpha=0.5, lw=0.7, label="Target")
    ax.axhline(0, color=COND_COLORS["source_response"], ls="--", lw=0.8, alpha=0.6)
    ax.axhline(1, color=COND_COLORS["target_response"], ls="--", lw=0.8, alpha=0.6)
    ax.set_xlabel("Prompt (sorted by disguised score)")
    ax.set_ylabel(f"PC{pc_idx+1} score (normalised)")
    ax.set_title(f"Per-sample PC{pc_idx+1}: Source → Disguised → Target trajectory")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path}")


# ── Linear Probe ──────────────────────────────────────────────────────────────
def run_probe(matrices: dict[str, np.ndarray]) -> dict:
    """
    Logistic regression trained on source(0) vs. target(1) latent features.
    Evaluated on disguised: what fraction looks like source, what like target?
    5-fold CV accuracy tells us how separable the two models are in this space.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score

    X_src = matrices["source_response"]
    X_tgt = matrices["target_response"]
    X_dis = matrices["disguised_response"]

    X_tr = np.vstack([X_src, X_tgt])
    y_tr = np.array([0] * len(X_src) + [1] * len(X_tgt))

    scaler = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_dis_s = scaler.transform(X_dis)

    clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
    cv_acc = cross_val_score(clf, X_tr_s, y_tr, cv=5, scoring="accuracy").mean()
    clf.fit(X_tr_s, y_tr)

    preds = clf.predict(X_dis_s)
    probs = clf.predict_proba(X_dis_s)[:, 1]

    return {
        "cv_accuracy":            float(cv_acc),
        "disguised_as_source_%":  float((preds == 0).mean() * 100),
        "disguised_as_target_%":  float((preds == 1).mean() * 100),
        "mean_target_prob":       float(probs.mean()),
        "median_target_prob":     float(np.median(probs)),
    }


def _short_name(model_id: str) -> str:
    """Strip org prefix and clean up model identifiers for plot titles."""
    name = model_id.split("/")[-1]                         # drop "openai/" etc.
    name = name.replace("Meta-Llama-", "Llama-")
    name = name.replace("-Instruct", "").replace("-instruct", "")
    name = name.replace("gpt-4.1-mini", "GPT-4.1-mini")
    name = name.replace("gpt-4o", "GPT-4o")
    return name


# ── Paper figures ─────────────────────────────────────────────────────────────
def plot_paper_pcs(U_by_cond: dict, S: np.ndarray, out_path: Path, title_tag: str = ""):
    """
    Single paper figure: 5 PCs side-by-side as violin plots.
    Each panel shows source / disguised / target distributions.
    Variance explained annotated on each panel title.
    """
    k = min(5, next(iter(U_by_cond.values())).shape[1])
    var_pct = (S[:k] ** 2) / (S ** 2).sum() * 100

    fig, axes = plt.subplots(1, k, figsize=(3.2 * k, 4.2), sharey=False)
    fig.subplots_adjust(wspace=0.35)

    for pc_idx, ax in enumerate(axes):
        data = [U_by_cond[c][:, pc_idx] for c in CONDITIONS]
        vp = ax.violinplot(data, positions=[0, 1, 2],
                           showmedians=True, showextrema=False, widths=0.75)
        for body, cond in zip(vp["bodies"], CONDITIONS):
            body.set_facecolor(COND_COLORS[cond])
            body.set_edgecolor("none")
            body.set_alpha(0.82)
        vp["cmedians"].set_color("white")
        vp["cmedians"].set_linewidth(1.8)

        # mean dots
        for pos, cond in enumerate(CONDITIONS):
            ax.scatter([pos], [U_by_cond[cond][:, pc_idx].mean()],
                       color="white", s=28, zorder=3, linewidths=0.8,
                       edgecolors=COND_COLORS[cond])

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Source", "Disguised", "Target"],
                           fontsize=8, rotation=15, ha="right")
        ax.set_title(f"PC{pc_idx+1}\n({var_pct[pc_idx]:.1f}% var)", fontsize=9)
        ax.tick_params(axis="y", labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        if pc_idx == 0:
            ax.set_ylabel("Latent score", fontsize=9)

    # shared legend — patch proxies
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=COND_COLORS[c], label=COND_LABELS[c])
               for c in CONDITIONS]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.04))

    base = "Latent Style Dimensions: Source / Disguised / Target"
    fig.suptitle(f"{base}\n{title_tag}" if title_tag else base,
                 fontsize=11, y=1.02)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path}")


def plot_paper_adjectives(V: np.ndarray, adj_list: list[str], out_path: Path,
                          n: int = 10, title_tag: str = ""):
    """
    Single paper figure: one panel per PC showing the top-n positive and
    negative loading adjectives as a horizontal diverging bar chart.
    Bars are coloured by Big-Five facet.
    """
    k = min(5, V.shape[1])
    fig, axes = plt.subplots(1, k, figsize=(3.6 * k, 5.5), sharey=False)
    fig.subplots_adjust(wspace=0.55)

    for pc_idx, ax in enumerate(axes):
        loads = V[:, pc_idx]
        # top-n positive and top-n negative, sorted by absolute value
        pos_idx = np.argsort(loads)[::-1][:n]
        neg_idx = np.argsort(loads)[:n]
        # combine and deduplicate, preserving order
        shown = list(dict.fromkeys(list(neg_idx) + list(pos_idx[::-1])))

        adjs   = [adj_list[i] for i in shown]
        values = [loads[i]    for i in shown]
        colors = [FACET_COLORS.get(FACET_OF.get(adj_list[i], "STY"), "#aaa")
                  for i in shown]

        y = np.arange(len(adjs))
        bars = ax.barh(y, values, color=colors, alpha=0.85, height=0.7)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(adjs, fontsize=7)
        ax.set_xlabel("Loading", fontsize=8)
        ax.set_title(f"PC{pc_idx+1}", fontsize=10, fontweight="bold")
        ax.tick_params(axis="x", labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        # zero reference
        ax.set_xlim(min(values) * 1.25, max(values) * 1.25)

    # facet legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, label=f) for f, c in FACET_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=len(FACET_COLORS),
               fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.04))

    base = "Top Adjective Loadings per Principal Component"
    fig.suptitle(f"{base}\n{title_tag}" if title_tag else base,
                 fontsize=11, y=1.02)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path}")


# ── Results text log ──────────────────────────────────────────────────────────
def save_results(
    rows, adj_list, args_method, S,
    persistence, frac, probe, V, adj_list_full,
    out_path: Path,
    title_tag: str = "",
):
    k = len(frac)
    var_pct = (S[:k] ** 2) / (S ** 2).sum() * 100

    lines = []
    lines.append("=" * 60)
    lines.append("LATENT ANALYSIS RESULTS")
    lines.append("=" * 60)
    lines.append(f"Data file    : {DATA_FILE}")
    lines.append(f"Method       : {args_method}")
    if title_tag:
        lines.append(f"Run tag      : {title_tag}")
    lines.append(f"N responses  : {len(rows)}  (per condition)")
    lines.append(f"Descriptors  : {len(adj_list)} adjectives")
    lines.append(f"PCs retained : {k}")
    lines.append("")
    lines.append("── Variance explained ──────────────────────────────────")
    for i in range(k):
        bar = "█" * int(var_pct[i] / 100 * 30)
        lines.append(f"  PC{i+1}: {var_pct[i]:5.2f}%  {bar}")
    lines.append(f"  Cumulative (PC1–{k}): {var_pct[:k].sum():.1f}%")
    lines.append("")
    lines.append("── Source persistence ──────────────────────────────────")
    lines.append(f"  Overall persistence score : {persistence * 100:.1f}%")
    lines.append(f"  Overall disguise effect   : {(1-persistence) * 100:.1f}%")
    lines.append("")
    lines.append("  Per-PC shift toward target (0% = source retained, 100% = full shift):")
    for i, f in enumerate(frac):
        bar = "█" * int(max(0, min(f, 1)) * 25)
        lines.append(f"    PC{i+1}: {f*100:+6.1f}%  {bar}")
    lines.append("")
    lines.append("── Linear probe (src=0, tgt=1) ─────────────────────────")
    lines.append(f"  5-fold CV accuracy         : {probe['cv_accuracy']*100:.1f}%")
    lines.append(f"  Disguised classified as source : {probe['disguised_as_source_%']:.1f}%")
    lines.append(f"  Disguised classified as target : {probe['disguised_as_target_%']:.1f}%")
    lines.append(f"  Mean  P(target | disguised)    : {probe['mean_target_prob']:.3f}")
    lines.append(f"  Median P(target | disguised)   : {probe['median_target_prob']:.3f}")
    lines.append("")
    lines.append("── Top adjective loadings per PC ───────────────────────")
    for pc in range(k):
        loads = V[:, pc]
        pos = np.argsort(loads)[::-1][:8]
        neg = np.argsort(loads)[:8]
        pos_str = ", ".join(f"{adj_list_full[i]}({loads[i]:+.3f})" for i in pos)
        neg_str = ", ".join(f"{adj_list_full[i]}({loads[i]:+.3f})" for i in neg)
        lines.append(f"  PC{pc+1}  +pole: {pos_str}")
        lines.append(f"  PC{pc+1}  −pole: {neg_str}")
    lines.append("")
    lines.append("=" * 60)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"  saved {out_path}")


# ── Console helpers ───────────────────────────────────────────────────────────
def print_top_adjectives(V: np.ndarray, adj_list: list[str], k: int = 3, n: int = 7):
    print("\nTop adjectives per PC (+ pole / − pole):")
    for pc in range(min(k, V.shape[1])):
        loads = V[:, pc]
        pos = np.argsort(loads)[::-1][:n]
        neg = np.argsort(loads)[:n]
        pos_str = "  ".join(f"{adj_list[i]}({loads[i]:+.3f})" for i in pos)
        neg_str = "  ".join(f"{adj_list[i]}({loads[i]:+.3f})" for i in neg)
        print(f"  PC{pc+1}+  {pos_str}")
        print(f"  PC{pc+1}−  {neg_str}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method",   choices=["embeddings", "logprob"], default="embeddings",
                        help="embeddings: local (default) | logprob: OpenAI API")
    parser.add_argument("--k",        type=int, default=5,   help="Number of PCs to keep")
    parser.add_argument("--max_rows", type=int, default=200, help="Rows to use from data file")
    parser.add_argument("--workers",  type=int, default=5,   help="Parallel API workers (logprob only)")
    parser.add_argument("--d_method",
                        choices=["random_sampling", "contrastive", "style", "vibe"],
                        default="contrastive",
                        help="Disguise strategy used to generate the data")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {DATA_FILE}…")
    rows = load_data(DATA_FILE, args.max_rows)
    print(f"  {len(rows)} rows, {len(ALL_ADJ)} descriptor adjectives")

    src_name = _short_name(rows[0].get("source_model", "source"))
    tgt_name = _short_name(rows[0].get("target_model", "target"))
    title_tag = f"{src_name} → {tgt_name}  |  {args.d_method}"
    print(f"  {title_tag}")

    if args.method == "logprob":
        print("Building log-probability matrix (OpenAI API)…")
        matrices, adj_list = matrix_logprob(rows, ALL_ADJ, workers=args.workers)
    else:
        print("Building embedding-cosine matrix (sentence-transformers)…")
        matrices = matrix_embeddings(rows, ALL_ADJ)
        adj_list = ALL_ADJ

    print(f"\nSVD factorization (k={args.k})…")
    U_by_cond, S, V = factorize_joint(matrices, k=args.k)

    print("\nGenerating figures…")
    plot_scatter(U_by_cond,  OUTPUT_DIR / "scatter.png")
    plot_violin(U_by_cond,   OUTPUT_DIR / "violin.png",      k=args.k)
    plot_scree(S,             OUTPUT_DIR / "scree.png")
    plot_loadings(V, adj_list, OUTPUT_DIR / "loadings.png",  k=min(4, args.k))
    persistence, frac = plot_persistence(
        U_by_cond, OUTPUT_DIR / "persistence.png", k=args.k, title_tag=title_tag)
    plot_per_sample_trajectory(
        U_by_cond, OUTPUT_DIR / "trajectory_pc1.png", pc_idx=0)

    print("\nGenerating paper figures…")
    plot_paper_pcs(U_by_cond, S, OUTPUT_DIR / "paper_pcs.png", title_tag=title_tag)
    plot_paper_adjectives(V, adj_list, OUTPUT_DIR / "paper_adjectives.png", title_tag=title_tag)

    print("\nRunning linear probe…")
    probe = run_probe(matrices)

    print_top_adjectives(V, adj_list, k=min(3, args.k))

    print("\n" + "=" * 58)
    print("LATENT ANALYSIS SUMMARY")
    print("=" * 58)
    print(f"  Descriptor matrix:           {len(rows)} × {len(adj_list)}")
    print(f"  Method:                      {args.method}")
    print(f"  Source persistence score:    {persistence * 100:.1f}%")
    print(f"  Disguise effectiveness:      {(1 - persistence) * 100:.1f}%")
    print()
    print("  Per-PC shift (source→target %):")
    for pc_i, f in enumerate(frac[:args.k]):
        bar = "█" * int(max(0, min(f, 1)) * 20)
        print(f"    PC{pc_i+1}: {f*100:5.1f}%  {bar}")
    print()
    print("  Linear probe (src=0, tgt=1):")
    print(f"    5-fold CV accuracy:        {probe['cv_accuracy'] * 100:.1f}%")
    print(f"    Disguised → source:        {probe['disguised_as_source_%']:.1f}%")
    print(f"    Disguised → target:        {probe['disguised_as_target_%']:.1f}%")
    print(f"    Mean P(target|disguised):  {probe['mean_target_prob']:.3f}")
    print()
    print(f"  Figures saved to {OUTPUT_DIR}/")

    print("\nSaving results text log…")
    save_results(
        rows, adj_list, args.method, S,
        persistence, frac, probe, V, adj_list,
        RESULTS_DIR / "summary.txt",
        title_tag=title_tag,
    )
    print("=" * 58)


if __name__ == "__main__":
    main()
