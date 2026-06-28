"""Decompose disguise/persistence along named behavioural directions instead of a
single opaque source-vs-target "fingerprint" axis.

Two axis sets (--axes):
  big5   — Big-Five personality facets (Goldberg 1992 TDA): EXT AGR CON NEU OPN.
  style  — model-identity style axes: verbosity, formality, certainty, structure,
           warmth. Tests whether the fingerprint that eludes personality facets
           lives in *style*.

Two scoring backends (--backend):
  embedding  — MiniLM cosine to each adjective (deterministic, no API key; blunt).
  logprob    — gpt-4.1-mini "the best adjective for this author is '____'" with
               logit_bias + top_logprobs (Naz's faithful method; far sharper).
               Reads OPENAI_API_KEY/OPENAI_BASE_URL from .env EXPLICITLY so a shell
               proxy base-URL can't shadow it.

Each axis's signed score is mean(+pole) − mean(−pole). Per rung we report, per axis,
the source/target/disguised mean, sep = target−source, the SCALE-STABLE separation
sep_d = sep/pooled_std (Cohen's d), and movement = (disguised−source)/sep. Movement
is gated on |sep_d| downstream since the raw ratio explodes when sep is tiny.
Source & target are scored once per cell; the matrix run caches per cell (resumable).

Usage:
  python -m experiments.analysis.big5_directions --cell <…/cells/A_to_B> --axes style --backend logprob --max-prompts 40
  python -m experiments.analysis.big5_directions --matrix --axes style --backend logprob --max-prompts 40 --workers 16
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Goldberg (1992) TDA — first 10 = positive pole, last 10 = negative pole.
BIG5 = {
    "EXT": ["talkative", "bold", "assertive", "extraverted", "energetic",
            "outgoing", "sociable", "lively", "adventurous", "enthusiastic",
            "withdrawn", "quiet", "reserved", "timid", "inhibited",
            "shy", "silent", "introverted", "submissive", "unadventurous"],
    "AGR": ["kind", "cooperative", "sympathetic", "warm", "helpful",
            "friendly", "trustful", "generous", "agreeable", "gentle",
            "harsh", "uncooperative", "unsympathetic", "cold", "unhelpful",
            "unfriendly", "distrustful", "stingy", "disagreeable", "unkind"],
    "CON": ["organized", "efficient", "systematic", "thorough", "careful",
            "reliable", "dependable", "precise", "diligent", "orderly",
            "careless", "disorganized", "inefficient", "haphazard", "sloppy",
            "unreliable", "undependable", "imprecise", "negligent", "disorderly"],
    "NEU": ["anxious", "nervous", "tense", "moody", "temperamental",
            "insecure", "unstable", "fearful", "emotional", "worrying",
            "relaxed", "calm", "stable", "secure", "confident",
            "easygoing", "untroubled", "composed", "serene", "balanced"],
    "OPN": ["creative", "imaginative", "insightful", "artistic", "curious",
            "intellectual", "inventive", "perceptive", "thoughtful", "philosophical",
            "unimaginative", "uncreative", "incurious", "conventional", "uninventive",
            "imperceptive", "shallow", "literal", "narrow", "rigid"],
}

# Style / model-identity axes (same 10 +pole / 10 −pole format).
STYLE = {
    "VERB": ["verbose", "wordy", "elaborate", "lengthy", "detailed",
             "expansive", "rambling", "chatty", "padded", "sprawling",
             "concise", "terse", "succinct", "brief", "compact",
             "crisp", "short", "laconic", "economical", "clipped"],
    "FORM": ["formal", "professional", "polished", "proper", "refined",
             "scholarly", "precise", "stiff", "academic", "ceremonious",
             "informal", "casual", "colloquial", "chatty", "relaxed",
             "conversational", "breezy", "slangy", "folksy", "loose"],
    "CERT": ["confident", "assertive", "definitive", "decisive", "emphatic",
             "categorical", "authoritative", "firm", "positive", "bold",
             "hedging", "tentative", "uncertain", "hesitant", "cautious",
             "qualified", "equivocal", "vague", "unsure", "timid"],
    "STRU": ["structured", "organized", "systematic", "methodical", "orderly",
             "sequential", "logical", "coherent", "tidy", "ordered",
             "rambling", "disorganized", "scattered", "fragmented", "chaotic",
             "meandering", "haphazard", "disjointed", "messy", "loose"],
    "WARM": ["warm", "friendly", "supportive", "empathetic", "kind",
             "encouraging", "caring", "gentle", "cheerful", "reassuring",
             "detached", "clinical", "cold", "impersonal", "neutral",
             "aloof", "distant", "blunt", "dispassionate", "sterile"],
}
AXIS_SETS = {"big5": BIG5, "style": STYLE}
RUNGS = ["just_name_it", "random_sampling", "stylistic", "sft", "dpo"]
TRUNCATE = 600


# ── embedding backend (deterministic) ──────────────────────────────────────────
def _load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def _axis_embeddings(model, axdict):
    return {f: (model.encode(adj[:10], normalize_embeddings=True),
                model.encode(adj[10:], normalize_embeddings=True))
            for f, adj in axdict.items()}


def emb_facet_scores(texts, model, axes) -> dict[str, np.ndarray]:
    emb = model.encode([str(t)[:TRUNCATE] for t in texts],
                       normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    return {f: (emb @ pos.T).mean(1) - (emb @ neg.T).mean(1) for f, (pos, neg) in axes.items()}


# ── logprob backend (OpenAI; sharper) ──────────────────────────────────────────
def _openai_client():
    from dotenv import dotenv_values
    from openai import OpenAI
    ev = dotenv_values(".env")  # explicit so a shell proxy OPENAI_BASE_URL can't win
    return OpenAI(api_key=ev["OPENAI_API_KEY"],
                  base_url=ev.get("OPENAI_BASE_URL") or "https://api.openai.com/v1")


def _single_token_poles(axdict):
    """Per axis (pos, neg) single-o200k-token adjectives, plus {adj: token_id}."""
    import tiktoken
    enc = tiktoken.get_encoding("o200k_base")

    def tid(a):
        t = enc.encode(" " + a)
        if len(t) == 1:
            return t[0]
        t2 = enc.encode(a)
        return t2[0] if len(t2) == 1 else None

    poles, ids = {}, {}
    for f, adj in axdict.items():
        pos = [a for a in adj[:10] if tid(a)]
        neg = [a for a in adj[10:] if tid(a)]
        poles[f] = (pos, neg)
        for a in pos + neg:
            ids[a] = tid(a)
    return poles, ids


def _logprobs_for_text(client, text, ids, model="gpt-4.1-mini"):
    items = list(ids.items())
    res: dict[str, float] = {}
    for i in range(0, len(items), 20):  # top_logprobs caps at 20 → chunk the vocab
        chunk = dict(items[i:i + 20])
        for attempt in range(5):  # retry transient errors / dropped sockets on sleep-wake
            try:
                r = client.chat.completions.create(
                    model=model, max_tokens=1, logprobs=True, top_logprobs=20,
                    logit_bias={str(v): 100 for v in chunk.values()},
                    messages=[{"role": "user", "content":
                               "A single adjective that best describes the writing style and "
                               "personality of the author of the following text is '____'.\n\nText: "
                               + str(text)[:TRUNCATE]}])
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
        for t in r.choices[0].logprobs.content[0].top_logprobs:
            a = t.token.strip(" '\"").lower()
            if a in chunk:
                res[a] = t.logprob
    return {a: res.get(a, -12.0) for a in ids}


def lp_facet_scores(texts, client, poles, ids, workers=8) -> dict[str, np.ndarray]:
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=workers) as ex:
        lps = list(ex.map(lambda t: _logprobs_for_text(client, t, ids), texts))
    return {f: np.array([np.mean([lp[a] for a in pos]) - np.mean([lp[a] for a in neg]) for lp in lps])
            for f, (pos, neg) in poles.items()}


# ── per-cell analysis (source/target scored once) ──────────────────────────────
def analyze_cell(cell: Path, *, backend="embedding", max_prompts=None, workers=8,
                 model=None, axes=None, client=None, poles=None, ids=None) -> pd.DataFrame:
    staged = cell / "staged"
    rfiles = {r: staged / f"{r}.csv" for r in RUNGS if (staged / f"{r}.csv").exists()}
    if not rfiles:
        return pd.DataFrame()

    if backend == "logprob":
        facets = list(poles)
        score = lambda texts: lp_facet_scores(texts, client, poles, ids, workers)
    else:
        facets = list(axes)
        score = lambda texts: emb_facet_scores(texts, model, axes)

    def col(df, name):
        return (df.head(max_prompts) if max_prompts else df)[name].tolist()

    base = pd.read_csv(next(iter(rfiles.values())))  # source/target shared across rungs
    sc = score(col(base, "source_response"))
    tc = score(col(base, "target_response"))
    src_mu = {f: float(sc[f].mean()) for f in facets}
    tgt_mu = {f: float(tc[f].mean()) for f in facets}
    pooled = {f: float(np.sqrt((sc[f].var() + tc[f].var()) / 2 + 1e-12)) for f in facets}

    rows = []
    for rung, f in rfiles.items():
        gc = score(col(pd.read_csv(f), "model_response"))
        for fac in facets:
            s, t, g = src_mu[fac], tgt_mu[fac], float(gc[fac].mean())
            sep = t - s
            rows.append({"rung": rung, "facet": fac, "source": s, "target": t,
                         "disguised": g, "sep": sep, "sep_d": sep / pooled[fac],
                         "movement": (g - s) / sep if abs(sep) > 1e-9 else np.nan})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", help="Path to one cell dir (…/cells/<src>_to_<tgt>).")
    ap.add_argument("--matrix", action="store_true", help="Aggregate all 36 cells.")
    ap.add_argument("--axes", choices=list(AXIS_SETS), default="big5")
    ap.add_argument("--backend", choices=["embedding", "logprob"], default="embedding")
    ap.add_argument("--max-prompts", type=int, default=None, help="Subsample prompts per condition (cost).")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    axdict = AXIS_SETS[a.axes]
    facets = list(axdict)
    kw = {"backend": a.backend, "max_prompts": a.max_prompts, "workers": a.workers}
    if a.backend == "logprob":
        kw["client"] = _openai_client()
        kw["poles"], kw["ids"] = _single_token_poles(axdict)
    else:
        kw["model"] = _load_model()
        kw["axes"] = _axis_embeddings(kw["model"], axdict)

    if a.cell:
        df = analyze_cell(Path(a.cell), **kw)
        piv = df.pivot(index="rung", columns="facet", values="movement").reindex(RUNGS)[facets]
        sep = df.pivot(index="rung", columns="facet", values="sep").reindex(RUNGS)[facets]
        print(f"\nCell: {a.cell}  (axes={a.axes}, backend={a.backend})")
        print("\nmovement toward target per axis (1=became target, 0=stayed source):")
        print(piv.round(2).to_string())
        print("\nsource↔target separation per axis (|small| ⇒ movement is noise):")
        print(sep.round(3).to_string())
        return

    if a.matrix:
        cells = [c for c in sorted(Path("data/results").glob("*/analysis/cells/*"))
                 if (c / "staged").is_dir()]
        cache = Path(f"data/results/big5_cache_{a.axes}_{a.backend}")
        cache.mkdir(parents=True, exist_ok=True)
        frames = []
        for i, c in enumerate(cells, 1):
            ds = c.parts[c.parts.index("results") + 1]
            cf = cache / f"{ds}__{c.name}.csv"
            if cf.exists():  # resume: cached cell (survives sleep/crash, no re-spend)
                print(f"  [{i}/{len(cells)}] {ds}/{c.name} (cached)", flush=True)
                frames.append(pd.read_csv(cf))
                continue
            print(f"  [{i}/{len(cells)}] {ds}/{c.name}", flush=True)
            d = analyze_cell(c, **kw)
            d["dataset"], d["cell"] = ds, c.name
            d.to_csv(cf, index=False)
            frames.append(d)
        allc = pd.concat(frames, ignore_index=True)
        out = Path(f"data/results/big5_directions_matrix_{a.axes}_{a.backend}.csv")
        allc.to_csv(out, index=False)
        # (1) SEPARATION MAP — which named axes distinguish the model pairs.
        cellsep = allc[allc["rung"] == RUNGS[-1]][["dataset", "cell", "facet", "sep_d"]]
        print(f"\n=== SEPARATION  |Cohen d| source↔target per axis (axes={a.axes}, backend={a.backend}) ===")
        for fac in facets:
            s = cellsep[cellsep["facet"] == fac]["sep_d"].abs()
            print(f"  {fac}: mean|d|={s.mean():.2f}   cells separating (|d|>0.3): {(s > 0.3).sum()}/{len(s)}")
        # (2) MOVEMENT — only where the pair actually separates on that axis.
        sig = allc[allc["sep_d"].abs() > 0.3]
        agg = sig.pivot_table(index="rung", columns="facet", values="movement",
                              aggfunc="mean").reindex(RUNGS)[facets]
        nn = sig[sig["rung"] == RUNGS[-1]].groupby("facet")["cell"].nunique()
        print("\n=== MOVEMENT toward target per axis × rung (cells with |d|>0.3 only) ===")
        print(agg.round(2).to_string())
        print("  cells contributing per axis:", {f: int(nn.get(f, 0)) for f in facets})
        print(f"\nwrote {out}")
        return

    ap.error("pass --cell <dir> or --matrix")


if __name__ == "__main__":
    main()
