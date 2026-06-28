"""D1 — Capability vs Style dissociation: join graded capability to cached style
persistence, run the gap census + H1 (false-clean) + H2 (dissociation statistic),
and emit figures. NO model calls, NO API spend; pure local stats over cached CSVs.

Reads:
  results/d1_gsm8k_accuracy_long.csv          (from grade_gsm8k.py)
  results/d1_gsm8k_peritem_correct.csv        (paired bootstrap)
  results/matrix_ladder/gsm8k_matrix_ladder.csv  (style persistence; no recompute)
  results/source_distinctiveness.csv          (style-survivor tiers)

Writes:
  results/d1_capability_vs_style_percell.csv
  results/d1_dissociation_stats.csv
  results/d1_fig_dissociation_scatter.png
  results/d1_fig_ladder_curves.png
  results/d1_fig_tier_box.png

Capability convention (see grade_gsm8k.py header): for cell {source}_to_{target},
acc_source = imitator's own competence (origin), acc_target = imitated model
(destination). cap_xfer = (acc_B - acc_source) / (acc_target - acc_source) = the
fraction of the source->target competence gap the disguised imitator closes. This
is the dual of style `persistence` (retention of the SOURCE style; ->0 = audit
"clean"). gap_ok = |acc_target - acc_source| >= 0.05 (cap_xfer undefined below).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
ACC = ROOT / "results/d1_gsm8k_accuracy_long.csv"
PERITEM = ROOT / "results/d1_gsm8k_peritem_correct.csv"
LADDER = ROOT / "results/matrix_ladder/gsm8k_matrix_ladder.csv"
DISTINCT = ROOT / "results/source_distinctiveness.csv"
PERCELL = ROOT / "results/d1_capability_vs_style_percell.csv"
STATS = ROOT / "results/d1_dissociation_stats.csv"
FIG_SCATTER = ROOT / "results/d1_fig_dissociation_scatter.png"
FIG_LADDER = ROOT / "results/d1_fig_ladder_curves.png"
FIG_TIER = ROOT / "results/d1_fig_tier_box.png"

GAP_DELTA = 0.05
TAU_CLEAN = 0.10
CAP_THRESH = 0.25
RUNG_ORDER = ["just_name_it", "random_sampling", "stylistic", "sft", "dpo"]
RNG = np.random.default_rng(42)


def build_percell() -> pd.DataFrame:
    acc = pd.read_csv(ACC)
    # acc_source / acc_target averaged over available seeds, per cell
    base = (acc[acc["rung_or_role"].isin(["source", "target"])]
            .groupby(["source", "target", "rung_or_role"])["acc"].mean().unstack())
    base.columns = [f"acc_{c}" for c in base.columns]
    base = base.reset_index()
    base["gap"] = base["acc_target"] - base["acc_source"]
    base["gap_ok"] = base["gap"].abs() >= GAP_DELTA

    # acc_B per rung
    rungs = acc[acc["rung_or_role"].isin(RUNG_ORDER)][
        ["source", "target", "rung_or_role", "acc"]].rename(
        columns={"rung_or_role": "rung", "acc": "acc_B"})

    ladder = pd.read_csv(LADDER)[["source", "target", "rung", "persistence",
                                  "anchored", "trustworthy"]].rename(
        columns={"anchored": "persistence_anchored"})

    df = rungs.merge(base, on=["source", "target"]).merge(
        ladder, on=["source", "target", "rung"], how="inner")
    assert len(df) == 60, f"expected 60 joined rows, got {len(df)}"

    df["cap_xfer_raw"] = (df["acc_B"] - df["acc_source"]) / df["gap"]
    df["cap_xfer"] = df["cap_xfer_raw"].clip(0, 1)
    df.loc[~df["gap_ok"], ["cap_xfer_raw", "cap_xfer"]] = np.nan  # undefined
    df["delta_acc"] = df["acc_B"] - df["acc_source"]

    # style-survivor tier from source_distinctiveness (per-source)
    dist = pd.read_csv(DISTINCT)[["model", "tier", "dpo_persistence"]].rename(
        columns={"model": "source"})
    df = df.merge(dist, on="source", how="left")

    order = {r: i + 1 for i, r in enumerate(RUNG_ORDER)}
    df["order"] = df["rung"].map(order)
    df = df.sort_values(["source", "target", "order"]).reset_index(drop=True)
    df.to_csv(PERCELL, index=False)
    print(f"[percell] wrote {len(df)} rows -> {PERCELL}")
    return df


def paired_bootstrap_delta(peritem, source, target, n_boot=2000):
    """Paired bootstrap of delta_acc = acc(dpo) - acc(source_seed1) over 200 items."""
    sub = peritem[(peritem["source"] == source) & (peritem["target"] == target)]
    b = sub[sub["condition"] == "rung_dpo"].set_index("prompt_norm")["correct"]
    s = sub[sub["condition"] == "source_seed1"].set_index("prompt_norm")["correct"]
    common = b.index.intersection(s.index)
    bv, sv = b.loc[common].values, s.loc[common].values
    diff = bv - sv
    n = len(diff)
    obs = diff.mean()
    boots = np.array([diff[RNG.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return obs, lo, hi, p


def bootstrap_corr(x, y, n_boot=5000):
    x, y = np.asarray(x), np.asarray(y)
    n = len(x)
    rs = []
    for _ in range(n_boot):
        idx = RNG.integers(0, n, n)
        if np.std(x[idx]) < 1e-9 or np.std(y[idx]) < 1e-9:
            continue
        rs.append(np.corrcoef(x[idx], y[idx])[0, 1])
    rs = np.array(rs)
    return np.percentile(rs, [2.5, 97.5])


def main() -> None:
    df = build_percell()
    dpo = df[df["rung"] == "dpo"].copy()
    peritem = pd.read_csv(PERITEM)

    # ---- gap census ----
    gap_census = df[df["rung"] == "dpo"][
        ["source", "target", "acc_source", "acc_target", "gap", "gap_ok"]].copy()
    n_gap_ok = int(gap_census["gap_ok"].sum())
    print(f"\n=== GAP CENSUS: {n_gap_ok}/12 cells have |gap|>={GAP_DELTA} ===")
    print(gap_census.round(3).to_string(index=False))

    okdpo = dpo[dpo["gap_ok"]].copy()

    # ---- H1: false-clean signature, with paired-bootstrap significance ----
    print(f"\n=== H1: false-clean cells (persistence<= {TAU_CLEAN}, "
          f"cap_xfer>= {CAP_THRESH}, delta_acc sig>0) over {len(okdpo)} gap-ok cells ===")
    h1_rows = []
    for _, r in okdpo.iterrows():
        obs, lo, hi, p = paired_bootstrap_delta(peritem, r["source"], r["target"])
        false_clean = (r["persistence"] <= TAU_CLEAN and r["cap_xfer"] >= CAP_THRESH
                       and lo > 0)
        h1_rows.append({
            "source": r["source"], "target": r["target"],
            "persistence": round(r["persistence"], 3),
            "cap_xfer": round(r["cap_xfer"], 3),
            "delta_acc": round(obs, 3), "delta_ci_lo": round(lo, 3),
            "delta_ci_hi": round(hi, 3), "delta_p": round(p, 4),
            "false_clean": false_clean,
        })
    h1 = pd.DataFrame(h1_rows)
    print(h1.to_string(index=False))
    n_false_clean = int(h1["false_clean"].sum())
    fc_list = [f"{r.source}->{r.target}" for r in h1.itertuples() if r.false_clean]

    # ---- H2: dissociation statistic over gap-ok DPO cells ----
    x = okdpo["persistence"].values
    xa = okdpo["persistence_anchored"].values
    y = okdpo["cap_xfer"].values
    yraw = okdpo["cap_xfer_raw"].values
    n = len(x)

    def corr_block(xv, yv, label):
        if len(xv) < 3 or np.std(xv) < 1e-9 or np.std(yv) < 1e-9:
            return dict(label=label, n=len(xv), pearson_r=np.nan, pearson_p=np.nan,
                        spearman_rho=np.nan, spearman_p=np.nan, r_ci_lo=np.nan,
                        r_ci_hi=np.nan)
        pr, pp = stats.pearsonr(xv, yv)
        sr, sp = stats.spearmanr(xv, yv)
        lo, hi = bootstrap_corr(xv, yv)
        return dict(label=label, n=len(xv), pearson_r=pr, pearson_p=pp,
                    spearman_rho=sr, spearman_p=sp, r_ci_lo=lo, r_ci_hi=hi)

    h2_main = corr_block(x, y, "cap_xfer_vs_persistence")
    h2_anch = corr_block(xa, y, "cap_xfer_vs_persistence_anchored")
    h2_raw = corr_block(x, yraw, "cap_xfer_raw_vs_persistence")
    print(f"\n=== H2: dissociation over {n} gap-ok DPO cells ===")
    for b in (h2_main, h2_anch, h2_raw):
        print(f"  {b['label']}: r={b['pearson_r']:.3f} (95%CI "
              f"[{b['r_ci_lo']:.3f},{b['r_ci_hi']:.3f}], p={b['pearson_p']:.3f}), "
              f"rho={b['spearman_rho']:.3f} (p={b['spearman_p']:.3f})")

    # ---- H2b: Mann-Whitney U of cap_xfer by style-survivor tier ----
    retains = okdpo[okdpo["tier"] == "retains"]["cap_xfer"].dropna().values
    launders = okdpo[okdpo["tier"] == "launders"]["cap_xfer"].dropna().values
    if len(retains) and len(launders):
        u, up = stats.mannwhitneyu(retains, launders, alternative="two-sided")
    else:
        u, up = np.nan, np.nan
    print(f"\n=== H2b: cap_xfer by style tier (gap-ok DPO) ===")
    print(f"  retains (n={len(retains)}): {np.round(retains,3)}  mean={np.mean(retains) if len(retains) else float('nan'):.3f}")
    print(f"  launders (n={len(launders)}): {np.round(launders,3)}  mean={np.mean(launders) if len(launders) else float('nan'):.3f}")
    print(f"  Mann-Whitney U={u}, p={up:.3f}")

    # ---- secondary: cap_xfer vs source distinctiveness ----
    dist = pd.read_csv(DISTINCT)[["model", "distinct_minilm"]].rename(columns={"model": "source"})
    okd = okdpo.merge(dist, on="source", how="left")
    if okd["distinct_minilm"].notna().sum() >= 3 and okd["cap_xfer"].std() > 1e-9:
        dr, dp = stats.pearsonr(okd["distinct_minilm"], okd["cap_xfer"])
    else:
        dr, dp = np.nan, np.nan

    # ---- directional summary (the real story: does competence move toward target?) ----
    # delta_acc at DPO over gap-ok cells; signed toward-target movement = delta_acc * sign(gap)
    h1_idx = h1.set_index(["source", "target"])
    okdpo2 = okdpo.copy()
    okdpo2["delta_acc_dpo"] = [h1_idx.loc[(r.source, r.target), "delta_acc"]
                              for r in okdpo2.itertuples()]
    okdpo2["toward_target"] = okdpo2["delta_acc_dpo"] * np.sign(okdpo2["gap"])
    n_regressed = int((okdpo2["delta_acc_dpo"] < 0).sum())
    n_moved_toward = int((okdpo2["toward_target"] > 0).sum())
    mean_delta_acc = float(okdpo2["delta_acc_dpo"].mean())

    # ---- write stats table ----
    stat_rows = [
        {"metric": "n_cells_total", "value": 12},
        {"metric": "n_gap_ok", "value": n_gap_ok},
        {"metric": "gap_delta", "value": GAP_DELTA},
        {"metric": "mean_delta_acc_dpo_gapok", "value": round(mean_delta_acc, 3)},
        {"metric": "n_cells_competence_regressed_dpo", "value": n_regressed},
        {"metric": "n_cells_moved_toward_target_dpo", "value": n_moved_toward},
        {"metric": "n_false_clean", "value": n_false_clean},
        {"metric": "false_clean_cells", "value": ";".join(fc_list) if fc_list else ""},
        {"metric": "mean_cap_xfer_dpo_gapok", "value": round(float(np.nanmean(y)), 3)},
        {"metric": "mean_persistence_dpo_gapok", "value": round(float(np.mean(x)), 3)},
        {"metric": "H2_pearson_r", "value": round(h2_main["pearson_r"], 3)},
        {"metric": "H2_pearson_p", "value": round(h2_main["pearson_p"], 3)},
        {"metric": "H2_pearson_r_ci_lo", "value": round(h2_main["r_ci_lo"], 3)},
        {"metric": "H2_pearson_r_ci_hi", "value": round(h2_main["r_ci_hi"], 3)},
        {"metric": "H2_spearman_rho", "value": round(h2_main["spearman_rho"], 3)},
        {"metric": "H2_spearman_p", "value": round(h2_main["spearman_p"], 3)},
        {"metric": "H2_anchored_pearson_r", "value": round(h2_anch["pearson_r"], 3)},
        {"metric": "H2_anchored_pearson_p", "value": round(h2_anch["pearson_p"], 3)},
        {"metric": "H2_raw_pearson_r", "value": round(h2_raw["pearson_r"], 3)},
        {"metric": "H2b_mannwhitney_U", "value": float(u) if not np.isnan(u) else np.nan},
        {"metric": "H2b_mannwhitney_p", "value": round(up, 3) if not np.isnan(up) else np.nan},
        {"metric": "H2b_mean_cap_xfer_retains", "value": round(float(np.mean(retains)), 3) if len(retains) else np.nan},
        {"metric": "H2b_mean_cap_xfer_launders", "value": round(float(np.mean(launders)), 3) if len(launders) else np.nan},
        {"metric": "secondary_cap_vs_distinct_minilm_r", "value": round(dr, 3) if not np.isnan(dr) else np.nan},
        {"metric": "secondary_cap_vs_distinct_minilm_p", "value": round(dp, 3) if not np.isnan(dp) else np.nan},
    ]
    pd.DataFrame(stat_rows).to_csv(STATS, index=False)
    # append H1 per-cell detail
    h1.to_csv(STATS.with_name("d1_h1_false_clean_detail.csv"), index=False)
    print(f"\n[stats] wrote -> {STATS}")

    make_figures(df, okdpo, h2_main, retains, launders)


def make_figures(df, okdpo, h2_main, retains, launders):
    src_colors = {s: c for s, c in zip(sorted(df["source"].unique()),
                  ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"])}

    # --- Fig 1: dissociation scatter (gap-ok DPO) ---
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for _, r in okdpo.iterrows():
        ax.scatter(r["persistence"], r["cap_xfer"], s=120,
                   color=src_colors[r["source"]], edgecolor="k", zorder=3)
        ax.annotate(f"{r['source'][:4]}->{r['target'][:4]}",
                    (r["persistence"], r["cap_xfer"]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("style persistence @ DPO  (->0 = audit returns 'clean')")
    ax.set_ylabel("capability transfer @ DPO  (frac of source->target gap closed)")
    r, p = h2_main["pearson_r"], h2_main["pearson_p"]
    rho = h2_main["spearman_rho"]
    ax.set_title(f"Capability vs Style dissociation (gap-ok cells, n={len(okdpo)})\n"
                 f"Pearson r={r:.2f} (p={p:.2f}), Spearman rho={rho:.2f}")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, mec="k", label=s)
               for s, c in src_colors.items() if s in okdpo["source"].values]
    ax.legend(handles=handles, title="source (imitator base)", fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG_SCATTER, dpi=140); plt.close(fig)
    print(f"[fig] {FIG_SCATTER}")

    # --- Fig 2: ladder small-multiples (all 12 cells) ---
    cells = sorted(df.groupby(["source", "target"]).groups.keys())
    fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharex=True)
    for ax, (s, t) in zip(axes.flat, cells):
        g = df[(df["source"] == s) & (df["target"] == t)].sort_values("order")
        x = g["order"]
        ax.plot(x, g["acc_B"], "o-", color="#1f77b4", label="acc_B")
        ax.axhline(g["acc_source"].iloc[0], ls="--", color="gray", lw=1, label="acc_source")
        ax.axhline(g["acc_target"].iloc[0], ls=":", color="green", lw=1, label="acc_target")
        ax2 = ax.twinx()
        ax2.plot(x, g["persistence"], "s-", color="#d62728", alpha=0.7, label="persistence")
        ax2.set_ylim(-0.05, 1.05)
        ax.set_ylim(0.4, 1.0)
        gap_ok = g["gap_ok"].iloc[0]
        ax.set_title(f"{s[:6]}->{t[:6]}  gap={g['gap'].iloc[0]:+.2f}"
                     f"{'' if gap_ok else '  (gap<.05)'}", fontsize=9,
                     color="black" if gap_ok else "gray")
        ax.set_xticks(range(1, 6)); ax.set_xticklabels(RUNG_ORDER, rotation=45, fontsize=7)
    axes[0, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle("GSM8K ladder: style persistence (red) dives to ~0 at SFT/DPO, but "
                 "acc_B (blue) does NOT track toward acc_target (green) — it often "
                 "regresses. No capability payload rides along.", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG_LADDER, dpi=120); plt.close(fig)
    print(f"[fig] {FIG_LADDER}")

    # --- Fig 3: cap_xfer by style-survivor tier (box) ---
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    data, labels = [], []
    if len(retains):
        data.append(retains); labels.append(f"style-retains\n(n={len(retains)})")
    if len(launders):
        data.append(launders); labels.append(f"style-launders\n(n={len(launders)})")
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.5)
    for patch, c in zip(bp["boxes"], ["#ff9896", "#aec7e8"]):
        patch.set_facecolor(c)
    for i, d in enumerate(data):
        ax.scatter(np.full(len(d), i + 1) + RNG.normal(0, 0.04, len(d)), d,
                   color="k", zorder=3, s=30)
    ax.set_ylabel("capability transfer @ DPO (gap-ok cells)")
    ax.set_title("Capability transfer does NOT track the style-survivor tier")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(FIG_TIER, dpi=140); plt.close(fig)
    print(f"[fig] {FIG_TIER}")


if __name__ == "__main__":
    main()
