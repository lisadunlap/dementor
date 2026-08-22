#!/usr/bin/env python
"""Regenerate the two steering figures from committed steering_figure_stats.json with LARGE fonts.
Matches committed style: cone=red, contrast=blue, random=gray."""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STATS = sys.argv[1]
OUT01 = sys.argv[2]  # dissociation
OUT03 = sys.argv[3]  # steering effects
d = json.load(open(STATS))

CONE = "#c0392b"; CONTRAST = "#3a6fa5"; RANDOM = "#9aa0a6"; INK = "#1a1a19"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#6b6a63", "axes.linewidth": 0.8,
})

# ---------- Figure 3: benchmark dissociation (grouped bars, BIG fonts) ----------
pb = d["per_benchmark"]
order = ["advbench", "harmbench", "strongreject", "sorrybench", "sgbench"]
names = ["AdvBench", "HarmBench", "StrongREJECT", "SORRY-Bench", "SG-Bench"]
FS = 9
fig, ax = plt.subplots(figsize=(3.45, 3.0))
x = np.arange(len(order)); w = 0.27
for off, arm, col, lab in [(-w, "fp", CONTRAST, "contrast"),
                           (0.0, "rand", RANDOM, "random"),
                           (w, "cone", CONE, "refusal cone")]:
    means = [pb[b][arm][0] for b in order]
    errs = [pb[b][arm][1] for b in order]
    ax.bar(x + off, means, w, yerr=errs, color=col, label=lab,
           error_kw=dict(lw=0.8, ecolor=INK, capsize=1.8))
ax.axhline(0, color=INK, lw=0.9)
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=FS-0.5, rotation=28, ha="right")
ax.set_ylabel("harm change under ablation (%)", fontsize=FS)
ax.tick_params(axis="y", labelsize=FS)
ax.legend(fontsize=FS-0.5, loc="upper left", frameon=False, handlelength=1.1,
          borderaxespad=0.2, labelspacing=0.25)
ax.margins(x=0.03)
fig.subplots_adjust(bottom=0.16, left=0.14, right=0.98, top=0.98)
fig.savefig(OUT01, bbox_inches="tight")
print("wrote", OUT01)

# ---------- Figure 4: per-model steering effects (2 panels, BIG fonts) ----------
pm = d["per_model"]
slugs = sorted(pm.keys(), key=lambda s: pm[s]["cone"], reverse=True)
labels = [f"{s} ({pm[s]['n']})" for s in slugs]
y = np.arange(len(slugs))[::-1]  # top = highest cone
h = 0.26
FS2 = 11
fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 6.4), gridspec_kw=dict(width_ratios=[2.15, 1]))
# panel a: all three arms
for off, arm, col in [(h, "cone", CONE), (0.0, "fp", CONTRAST), (-h, "rand", RANDOM)]:
    a.barh(y + off, [pm[s][arm] for s in slugs], h, color=col)
a.axvline(0, color=INK, lw=1.0)
a.set_yticks(y); a.set_yticklabels(labels, fontsize=FS2-1.5)
a.set_xlabel("harm change under ablation (pp)", fontsize=FS2)
a.tick_params(axis="x", labelsize=FS2)
a.set_title("(a) all three arms", fontsize=FS2+1)
from matplotlib.patches import Patch
a.legend(handles=[Patch(color=CONE, label="refusal cone"),
                  Patch(color=CONTRAST, label="cross-model contrast"),
                  Patch(color=RANDOM, label="random control")],
         fontsize=FS2-1, loc="lower right", frameon=False)
# panel b: contrast vs random zoomed
for off, arm, col in [(h/2, "fp", CONTRAST), (-h/2, "rand", RANDOM)]:
    b.barh(y + off, [pm[s][arm] for s in slugs], h, color=col)
b.axvline(0, color=INK, lw=1.0)
b.set_yticks(y); b.set_yticklabels([])
b.set_xlabel("harm change (pp), zoomed", fontsize=FS2)
b.tick_params(axis="x", labelsize=FS2)
b.set_title("(b) contrast vs random only", fontsize=FS2+1)
fig.subplots_adjust(left=0.24, right=0.99, top=0.96, bottom=0.07, wspace=0.06)
fig.savefig(OUT03, bbox_inches="tight")
print("wrote", OUT03)
