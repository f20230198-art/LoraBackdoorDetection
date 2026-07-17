#!/usr/bin/env python3
r"""
Figure 5 of "Weights Aren't Enough" -- data + standalone plotting script.
Prepared for redrawing. Self-contained: no project imports, no data files needed.

    Figure 5 (\label{fig:scenario}, figures/fig_scenario_comparison.png)
    "Detection collapses across attack scenarios; the backdoors keep working"

WHAT THE FIGURE SHOWS
---------------------
Four screening scenarios, two bars each:
  * detection (%)  -- how often the weight-space detector flags the adapter as poisoned.
  * ASR (%)        -- attack success rate: how often the backdoor actually fires when run.
The point of the figure is the CONTRAST: detection falls to 0 while ASR stays high, i.e.
the detector clears backdoors that demonstrably still work.

    Scenario          Detection %   ASR %    Note
    ---------------------------------------------------------------------------
    Standard spiky        100        n/a     Detector works. ASR not plotted: this is the
                                             detector's own poison recipe, which is
                                             behaviorally hollow (only 15% of 400 adapters
                                             fire reliably), so a single bank-level ASR
                                             would misrepresent it. See Figure 4.
    Diffuse                21         74     Backdoor spread over all 36 layers.
    Dataset-match           0         51     Backdoor trained on the defender's own
                                             8-dataset benign mixture.
    CBA (published)         0         96     An off-the-shelf published attack we did not
                                             design (Chen et al., NDSS 2026).

IMPORTANT -- ASR is measured over WORKING backdoors only (ASR >= 0.5), not over the whole
bank. Bank-level means are lower because they average in adapters that never planted
(e.g. diffuse: working-only 0.74 vs bank-level 0.56). Keep the working-only reading if the
figure is redrawn; it is the number the paper's Table 3 reports.

All rates are percentages in [0, 100]. Detection is measured at the detector's deployed
operating point (layer 20, threshold tau = 0.5853).

PROVENANCE (where each number comes from)
-----------------------------------------
  diffuse    det 21%, ASR 74%  -> results/diffuse_eval_results.json (detection_rate 0.21)
                                  + results/asr_results.json (working-only mean 0.7363, n=73)
  dsmatch    det  0%, ASR 51%  -> results_c2/dsmatch_eval_results.json (detection_rate 0.0)
                                  + results_c2/dsmatch_asr_results.json (n_working = 52)
  CBA        det  0%, ASR 96%  -> results/cba_eval_pii-masker.json
  spiky      det 100%          -> calibration/reproduction run (AUC 1.00, 0% FPR)

RUN
---
    python figure5_for_redraw.py              # writes fig5_scenario_comparison.png
    python figure5_for_redraw.py --csv        # also writes fig5_data.csv
Requires only matplotlib + numpy.
"""

import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# THE DATA -- this is the whole figure. Edit here to redraw with new numbers.
# (scenario, detection_pct, asr_pct)   asr_pct = None means "not plotted"
# ---------------------------------------------------------------------------
DATA = [
    ("Standard\nspiky",   100, None),
    ("Diffuse",            21,   74),
    ("Dataset-\nmatch",     0,   51),
    ("CBA\n(published)",    0,   96),
]

# Colorblind-safe palette (Okabe-Ito), matching the rest of the paper's figures.
BLUE   = "#0072B2"   # detector / detection -- always blue in this paper
ORANGE = "#D55E00"   # spiky
GREEN  = "#009E73"   # diffuse
PURPLE = "#CC79A7"   # dataset-match
GREY   = "#7f7f7f"   # CBA / published / dead
INK    = "#1a1a1a"
MUTED  = "#666666"

# Each attack keeps its identity color; ASR is distinguished by hatch, not by color,
# because ASR is a metric rather than an attack.
ATTACK_COLOR = {
    "Standard\nspiky":   ORANGE,
    "Diffuse":           GREEN,
    "Dataset-\nmatch":   PURPLE,
    "CBA\n(published)":  GREY,
}
ASR_HATCH = "//"


def write_csv(path="fig5_data.csv"):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "detection_pct", "asr_pct"])
        for label, det, asr in DATA:
            # "Dataset-\nmatch" is hyphenated across the line break, so join on the
            # hyphen without a space; other labels wrap at a real word boundary.
            flat = label.replace("-\n", "-").replace("\n", " ")
            w.writerow([flat, det, "" if asr is None else asr])
    print(f"wrote {path}")


def make_figure(path="fig5_scenario_comparison.png"):
    labels = [d[0] for d in DATA]
    det    = [d[1] for d in DATA]
    asr    = [d[2] for d in DATA]

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.2, 3.9), dpi=220)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # Detection bars (left of each pair) -- always blue: it is a detector property.
    b1 = ax.bar(x - width / 2, det, width, color=BLUE, edgecolor="white", linewidth=0.6)

    # ASR bars (right of each pair) -- attack's own color + hatch. Skipped where None.
    idx = [i for i, a in enumerate(asr) if a is not None]
    b2 = ax.bar([x[i] + width / 2 for i in idx], [asr[i] for i in idx], width,
                color=[ATTACK_COLOR[labels[i]] for i in idx],
                hatch=ASR_HATCH, edgecolor="white", linewidth=0.6)

    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 1.5,
                    f"{r.get_height():.0f}", ha="center", fontsize=9, color=INK)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=BLUE, edgecolor="white", label="detector detection"),
        Patch(facecolor="#cccccc", hatch=ASR_HATCH, edgecolor="white",
              label="backdoor ASR (still works)"),
    ], frameon=False, fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 112)
    ax.set_ylabel("rate (%)", color=MUTED)
    ax.set_title("Detection collapses across attack scenarios; the backdoors keep working",
                 color=INK, fontsize=11)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"wrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", action="store_true", help="also write fig5_data.csv")
    args = ap.parse_args()

    make_figure()
    if args.csv:
        write_csv()
