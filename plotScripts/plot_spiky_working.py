#!/usr/bin/env python3
"""
Figure: spiky-working planting yield + function-gated detection (n=40).
Visualizes the audit finding that the AUC-1.00 poison recipe is behaviorally hollow:
most adapters never fire, and among the few that do, the detector calls them benign.

Reads the two JSONs produced by the 2026-07-10 run:
  - spiky_working_asr.json     (measure_asr.py; per_adapter list with 'adapter','asr')
  - spiky_working_scored.json  (evaluate_diffuse.py; per-adapter 'name','score')

Usage:
  python plotScripts/plot_spiky_working.py \
      --asr   output_qwen/results_aaai/spiky_working_asr.json \
      --score output_qwen/results_aaai/spiky_working_scored.json \
      --thr 0.585321 \
      --out literature/literatureReview/figures/fig_spiky_working.png
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito palette (matches the paper's TikZ / other figures).
BLUE, ORANGE, GREEN, GRAY = "#0072B2", "#D55E00", "#009E73", "#555555"


def per_adapter_list(d):
    if isinstance(d, list):
        return d
    for v in d.values():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return v
    return []


def base(r):
    p = r.get("adapter") or r.get("name") or r.get("path") or ""
    return os.path.basename(str(p).rstrip("/"))


def getnum(r, *keys):
    for k in keys:
        if k in r and r[k] is not None:
            return r[k]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr", required=True)
    ap.add_argument("--score", required=True)
    ap.add_argument("--thr", type=float, default=0.585321)
    ap.add_argument("--out", default="literature/literatureReview/figures/fig_spiky_working.png")
    a = ap.parse_args()

    asr_rows = per_adapter_list(json.load(open(a.asr)))
    score_rows = per_adapter_list(json.load(open(a.score)))
    asr_by = {base(r): getnum(r, "asr", "ASR") for r in asr_rows}
    score_by = {base(r): getnum(r, "score", "poison_score") for r in score_rows}

    names = sorted(asr_by)
    asrs = [asr_by[n] for n in names]
    working = [n for n in names if (asr_by[n] or 0) >= 0.5]
    caught = [n for n in working if (score_by.get(n) or 0) >= a.thr]

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.2, 2.9))

    # LEFT: ASR distribution — how little of the standard recipe fires.
    bins = [i / 20 for i in range(0, 21)]
    axL.hist(asrs, bins=bins, color=GRAY, edgecolor="white")
    axL.axvline(0.5, color=ORANGE, ls="--", lw=1.5, label="working (ASR$\\geq$0.5)")
    axL.set_xlabel("attack success rate (ASR)")
    axL.set_ylabel("adapters")
    axL.set_title(f"Planting yield: {len(working)}/{len(names)} fire reliably", fontsize=9)
    axL.legend(fontsize=7, frameon=False)

    # RIGHT: among working adapters, detector score vs threshold.
    wnames = working
    wscores = [score_by.get(n) or 0 for n in wnames]
    colors = [ORANGE if s >= a.thr else BLUE for s in wscores]
    axR.bar(range(len(wnames)), wscores, color=colors)
    axR.axhline(a.thr, color="black", ls="--", lw=1.2, label=f"threshold {a.thr:.3f}")
    axR.set_xticks(range(len(wnames)))
    axR.set_xticklabels([f"#{i+1}" for i in range(len(wnames))], fontsize=7)
    axR.set_ylabel("detector score")
    axR.set_ylim(0, 1)
    axR.set_title(f"Working backdoors: {len(caught)}/{len(working)} caught", fontsize=9)
    axR.legend(fontsize=7, frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    print(f"wrote {a.out}")
    print(f"n={len(names)}  fire>0: {sum((x or 0)>0 for x in asrs)}  "
          f"working: {len(working)}  caught among working: {len(caught)}/{len(working)}")


if __name__ == "__main__":
    main()
