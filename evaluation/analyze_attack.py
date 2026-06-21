#!/usr/bin/env python3
"""
Joint ASR x evasion analysis for the diffuse attack — the paper's headline figure.
=================================================================================

Two means (mean ASR, mean evasion) under-sell the result because ~18% of adapters
have ASR=0 (the backdoor never planted at certain lr/batch combos) and drag the mean
ASR down. The honest, stronger framing is the JOINT distribution:

  - Among adapters whose backdoor ACTUALLY WORKS (ASR >= asr_min), what fraction evade
    the unchanged detector? That isolates "successful stealthy backdoor", which is the
    quantity the threat model cares about.

Joins evaluation/asr_results.json with evaluation/diffuse_eval_results.json per adapter,
prints the breakdown, and writes a scatter (ASR vs detector score, threshold line) to
evaluation/attack_scatter.png.

Usage:
  python evaluation/analyze_attack.py [--asr_min 0.5]
  python evaluation/analyze_attack.py --asr asr_results.json --eval diffuse_eval_results.json
"""

import os
import sys
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr", default="evaluation/asr_results.json")
    ap.add_argument("--eval", default="evaluation/diffuse_eval_results.json")
    ap.add_argument("--asr_min", type=float, default=0.5,
                    help="ASR threshold for 'backdoor actually works' (default 0.5)")
    ap.add_argument("--plot", default="evaluation/attack_scatter.png")
    args = ap.parse_args()

    asr_data = load(args.asr)
    eval_data = load(args.eval)
    threshold = eval_data["threshold"]

    asr_by = {r["adapter"]: r["asr"] for r in asr_data["per_adapter"]}
    rows = []
    for r in eval_data["per_adapter"]:
        name = r["name"]
        if name not in asr_by:
            continue
        rows.append({
            "name": name,
            "asr": asr_by[name],
            "score": r["score"],
            "evaded": r["evaded"],
        })

    n = len(rows)
    asr = np.array([r["asr"] for r in rows])
    score = np.array([r["score"] for r in rows])
    evaded = np.array([r["evaded"] for r in rows])

    working = asr >= args.asr_min
    dead = asr == 0.0

    print("=" * 64)
    print("DIFFUSE ATTACK — JOINT ASR x EVASION ANALYSIS")
    print("=" * 64)
    print(f"Adapters:                  {n}")
    print(f"Detector threshold:        {threshold:.4f}")
    print("-" * 64)
    print(f"Mean ASR (all):            {asr.mean():.3f}")
    print(f"Mean ASR (working only):   {asr[working].mean():.3f}   "
          f"[{working.sum()} adapters with ASR>={args.asr_min}]")
    print(f"Dead adapters (ASR=0):     {int(dead.sum())}  "
          f"(backdoor never planted — lr/batch artifact)")
    print("-" * 64)
    print(f"Overall evasion (all):     {evaded.mean()*100:.1f}%")
    if working.sum():
        print(f"Evasion among WORKING:     {evaded[working].mean()*100:.1f}%   "
              f"<-- headline: stealthy AND functional")
    print(f"Mean detector score (all): {score.mean():.3f}  (threshold {threshold:.3f})")
    if working.sum():
        print(f"Mean score (working):      {score[working].mean():.3f}")
    print("-" * 64)
    # The money cell of the 2x2: high-ASR AND evaded.
    strong = working & (evaded == 1)
    print(f"STRONG attacks (ASR>={args.asr_min} AND evaded): "
          f"{int(strong.sum())}/{n}  ({strong.sum()/n*100:.1f}% of all adapters)")
    print("=" * 64)

    # Scatter: ASR vs detector score, colored by evaded, threshold line.
    plt.figure(figsize=(7, 5))
    ev = evaded == 1
    plt.scatter(score[ev], asr[ev], c="green", alpha=0.7, label="evaded (called benign)")
    plt.scatter(score[~ev], asr[~ev], c="red", alpha=0.7, label="caught")
    plt.axvline(threshold, color="black", linestyle="--", label=f"threshold={threshold:.3f}")
    plt.axhline(args.asr_min, color="gray", linestyle=":", alpha=0.6,
                label=f"ASR={args.asr_min} (working)")
    plt.xlabel("Detector poison score  (left of line = evades)")
    plt.ylabel("Attack Success Rate")
    plt.title("Diffuse attack: backdoor strength vs detector score")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.plot, dpi=150)
    print(f"Wrote {args.plot}")


if __name__ == "__main__":
    main()
