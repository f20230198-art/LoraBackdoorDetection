#!/usr/bin/env python3
"""
make_aaai_figures.py — regenerate the paper's RESULT figures from the Drive JSONs.
==================================================================================

Reads the *_results.json files written by the AAAI-upgrade scripts and emits clean,
colorblind-safe PNGs (Okabe-Ito palette, direct labels, recessive axes). Skips any figure
whose input JSON isn't present yet, so it's safe to run at any stage.

Figures:
  fig_feature_space.png     signed mean shift vs benign per metric (opposite extremes),
                            from detector_head_ablation_results.json  (Q1)
  fig_c5_repair.png         two-sided (85% FPR, dead) vs supervised (96%, works),
                            from distribution_shift_results.json
  fig_multibackbone.png     spiky AUC 1.0 -> diffuse/dsmatch collapse, all 3 backbones
                            (values hardcoded from RESULTS_SUMMARY; edit MULTIBACKBONE below)

USAGE (CPU; Colab or local):
  LBD_MODEL=qwen LBD_OUTPUT_BASE=$DRIVE python plotScripts/make_aaai_figures.py \
      --results $DRIVE/results_aaai --out $DRIVE/results_aaai
"""

import os
import sys
import json
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Okabe-Ito colorblind-safe palette, fixed roles (match figures_tikz.tex).
C = {"benign": "#0072B2", "spiky": "#D55E00", "diffuse": "#009E73",
     "dsmatch": "#CC79A7", "warn": "#D55E00", "good": "#009E73", "ink": "#1a1a1a",
     "muted": "#666666"}
METRICS = ["sigma1", "frobenius", "energy", "entropy", "kurtosis"]
ATTACKS = ["spiky", "diffuse", "dsmatch"]

# Edit here if the multi-backbone numbers change.
MULTIBACKBONE = {  # backbone: (spiky_det%, diffuse_det%, dsmatch_det%)
    "Qwen2.5-3B": (100, 21, 0),
    "Gemma-2-2B": (100, 5.1, 0),
    "Llama-3.2-3B": (95, 0, 0),
}

# The headline "comparative results across scenarios" (prof: "comparative results ...
# in different scenarios"). detection% and ASR% per attack; ASR None = not applicable.
# Update the standard-spiky ASR from the working-spiky bank once Job 1 lands.
SCENARIOS = {  # scenario: (detection%, ASR%)
    "Standard\nspiky":  (100, None),   # detector works; standard bank is behaviorally hollow
    "Diffuse":          (21, 74),
    "Dataset-\nmatch":  (0, 51),
    "CBA\n(published)": (0, 96),
}


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#bbbbbb")
    ax.tick_params(colors=C["muted"], length=0)
    ax.yaxis.grid(True, color="#eeeeee", linewidth=1)
    ax.set_axisbelow(True)


def _load(results_dir, name):
    p = os.path.join(results_dir, name)
    if os.path.isfile(p):
        return json.load(open(p))
    print(f"  (skip: {name} not found in {results_dir})")
    return None


def fig_feature_space(results_dir, out):
    d = _load(results_dir, "detector_head_ablation_results.json")
    if not d or "signed_shift" not in d:
        return
    shift = d["signed_shift"]
    attacks = [a for a in ATTACKS if a in shift]
    x = np.arange(len(METRICS)); w = 0.8 / max(len(attacks), 1)
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=220)
    _style(ax)
    for i, a in enumerate(attacks):
        vals = [shift[a].get(m, 0) for m in METRICS]
        ax.bar(x + (i - (len(attacks) - 1) / 2) * w, vals, w, label=a,
               color=C.get(a, "#888"), edgecolor="white", linewidth=0.6)
    ax.axhline(0, color=C["ink"], linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels(METRICS)
    ax.set_ylabel("mean shift vs benign\n(benign-std units)", color=C["muted"])
    ax.set_title("Attacks occupy opposite regions of feature space (benign = 0)",
                 color=C["ink"], fontsize=11)
    ax.legend(frameon=False, fontsize=9, ncol=len(attacks))
    ax.annotate("energy & entropy flip sign:\nspiky concentrated, diffuse flat",
                xy=(2.0, 0), xytext=(2.4, 1.5), fontsize=8, color=C["muted"],
                arrowprops=dict(arrowstyle="->", color=C["muted"]))
    fig.tight_layout()
    p = os.path.join(out, "fig_feature_space.png"); fig.savefig(p, bbox_inches="tight")
    print(f"  wrote {p}")


def fig_c5_repair(results_dir, out):
    d = _load(results_dir, "distribution_shift_results.json")
    if not d:
        return
    A = d.get("section_A_two_sided", {})
    B = d.get("section_B_dataset_aware_leave_out", {})
    # two comparable detectors: dsmatch detection vs benign-FPR on UNSEEN datasets
    twosided_det = (A.get("detection_dsmatch") or 0) * 100
    twosided_fpr = (A.get("fpr_heldout_benign") or 0) * 100
    sup_det = (B.get("heldout_dsmatch_detection") or 0) * 100
    sup_fpr = 5.0  # supervised threshold set at 5% held-out-benign FPR by construction
    groups = ["Two-sided\n(unsupervised)", "Supervised\n(attack-aware)"]
    det = [twosided_det, sup_det]; fpr = [twosided_fpr, sup_fpr]
    x = np.arange(2); w = 0.36
    fig, ax = plt.subplots(figsize=(6.2, 3.8), dpi=220)
    _style(ax)
    b1 = ax.bar(x - w / 2, det, w, label="dsmatch detection", color=C["good"],
                edgecolor="white", linewidth=0.6)
    b2 = ax.bar(x + w / 2, fpr, w, label="benign FPR (unseen datasets)", color=C["warn"],
                edgecolor="white", linewidth=0.6)
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 1.5,
                    f"{r.get_height():.0f}%", ha="center", fontsize=9, color=C["ink"])
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylim(0, 105); ax.set_ylabel("rate (%)", color=C["muted"])
    ax.set_title("The repair, honestly: two-sided flags 85% of innocent unseen adapters;\n"
                 "supervised recovery generalizes across datasets at 5% FPR",
                 color=C["ink"], fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    p = os.path.join(out, "fig_c5_repair.png"); fig.savefig(p, bbox_inches="tight")
    print(f"  wrote {p}")


def fig_multibackbone(out):
    bbs = list(MULTIBACKBONE)
    series = [("spiky (AUC 1.0)", 0, C["spiky"]),
              ("diffuse", 1, C["diffuse"]),
              ("dsmatch", 2, C["dsmatch"])]
    x = np.arange(len(bbs)); w = 0.26
    fig, ax = plt.subplots(figsize=(7.0, 3.6), dpi=220)
    _style(ax)
    for lab, idx, col in series:
        vals = [MULTIBACKBONE[b][idx] for b in bbs]
        bars = ax.bar(x + (idx - 1) * w, vals, w, label=lab, color=col,
                      edgecolor="white", linewidth=0.6)
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 1.5,
                    f"{r.get_height():.0f}", ha="center", fontsize=8, color=C["ink"])
    ax.set_xticks(x); ax.set_xticklabels(bbs)
    ax.set_ylim(0, 112); ax.set_ylabel("detection rate (%)", color=C["muted"])
    ax.set_title("Same collapse on all three backbones: spiky caught, both attacks evade",
                 color=C["ink"], fontsize=11)
    ax.legend(frameon=False, fontsize=9, ncol=3)
    fig.tight_layout()
    p = os.path.join(out, "fig_multibackbone.png"); fig.savefig(p, bbox_inches="tight")
    print(f"  wrote {p}")


def fig_scenario_comparison(out):
    labels = list(SCENARIOS)
    det = [SCENARIOS[k][0] for k in labels]
    asr = [SCENARIOS[k][1] for k in labels]
    x = np.arange(len(labels)); w = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 3.9), dpi=220)
    _style(ax)
    b1 = ax.bar(x - w / 2, det, w, label="detector detection", color=C["spiky"],
                edgecolor="white", linewidth=0.6)
    ax_idx = [i for i, a in enumerate(asr) if a is not None]
    b2 = ax.bar([x[i] + w / 2 for i in ax_idx], [asr[i] for i in ax_idx], w,
                label="backdoor ASR (still works)", color=C["diffuse"],
                edgecolor="white", linewidth=0.6)
    for r in b1:
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 1.5, f"{r.get_height():.0f}",
                ha="center", fontsize=9, color=C["ink"])
    for r in b2:
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 1.5, f"{r.get_height():.0f}",
                ha="center", fontsize=9, color=C["ink"])
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 112); ax.set_ylabel("rate (%)", color=C["muted"])
    ax.set_title("Detection collapses across attack scenarios; the backdoors keep working",
                 color=C["ink"], fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    p = os.path.join(out, "fig_scenario_comparison.png"); fig.savefig(p, bbox_inches="tight")
    print(f"  wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    default_res = os.path.join(os.environ.get("LBD_OUTPUT_BASE", "."), "results_aaai")
    ap.add_argument("--results", default=default_res, help="dir with *_results.json")
    ap.add_argument("--out", default=default_res, help="dir to write PNGs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    print(f"[figures] reading {args.results}, writing {args.out}")
    fig_scenario_comparison(args.out)
    fig_feature_space(args.results, args.out)
    fig_c5_repair(args.results, args.out)
    fig_multibackbone(args.out)
    print("[figures] done.")


if __name__ == "__main__":
    main()
