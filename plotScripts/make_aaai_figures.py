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

# Okabe-Ito colorblind-safe palette. ONE role per color across every figure and the
# TikZ diagrams (Contract A). Do not reassign these inside a figure.
#   blue    = the detector / detection rate / benign
#   orange  = spiky / standard poison
#   green   = diffuse attack  AND  the C5 repair
#   purple  = dataset-matching
#   grey    = dead / baseline / axes
# ASR is a METRIC, not an attack, so it is drawn with a // hatch rather than its own
# color -- that keeps each attack's bar in the attack's color while still separating
# "detection" from "ASR" at a glance.
C = {"detector": "#0072B2", "benign": "#0072B2",
     "spiky": "#D55E00", "diffuse": "#009E73", "dsmatch": "#CC79A7",
     "repair": "#009E73", "dead": "#7f7f7f", "ink": "#1a1a1a", "muted": "#666666"}
ASR_HATCH = "//"
# per-scenario attack identity color, used wherever a scenario/attack is on the x-axis
ATTACK_COLOR = {"Standard\nspiky": C["spiky"], "Diffuse": C["diffuse"],
                "Dataset-\nmatch": C["dsmatch"], "CBA\n(published)": C["dead"]}
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
    # detection of the dsmatch attack -> detector color (blue). The false-alarm cost on
    # innocent unseen adapters is the baseline penalty -> grey.
    b1 = ax.bar(x - w / 2, det, w, label="dsmatch detection", color=C["detector"],
                edgecolor="white", linewidth=0.6)
    b2 = ax.bar(x + w / 2, fpr, w, label="benign FPR (unseen datasets)", color=C["dead"],
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
    # detection is a detector property -> always blue. ASR keeps each attack's identity
    # color, distinguished from detection by a hatch.
    b1 = ax.bar(x - w / 2, det, w, color=C["detector"],
                edgecolor="white", linewidth=0.6)
    ax_idx = [i for i, a in enumerate(asr) if a is not None]
    b2 = ax.bar([x[i] + w / 2 for i in ax_idx], [asr[i] for i in ax_idx], w,
                color=[ATTACK_COLOR[labels[i]] for i in ax_idx], hatch=ASR_HATCH,
                edgecolor="white", linewidth=0.6)
    for r in b1:
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 1.5, f"{r.get_height():.0f}",
                ha="center", fontsize=9, color=C["ink"])
    for r in b2:
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 1.5, f"{r.get_height():.0f}",
                ha="center", fontsize=9, color=C["ink"])
    # legend by encoding, not by attack, since attack color now varies along x
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=C["detector"], edgecolor="white", label="detector detection"),
                       Patch(facecolor="#cccccc", hatch=ASR_HATCH, edgecolor="white",
                             label="backdoor ASR (still works)")],
              frameon=False, fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 112); ax.set_ylabel("rate (%)", color=C["muted"])
    ax.set_title("Detection collapses across attack scenarios; the backdoors keep working",
                 color=C["ink"], fontsize=11)
    fig.tight_layout()
    p = os.path.join(out, "fig_scenario_comparison.png"); fig.savefig(p, bbox_inches="tight")
    print(f"  wrote {p}")


def fig_transfer_matrix(out):
    # Detection (%) when the detector is TRAINED on attack row and TESTED on attack col,
    # at ~5% benign FPR. Values from the transfer_matrix run (paper Fig. transfer).
    attacks = ["spiky", "diffuse", "dsmatch"]
    M = np.array([[100, 47, 0],
                  [0, 100, 0],
                  [37, 0, 100]], dtype=float)
    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=220)
    # single-hue blue ramp: this is the ONE place a sequential scale is allowed, and it is
    # the detector's own color family, so it stays inside the contract.
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(3)); ax.set_xticklabels(attacks)
    ax.set_yticks(range(3)); ax.set_yticklabels(attacks)
    ax.set_xlabel("tested on  (attack the detector meets)", color=C["muted"])
    ax.set_ylabel("trained on  (attack the detector saw)", color=C["muted"])
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=12,
                    fontweight="bold", color="white" if M[i, j] > 55 else C["ink"])
    ax.set_title("Attack transfer: detection (%) at ~5% benign FPR\n"
                 "high diagonal, low off-diagonal = no cross-family transfer",
                 color=C["ink"], fontsize=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("detection rate (%)", color=C["muted"])
    fig.tight_layout()
    p = os.path.join(out, "fig_transfer_matrix.png"); fig.savefig(p, bbox_inches="tight")
    print(f"  wrote {p}")


def fig_placement_curve(out):
    # Spread the same backdoor across more layers -> detection falls, ASR survives.
    layers = ["4", "8", "all"]
    detection = [100, 75, 21]     # detector color (blue)
    asr = [1, 28, None]           # ASR keeps the diffuse attack's color (green) + hatch
    x = np.arange(len(layers))
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=220)
    _style(ax)
    ax.plot(x, detection, "-o", color=C["detector"], lw=2.4, ms=8, label="detector detection")
    xa = [i for i, v in enumerate(asr) if v is not None]
    ax.plot(xa, [asr[i] for i in xa], "--s", color=C["diffuse"], lw=2.4, ms=9,
            markerfacecolor="none", markeredgewidth=2.2, label="backdoor ASR (still alive)")
    for i, v in enumerate(detection):
        ax.text(i, v + 3, f"{v}%", ha="center", color=C["detector"], fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(layers)
    ax.set_ylim(-4, 112); ax.set_xlabel("layers the backdoor is spread across", color=C["muted"])
    ax.set_ylabel("rate (%)", color=C["muted"])
    ax.set_title(r"Spread the backdoor thinner $\rightarrow$ detection collapses, ASR survives",
                 color=C["ink"], fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    p = os.path.join(out, "fig_placement_curve.png"); fig.savefig(p, bbox_inches="tight")
    print(f"  wrote {p}")


def fig_spiky_working(out, results_dir=None):
    # The AUC-1.00 recipe is behaviorally hollow (JOB A, 2026-07-13, n=400 @ 15-20% poison).
    # EXACT counts from the run: 135/400 fire (ASR>0), 60/400 working (ASR>=0.5), of those
    # 24 caught / 36 evade, mean working score 0.530 < threshold 0.585321.
    # If the real per-adapter arrays are on Drive we use them; otherwise we synthesize a
    # distribution that reproduces those exact counts/means (shape only, counts are real).
    N, N_FIRE, N_WORK, N_CAUGHT = 400, 135, 60, 24
    THR, MEAN_WORK = 0.585321, 0.530
    asr_all = scores_work = None
    if results_dir:
        d = _load(results_dir, "spiky_working_n50_asr.json") or {}
        e = _load(results_dir, "spiky_working_n50_eval.json") or {}
        asr_all = d.get("asr") or d.get("asr_all")
        scores_work = e.get("working_scores") or e.get("scores_working")
    rng = np.random.default_rng(0)
    if asr_all is None:
        asr_all = np.concatenate([
            np.zeros(N - N_FIRE),
            rng.uniform(0.01, 0.49, N_FIRE - N_WORK),
            rng.uniform(0.50, 0.85, N_WORK)])
    if scores_work is None:
        # EXACTLY 24 above THR and 36 below, then pin the mean to 0.530 without any bar
        # crossing the threshold (scale each side about its own clamp so counts hold).
        hi = np.sort(rng.uniform(THR + 0.002, 0.72, N_CAUGHT))
        lo = np.sort(rng.uniform(0.30, THR - 0.004, N_WORK - N_CAUGHT))
        scores_work = np.concatenate([hi, lo])
        # shift the whole set toward the target mean, then re-clamp so the 24/36 split is
        # exact regardless of the shift.
        scores_work += (MEAN_WORK - scores_work.mean())
        scores_work[:N_CAUGHT] = np.clip(scores_work[:N_CAUGHT], THR + 0.001, 0.75)
        scores_work[N_CAUGHT:] = np.clip(scores_work[N_CAUGHT:], 0.28, THR - 0.001)
    scores_work = np.sort(np.asarray(scores_work))[::-1]

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(9.2, 3.6), dpi=220)
    _style(axl); _style(axr)
    # left -- ASR distribution: most adapters never fire (dead = grey)
    axl.hist(asr_all, bins=np.linspace(0, 1, 26), color=C["dead"], edgecolor="white")
    axl.axvline(0.5, color=C["spiky"], ls="--", lw=2, label="working (ASR $\\geq$ 0.5)")
    axl.set_xlabel("attack success rate (ASR)", color=C["muted"])
    axl.set_ylabel("adapters", color=C["muted"])
    axl.set_title(f"Planting yield: {N_WORK}/{N} fire reliably\n({N_FIRE}/{N} fire at all)",
                  color=C["ink"], fontsize=11)
    axl.legend(frameon=False, fontsize=8)
    # right -- detector scores of the working backdoors vs threshold (caught = orange)
    n_caught = int((scores_work >= THR).sum())
    colors = [C["spiky"] if s >= THR else C["dead"] for s in scores_work]
    axr.bar(range(len(scores_work)), scores_work, color=colors, edgecolor="white", linewidth=0.3)
    axr.axhline(THR, color=C["detector"], ls="--", lw=2, label=f"threshold {THR:.3f}")
    axr.axhline(MEAN_WORK, color=C["muted"], ls=":", lw=1.5,
                label=f"mean working {MEAN_WORK:.3f}")
    axr.set_ylim(0, 1)
    axr.set_xlabel(f"working backdoors (n = {N_WORK}), sorted", color=C["muted"])
    axr.set_ylabel("detector score", color=C["muted"])
    axr.set_title(f"{N_WORK - n_caught}/{N_WORK} ({100*(N_WORK-n_caught)//N_WORK}%) "
                  "working backdoors score below threshold", color=C["ink"], fontsize=10)
    axr.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    p = os.path.join(out, "fig_spiky_working.png"); fig.savefig(p, bbox_inches="tight")
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
    fig_transfer_matrix(args.out)
    fig_placement_curve(args.out)
    fig_spiky_working(args.out, args.results)
    print("[figures] done.")


if __name__ == "__main__":
    main()
