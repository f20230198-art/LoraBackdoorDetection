"""Generate the paper figures that can be built from the local result JSONs
(no GPU). Addresses REVIEW_FINDINGS P2-5 (more figures) and P1-5 (threshold-free
view) and P1-3 (C3 lambda is optimization noise, not a trade-off curve).

Outputs (300 dpi PNG, single-column IEEE width) next to paper_final.tex:
  fig_threshold_sweep.png      detection vs threshold for both C2 attacks
  fig_c3_lambda.png            C3 lambda sweep (non-monotonic, n=1)
  fig_dsmatch_perdataset.png   dataset-matching per-dataset ASR spread

Usage:  python plotScripts/make_review_figures.py [--drive DIR] [--out DIR]
The default --drive points at the gitignored local Drive dump; override if the
folder name differs. Colours are colourblind-safe (Okabe-Ito) and legible in
grayscale.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Okabe-Ito colourblind-safe palette
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"
GREY = "#555555"

THRESHOLDS = {"target ($0.417$)": 0.417,
              "repro ($0.501$)": 0.501,
              "attack-eval ($0.585$)": 0.585}


def _load(drive, rel):
    with open(os.path.join(drive, rel)) as f:
        return json.load(f)


def _join(eval_j, asr_j):
    asr = {a["adapter"]: a["asr"] for a in asr_j["per_adapter"]}
    scores = np.array([e["score"] for e in eval_j["per_adapter"]])
    asrs = np.array([asr.get(e["name"], np.nan) for e in eval_j["per_adapter"]])
    return scores, asrs


def fig_threshold_sweep(drive, out):
    d_s, d_a = _join(_load(drive, "results/diffuse_eval_results.json"),
                     _load(drive, "results/asr_results.json"))
    m_s, m_a = _join(_load(drive, "results_c2/dsmatch_eval_results.json"),
                     _load(drive, "results_c2/dsmatch_asr_results.json"))
    taus = np.linspace(0.30, 0.70, 200)

    def det(scores, taus):
        return np.array([(scores >= t).mean() * 100 for t in taus])

    d_work = d_s[d_a >= 0.5]
    m_work = m_s[m_a >= 0.5]

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.plot(taus, det(d_s, taus), color=BLUE, lw=1.8, label="Diffuse (all)")
    ax.plot(taus, det(d_work, taus), color=BLUE, lw=1.4, ls="--",
            label="Diffuse (working)")
    ax.plot(taus, det(m_s, taus), color=VERMILLION, lw=1.8,
            label="Dataset-match (all)")
    ax.plot(taus, det(m_work, taus), color=VERMILLION, lw=1.4, ls="--",
            label="Dataset-match (working)")
    for lab, t in THRESHOLDS.items():
        ax.axvline(t, color=GREY, lw=0.8, ls=":")
        ax.text(t, 101, lab.split(" ")[0], rotation=90, fontsize=5.5,
                va="bottom", ha="center", color=GREY)
    ax.set_xlabel(r"detection threshold $\tau$")
    ax.set_ylabel("detection rate (%)")
    ax.set_ylim(-3, 108)
    ax.set_xlim(0.30, 0.70)
    ax.legend(fontsize=5.8, loc="upper left", framealpha=0.9)
    ax.set_title("Detection is low across every threshold used", fontsize=7)
    fig.tight_layout()
    p = os.path.join(out, "fig_threshold_sweep.png")
    fig.savefig(p, dpi=300)
    plt.close(fig)
    print("wrote", p)


def fig_c3_lambda(drive, out):
    c3 = _load(drive, "results_c3/c3_results.json")
    sweep = c3["per_adapter"][0]["lambda_sweep"]
    lam = [s["lambda"] for s in sweep]
    asr = [s["asr"] for s in sweep]
    logit = [s["surrogate_logit_after"] for s in sweep]
    x = np.arange(len(lam))

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    bars = ax.bar(x, asr, color=BLUE, width=0.6, label="ASR after evasion")
    ax.axhline(0.5, color=VERMILLION, lw=1.0, ls="--", label="ASR working bar")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l:g}" for l in lam])
    ax.set_xlabel(r"fidelity weight $\lambda$")
    ax.set_ylabel("ASR (real detector score $\\approx 0$ at all $\\lambda$)")
    ax.set_ylim(0, 0.6)
    for xi, a, lg in zip(x, asr, logit):
        ax.text(xi, a + 0.01, f"logit\n{lg:.0f}", ha="center", va="bottom",
                fontsize=5.0, color=GREY)
    ax.legend(fontsize=6, loc="upper right")
    ax.set_title("C3 ($n{=}1$): evasion is free, ASR is non-monotonic in $\\lambda$",
                 fontsize=6.5)
    fig.tight_layout()
    p = os.path.join(out, "fig_c3_lambda.png")
    fig.savefig(p, dpi=300)
    plt.close(fig)
    print("wrote", p)


def fig_dsmatch_perdataset(drive, out):
    per = _load(drive, "results_c2/dsmatch_asr_results.json")["per_adapter"]
    by = {}
    for a in per:
        by.setdefault(a["dataset"], []).append(a["asr"])
    items = sorted(by.items(), key=lambda kv: np.mean(kv[1]))
    names = [k.split("/")[-1] for k, _ in items]
    means = [float(np.mean(v)) for _, v in items]
    colors = [VERMILLION if m < 0.05 else BLUE for m in means]

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    y = np.arange(len(names))
    ax.barh(y, means, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel("mean ASR")
    ax.set_xlim(0, 1.05)
    ax.axvline(0.5, color=GREY, lw=0.8, ls=":")
    for yi, m in zip(y, means):
        ax.text(m + 0.02, yi, f"{m:.2f}", va="center", fontsize=5.5)
    ax.set_title("Dataset-matching: honest per-dataset planting spread",
                 fontsize=7)
    fig.tight_layout()
    p = os.path.join(out, "fig_dsmatch_perdataset.png")
    fig.savefig(p, dpi=300)
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", default=os.path.join(
        repo, "drive-download-20260704T074054Z-3-001"))
    ap.add_argument("--out", default=os.path.join(
        repo, "literature", "literatureReview"))
    args = ap.parse_args()
    fig_threshold_sweep(args.drive, args.out)
    fig_c3_lambda(args.drive, args.out)
    fig_dsmatch_perdataset(args.drive, args.out)
