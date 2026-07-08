#!/usr/bin/env python3
"""
Attack transfer matrix — "train the detector on attack X, can it catch attack Y?"
=================================================================================

The one-figure proof of the paper's mechanism (distribution mismatch): a weight-space
detector trained to catch one attack family does NOT transfer to the others, because each
attack lives in a different region of spectral-feature space (spiky = too concentrated,
diffuse = too flat, dsmatch = benign-shaped/bigger). The diagonal is high (train on X,
catch X); the off-diagonal collapses (no cross-family transfer). That is exactly why the
target detector (trained on spiky poison) is blind to diffuse and dsmatch.

For each TRAIN attack A: fit a logistic head on benign_train vs A_train (the same 20-dim
spectral features the target uses), set the threshold at ~5% FPR on held-out benign, then
measure detection on the held-out slice of every TEST attack B. Rows = trained on, cols =
tested on.

Outputs: a JSON matrix and a publication-grade heatmap PNG (sequential single hue, cells
directly labeled, colorblind-safe by monotonic lightness).

USAGE (CPU, on existing banks):
  LBD_MODEL=qwen LBD_OUTPUT_BASE=$DRIVE python evaluation/transfer_matrix.py \
      --out $DRIVE/results_aaai/transfer_matrix.json \
      --fig $DRIVE/results_aaai/fig_transfer_matrix.png
Add banks with --bank name:path (e.g. place4, spiky_working) once they exist.
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from core.detector import BackdoorDetector

RS = 42


def is_adapter_dir(p): return os.path.isfile(os.path.join(p, "adapter_config.json"))


def feats(bank_dir, layer):
    rows = []
    if not os.path.isdir(bank_dir):
        return np.empty((0, 0))
    for name in sorted(os.listdir(bank_dir)):
        d = os.path.join(bank_dir, name)
        if is_adapter_dir(d):
            f = BackdoorDetector._extract_features_from_adapter(Path(d), layer)
            if f is not None:
                rows.append(f)
    return np.vstack(rows) if rows else np.empty((0, 0))


def plot_heatmap(M, labels, fpr_col, fig_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(labels)
    fig, ax = plt.subplots(figsize=(1.15 * n + 2.2, 1.05 * n + 1.6), dpi=220)
    # sequential single hue, light->dark = detection magnitude (CVD-safe by lightness)
    cmap = plt.cm.Blues
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=100, aspect="equal")

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("tested on  (attack the detector meets)", fontsize=11)
    ax.set_ylabel("trained on  (attack the detector saw)", fontsize=11)
    ax.set_title("Attack transfer: detection rate (%) at ~5% benign FPR",
                 fontsize=12, pad=12)

    # direct cell labels; text color by cell luminance for contrast (accessibility)
    for i in range(n):
        for j in range(n):
            v = M[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    color=("white" if v >= 55 else "#1a1a1a"),
                    fontsize=11, fontweight="bold")
    # recessive frame + thin separating grid
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("detection rate (%)", fontsize=10)
    cbar.outline.set_visible(False)

    # benign FPR annotation per row (sanity: ~5% each)
    txt = "  ".join(f"{lab}:{f:.0f}%" for lab, f in zip(labels, fpr_col))
    fig.text(0.5, 0.005, f"benign FPR per trained detector — {txt}",
             ha="center", fontsize=8, color="#555")

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    os.makedirs(os.path.dirname(fig_path) or ".", exist_ok=True)
    fig.savefig(fig_path, bbox_inches="tight")
    print(f"[fig] wrote {fig_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", action="append", default=[], help="name:path (repeatable)")
    ap.add_argument("--layer", type=int, default=config.TARGET_LAYERS[0])
    ap.add_argument("--out", default="evaluation/transfer_matrix.json")
    ap.add_argument("--fig", default="evaluation/fig_transfer_matrix.png")
    args = ap.parse_args()
    L = args.layer

    attacks = {"spiky": config.POISON_DIR, "diffuse": config.DIFFUSE_POISON_DIR,
               "dsmatch": config.DSMATCH_POISON_DIR}
    for spec in args.bank:
        if ":" in spec:
            nm, pth = spec.split(":", 1); attacks[nm] = pth

    print(f"[extract] layer {L}: benign + {list(attacks)}")
    Xb = feats(config.BENIGN_DIR, L)
    if Xb.size == 0:
        sys.exit(f"No benign features at {config.BENIGN_DIR}")
    A = {nm: feats(p, L) for nm, p in attacks.items()}
    A = {nm: X for nm, X in A.items() if X.size}   # drop empty banks
    labels = list(A)
    print("  n: benign={}, ".format(len(Xb)) + ", ".join(f"{k}={len(v)}" for k, v in A.items()))

    # fixed train/test split per bank (reused for every trained detector)
    b_tr, b_te = train_test_split(Xb, test_size=0.3, random_state=RS)
    splits = {nm: train_test_split(X, test_size=0.3, random_state=RS) for nm, X in A.items()}

    n = len(labels)
    M = np.zeros((n, n))
    fpr_col = np.zeros(n)
    results = {"layer": L, "labels": labels, "matrix_detection_pct": {}, "benign_fpr_pct": {}}

    for i, train_a in enumerate(labels):
        a_tr, _ = splits[train_a]
        scaler = StandardScaler().fit(b_tr)
        Xtr = np.vstack([scaler.transform(b_tr), scaler.transform(a_tr)])
        ytr = np.hstack([np.zeros(len(b_tr)), np.ones(len(a_tr))])
        clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr, ytr)
        # threshold @ ~5% FPR on held-out benign
        pb = clf.predict_proba(scaler.transform(b_te))[:, 1]
        thr = float(np.quantile(pb, 0.95))
        fpr_col[i] = float(np.mean(pb >= thr) * 100)
        row = {}
        for j, test_b in enumerate(labels):
            _, b_test_slice = splits[test_b]
            pp = clf.predict_proba(scaler.transform(b_test_slice))[:, 1]
            M[i, j] = float(np.mean(pp >= thr) * 100)
            row[test_b] = M[i, j]
        results["matrix_detection_pct"][train_a] = row
        results["benign_fpr_pct"][train_a] = fpr_col[i]

    # console view
    print("\n" + "=" * (14 + 9 * n))
    print("TRANSFER MATRIX — detection % (rows=trained on, cols=tested on), thr@5% benign FPR")
    print("=" * (14 + 9 * n))
    print(f"{'train\\test':13s}" + "".join(f"{l[:8]:>9s}" for l in labels) + "   benignFPR")
    for i, l in enumerate(labels):
        print(f"{l:13s}" + "".join(f"{M[i, j]:>8.0f}%" for j in range(n)) + f"   {fpr_col[i]:>6.0f}%")
    print("=" * (14 + 9 * n))
    print("READ: high diagonal, low off-diagonal => each attack must be trained on; the")
    print("      detector does NOT transfer across attack families (distribution mismatch).")

    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")
    try:
        plot_heatmap(M, labels, fpr_col, args.fig)
    except Exception as e:
        print(f"[fig] skipped ({e}); JSON still written.")


if __name__ == "__main__":
    main()
