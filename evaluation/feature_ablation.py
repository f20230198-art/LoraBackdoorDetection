#!/usr/bin/env python3
"""
Per-feature ablation — the evidence for Assumption A1 (the impossibility result's premise).
==========================================================================================

WHY. contributions/C5_impossibility.md proves that no detector built from the per-layer spectral
features can recover the diffuse attack — but only UNDER Assumption A1: that a working diffuse
backdoor's features, at the inspected layer, look benign (small separation). A1 is EMPIRICAL. This
script measures it directly, and as a bonus answers "which of the 5 statistics dies under which
attack."

WHAT IT DOES (CPU only, on the banks you already have). For each of the detector's 5 spectral
statistics (sigma1, Frobenius norm, energy-concentration, entropy, kurtosis), averaged over the
q/k/v/o projections at layer 20, it computes the *orientation-free univariate ROC-AUC* separating
benign adapters from each poison bank — exactly the target paper's own feature-diagnostic style
(main.tex, univariate U_m), so it is a like-for-like measurement.

HOW TO READ IT.
  * Standard spiky poison: each statistic should separate WELL (AUC near 1.0) -> the detector has
    real single-feature signal on spiky poison. (Sanity: this is WHY it scores AUC 1.0.)
  * Diffuse / dataset-matching: the statistics should collapse toward AUC ~0.5 (no separation) ->
    the features look benign -> Assumption A1 holds -> the impossibility result applies.
A near-0.5 row for the diffuse bank is the receipt that "the signal is not in these features."

USAGE:
  LBD_MODEL=qwen LBD_OUTPUT_BASE=$DRIVE python evaluation/feature_ablation.py \
      --bank spiky_poison:$DRIVE/poison \
      --bank diffuse:$DRIVE/diffuse_poison \
      --bank dsmatch:$DRIVE/dsmatch_poison \
      --bank place4:$DRIVE/diffuse_poison_seed11 \
      --bank place8:$DRIVE/diffuse_poison_seed12
(benign is always taken from config.BENIGN_DIR as the negative class.)
Writes evaluation/feature_ablation_results.json (override with --out).
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from core.detector import BackdoorDetector

METRIC_NAMES = ["sigma1", "frobenius", "energy", "entropy", "kurtosis"]


def is_adapter_dir(p: str) -> bool:
    return os.path.isfile(os.path.join(p, "adapter_config.json"))


def feature_matrix(bank_dir: str, layer: int) -> np.ndarray:
    """Stack the detector's own feature vector (5 metrics x #projections) for every adapter
    under bank_dir. Returns (N, 5*P) or an empty array."""
    rows = []
    if not os.path.isdir(bank_dir):
        return np.empty((0, 0))
    entries = [bank_dir] if is_adapter_dir(bank_dir) else [
        os.path.join(bank_dir, d) for d in sorted(os.listdir(bank_dir))
    ]
    for d in entries:
        if not is_adapter_dir(d):
            continue
        feat = BackdoorDetector._extract_features_from_adapter(Path(d), layer)
        if feat is not None:
            rows.append(feat)
    return np.vstack(rows) if rows else np.empty((0, 0))


def per_metric(F: np.ndarray) -> np.ndarray:
    """Average each of the 5 metrics across the P projections -> (N, 5)."""
    n, D = F.shape
    P = D // 5
    cols = {m: [m + 5 * p for p in range(P)] for m in range(5)}
    return np.column_stack([F[:, cols[m]].mean(axis=1) for m in range(5)])


def orientation_free_auc(neg: np.ndarray, pos: np.ndarray) -> float:
    """Univariate AUC that ignores sign (max(auc, 1-auc)), matching the target's U_m."""
    y = np.hstack([np.zeros(len(neg)), np.ones(len(pos))])
    x = np.hstack([neg, pos])
    if len(np.unique(x)) < 2:
        return 0.5
    try:
        a = roc_auc_score(y, x)
    except ValueError:
        return 0.5
    return float(max(a, 1.0 - a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", action="append", default=[],
                    help="poison bank as name:path (repeatable)")
    ap.add_argument("--layer", type=int, default=config.TARGET_LAYERS[0])
    ap.add_argument("--out", default="evaluation/feature_ablation_results.json")
    args = ap.parse_args()

    print(f"[ablation] benign negative class: {config.BENIGN_DIR}  (layer {args.layer})")
    Fb = feature_matrix(config.BENIGN_DIR, args.layer)
    if Fb.size == 0:
        sys.exit(f"No benign features from {config.BENIGN_DIR}.")
    Mb = per_metric(Fb)
    print(f"[ablation] benign n={len(Fb)}")

    banks = {}
    # sensible defaults if none passed
    if not args.bank:
        for nm, pth in [("spiky_poison", config.POISON_DIR),
                        ("diffuse", config.DIFFUSE_POISON_DIR),
                        ("dsmatch", config.DSMATCH_POISON_DIR)]:
            if os.path.isdir(pth):
                banks[nm] = pth
    for spec in args.bank:
        if ":" in spec:
            nm, pth = spec.split(":", 1)
            banks[nm] = pth

    results = {"layer": args.layer, "benign_n": int(len(Fb)), "banks": {}}
    header = f"{'metric':12s}" + "".join(f"{nm[:12]:>13s}" for nm in banks)
    print("\n" + "=" * len(header))
    print("Orientation-free univariate AUC (benign vs bank).  ~1.0 = separates,  ~0.5 = blind.")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    per_bank_metric = {}
    for nm, pth in banks.items():
        Fp = feature_matrix(pth, args.layer)
        if Fp.size == 0:
            per_bank_metric[nm] = None
            continue
        Mp = per_metric(Fp)
        aucs = [orientation_free_auc(Mb[:, m], Mp[:, m]) for m in range(5)]
        per_bank_metric[nm] = aucs
        results["banks"][nm] = {"path": pth, "n": int(len(Fp)),
                                "auc_per_metric": dict(zip(METRIC_NAMES, aucs))}

    for m, met in enumerate(METRIC_NAMES):
        line = f"{met:12s}"
        for nm in banks:
            a = per_bank_metric.get(nm)
            line += f"{(f'{a[m]:.3f}' if a else '  --'):>13s}"
        print(line)
    # a compact "mean over the 5 metrics" row = a one-number separability summary per bank
    print("-" * len(header))
    line = f"{'MEAN':12s}"
    for nm in banks:
        a = per_bank_metric.get(nm)
        line += f"{(f'{np.mean(a):.3f}' if a else '  --'):>13s}"
    print(line)
    print("=" * len(header))
    print("A1 CHECK: diffuse / dsmatch columns near 0.5 = features look benign = the diffuse")
    print("signal is NOT in these statistics = the impossibility premise (A1) holds.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
