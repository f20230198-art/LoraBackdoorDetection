#!/usr/bin/env python3
"""
Detector-head ablation — does the collapse survive changing the CLASSIFIER, and WHY?
====================================================================================

Replaces the broken hand-weighted z-score script (second_detector_zscore.py failed its own
sanity gate: AUC 0.39 on standard poison — it never separated, so its diffuse/dsmatch numbers
were meaningless). This script asks the questions that output actually raised:

  Q1 (DIRECTION). The feature ablation showed diffuse features are HIGHLY separable from benign
      (AUC 0.945) yet the detector misses them. Are spiky poison and diffuse poison on OPPOSITE
      sides of benign? -> signed standardized mean shift per metric.
  Q2 (DIFFERENT HEAD). Keep the SAME 20-dim spectral features; swap the logistic head for a
      Random Forest trained the SAME way (benign vs spiky poison). If it ALSO aces spiky and
      ALSO misses diffuse/dsmatch, the failure is the feature family + one-sided training, not
      the logistic regressor -> paradigm evidence.
  Q3 (TWO-SIDED). A Mahalanobis "distance from benign in ANY direction" detector (unsupervised,
      two-sided) — does it catch diffuse where the one-sided detectors fail? This is the likely
      FIX direction and an honest C5 defense lead.
  Q4 (IS THE SIGNAL THERE?). Train a detector DIRECTLY on benign-vs-diffuse (and benign-vs-dsmatch).
      If AUC ~1.0, the diffuse signal IS in the features — so the target's miss is a
      GENERALIZATION / one-sidedness failure, NOT an information-theoretic impossibility. This is
      the honest reframe of the (now-retracted) "features look benign" assumption.

CPU only, on the banks you already have. Uses the SAME feature extractor the attacked detector
uses (core.detector.BackdoorDetector._extract_features_from_adapter), so it reads adapters
identically.

USAGE:
  LBD_MODEL=qwen LBD_OUTPUT_BASE=$DRIVE python evaluation/detector_head_ablation.py
  (defaults: benign, poison(spiky), diffuse_poison, dsmatch_poison; override paths with --bank name:path)
Writes evaluation/detector_head_ablation_results.json (override with --out).
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from core.detector import BackdoorDetector

METRIC_NAMES = ["sigma1", "frobenius", "energy", "entropy", "kurtosis"]
RS = 42


def is_adapter_dir(p: str) -> bool:
    return os.path.isfile(os.path.join(p, "adapter_config.json"))


def feature_matrix(bank_dir: str, layer: int) -> np.ndarray:
    rows = []
    if not os.path.isdir(bank_dir):
        return np.empty((0, 0))
    entries = [bank_dir] if is_adapter_dir(bank_dir) else [
        os.path.join(bank_dir, d) for d in sorted(os.listdir(bank_dir))
    ]
    for d in entries:
        if not is_adapter_dir(d):
            continue
        f = BackdoorDetector._extract_features_from_adapter(Path(d), layer)
        if f is not None:
            rows.append(f)
    return np.vstack(rows) if rows else np.empty((0, 0))


def detection_rate(scores, thr):
    return float(np.mean(np.asarray(scores) >= thr)) if len(scores) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", action="append", default=[], help="name:path (repeatable)")
    ap.add_argument("--layer", type=int, default=config.TARGET_LAYERS[0])
    ap.add_argument("--out", default="evaluation/detector_head_ablation_results.json")
    args = ap.parse_args()
    L = args.layer

    banks = {"benign": config.BENIGN_DIR, "spiky": config.POISON_DIR,
             "diffuse": config.DIFFUSE_POISON_DIR, "dsmatch": config.DSMATCH_POISON_DIR}
    for spec in args.bank:
        if ":" in spec:
            nm, pth = spec.split(":", 1); banks[nm] = pth

    print(f"[extract] layer {L} features for: {list(banks)}")
    F = {nm: feature_matrix(p, L) for nm, p in banks.items()}
    for nm in list(F):
        if F[nm].size == 0:
            print(f"  (dropping {nm}: no adapters at {banks[nm]})"); F.pop(nm)
    if "benign" not in F or "spiky" not in F:
        sys.exit("Need benign + spiky poison to run.")
    Xb, Xs = F["benign"], F["spiky"]
    print("  n: " + ", ".join(f"{k}={len(v)}" for k, v in F.items()))

    scaler = StandardScaler().fit(Xb)          # standardize on benign
    Z = {nm: scaler.transform(X) for nm, X in F.items()}
    results = {"layer": L, "n": {k: int(len(v)) for k, v in F.items()}}

    # ---- Q1: signed direction (pooled per metric), in benign-std units --------------------
    P = Xb.shape[1] // 5
    def pooled(Zx):  # (N,20)->(N,5) average the 4 projections per metric
        return np.column_stack([Zx[:, [m + 5 * p for p in range(P)]].mean(1) for m in range(5)])
    print("\n" + "=" * 66)
    print("Q1  Signed mean shift vs benign (benign-std units).  + = higher than benign")
    print("=" * 66)
    print(f"{'metric':12s}" + "".join(f"{nm[:9]:>11s}" for nm in F if nm != "benign"))
    dir_tbl = {}
    for m, met in enumerate(METRIC_NAMES):
        line = f"{met:12s}"
        for nm in F:
            if nm == "benign":
                continue
            shift = float(pooled(Z[nm])[:, m].mean())
            dir_tbl.setdefault(nm, {})[met] = shift
            line += f"{shift:>+11.2f}"
        print(line)
    results["signed_shift"] = dir_tbl
    print("READ: spiky should be + (spikier); diffuse should be - (flatter, opposite side).")

    # ---- Q2: Random Forest head, trained the SAME way (benign vs spiky) --------------------
    Xtr = np.vstack([Z["benign"], Z["spiky"]])
    ytr = np.hstack([np.zeros(len(Z["benign"])), np.ones(len(Z["spiky"]))])
    Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.3, stratify=ytr, random_state=RS)
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RS).fit(Xa, ya)
    pv = rf.predict_proba(Xv)[:, 1]
    rf_auc = roc_auc_score(yv, pv)
    # threshold at ~5% FPR on held-out benign
    vb = pv[yv == 0]
    thr = float(np.quantile(vb, 0.95)) if len(vb) else 0.5

    print("\n" + "=" * 66)
    print(f"Q2  Random-Forest head (SAME 20-dim features, trained benign vs spiky)")
    print(f"    sanity: held-out spiky AUC = {rf_auc:.4f}   thr@~5%FPR = {thr:.3f}")
    print("=" * 66)
    rf_rows = {}
    for nm in F:
        if nm == "benign":
            det = detection_rate(rf.predict_proba(Z[nm])[:, 1], thr); tag = "(FPR)"
        else:
            det = detection_rate(rf.predict_proba(Z[nm])[:, 1], thr); tag = ""
        rf_rows[nm] = det
        print(f"  {nm:10s} detection {det*100:5.1f}% {tag}")
    results["rf"] = {"sanity_auc": float(rf_auc), "thr": thr, "detection": rf_rows}
    print("READ: if RF also aces spiky but misses diffuse/dsmatch -> not the logistic's fault.")

    # ---- Q3: two-sided Mahalanobis distance from benign (unsupervised) ---------------------
    mu = Z["benign"].mean(0)
    cov = np.cov(Z["benign"], rowvar=False) + 1e-6 * np.eye(Z["benign"].shape[1])
    Cinv = np.linalg.pinv(cov)
    def maha(Zx):
        d = Zx - mu
        return np.einsum("ij,jk,ik->i", d, Cinv, d)  # squared distance
    mb = maha(Z["benign"])
    mthr = float(np.quantile(mb, 0.95))  # 5% benign FPR
    print("\n" + "=" * 66)
    print(f"Q3  Two-sided Mahalanobis distance from benign (unsupervised)  thr@5%FPR={mthr:.1f}")
    print("=" * 66)
    maha_rows = {}
    for nm in F:
        det = detection_rate(maha(Z[nm]), mthr)
        maha_rows[nm] = det
        print(f"  {nm:10s} detection {det*100:5.1f}%" + ("  (FPR)" if nm == "benign" else ""))
    results["mahalanobis"] = {"thr": mthr, "detection": maha_rows}
    print("READ: if this CATCHES diffuse where the 1-sided heads miss -> the fix is 2-sided,")
    print("      and 'impossibility' becomes 'linear/one-sided detectors can't catch both'.")

    # ---- Q4: is the signal THERE? train directly on benign-vs-attack ----------------------
    print("\n" + "=" * 66)
    print("Q4  Separability if you TRAIN ON THE ATTACK directly (held-out AUC)")
    print("=" * 66)
    sep = {}
    for nm in F:
        if nm in ("benign", "spiky"):
            continue
        X = np.vstack([Z["benign"], Z[nm]])
        y = np.hstack([np.zeros(len(Z["benign"])), np.ones(len(Z[nm]))])
        Xa2, Xv2, ya2, yv2 = train_test_split(X, y, test_size=0.3, stratify=y, random_state=RS)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xa2, ya2)
        auc = roc_auc_score(yv2, clf.predict_proba(Xv2)[:, 1])
        sep[nm] = float(auc)
        print(f"  benign-vs-{nm:10s} held-out AUC = {auc:.4f}")
    results["separable_if_trained_on_attack"] = sep
    print("READ: AUC ~1.0 => the signal IS in the features. The target misses it only because")
    print("      it trained on spiky poison => a GENERALIZATION failure, not impossibility.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
