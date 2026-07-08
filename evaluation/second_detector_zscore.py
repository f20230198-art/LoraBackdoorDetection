#!/usr/bin/env python3
"""
Rung-2 second detector — the z-score / bank-anomaly detector, run on the SAME banks.
====================================================================================

WHY (AAAI paradigm claim). A reviewer's first objection to "weight-space LoRA backdoor
detection is fragile as a paradigm" is: *you only broke ONE detector.* This script defeats
a SECOND, methodologically-distinct detector that already ships in this repo:

  * detector.py      : per-projection 20-dim features -> StandardScaler -> LogisticRegression
                       (a TRAINED discriminative head). This is the target we attack in C1-C4.
  * deep_scan.py     : per-layer z-score of each spectral metric against a BENIGN reference
                       bank -> tanh squash -> fixed-weight sum -> threshold (an ANOMALY head,
                       no trained classifier). Different aggregation, different calibration.

Both read the SAME spectral feature family (sigma1, Frobenius, energy-concentration, entropy,
kurtosis) at the SAME inspected layer (20, q/k/v/o). So if BOTH are perfect on standard spiky
poison and BOTH collapse on the diffuse / dataset-matching banks, the fragility is in the
weight-space *feature family*, not in the logistic regressor. That is the sentence that turns
"one detector" into "paradigm" (see contributions/C5_impossibility.md for why no head over
these features can do better).

WHAT IT DOES (CPU only, no GPU, runs on the banks you already have on Drive):
  1. Load (or build) the benign reference bank -> per-layer mean/std of the 5 metrics.
  2. Calibrate the z-score detector on benign (negative) vs standard spiky poison (positive):
     report ROC-AUC and pick a Youden-J threshold. (Sanity gate: AUC should be ~1.0, i.e. the
     second detector also catches standard spiky poison.)
  3. Score every attack bank with the SAME calibrated z-score detector and report detection
     rate + mean score + per-bank AUC vs benign.

USAGE (paths default to config dirs; pass extra banks as name:path):
  LBD_MODEL=qwen LBD_OUTPUT_BASE=$DRIVE python evaluation/second_detector_zscore.py \
      --bank diffuse:$DRIVE/diffuse_poison \
      --bank dsmatch:$DRIVE/dsmatch_poison \
      --bank spiky_working:$DRIVE/spiky_working_poison \
      --bank place4:$DRIVE/diffuse_poison_seed11 \
      --bank place8:$DRIVE/diffuse_poison_seed12 \
      --bank pr1:$DRIVE/diffuse_poison_seed13

Writes evaluation/second_detector_zscore_results.json (override with --out).
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from core.benign_bank import BenignBank
from core.deep_scan import DeepGeometricAnalysis
# reuse the EXACT delta-W reconstruction the reference bank was built with, so the
# second detector reads adapters identically to how its benign stats were computed.
from bankCreation.build_reference_bank import extract_delta_w, build_reference_bank


def _expanded_layers():
    """One layer index per (layer, module) matrix, matching extract_delta_w's order and the
    expanded_layer_indices the benign bank was built with."""
    return [layer for layer in config.TARGET_LAYERS for _ in config.TARGET_MODULES]


def load_or_build_bank() -> BenignBank:
    bank = BenignBank(config.BANK_FILE)
    if not bank.is_trained:
        print(f"[bank] {config.BANK_FILE} missing/empty — building from {config.BENIGN_DIR} ...")
        build_reference_bank()
        bank = BenignBank(config.BANK_FILE)
    if not bank.is_trained:
        sys.exit(f"Could not build a benign reference bank from {config.BENIGN_DIR}.")
    print(f"[bank] loaded reference stats for layers: {sorted(bank.layer_stats.keys())}")
    return bank


def is_adapter_dir(p: str) -> bool:
    return os.path.isfile(os.path.join(p, "adapter_config.json"))


def score_dir(bank: BenignBank, det: DeepGeometricAnalysis, bank_dir: str):
    """Return (names, scores) for every adapter under bank_dir the detector can read."""
    tl = _expanded_layers()
    names, scores = [], []
    if not os.path.isdir(bank_dir):
        return names, scores
    entries = [bank_dir] if is_adapter_dir(bank_dir) else [
        os.path.join(bank_dir, d) for d in sorted(os.listdir(bank_dir))
    ]
    for d in entries:
        if not is_adapter_dir(d):
            continue
        mats = extract_delta_w(d)
        if not mats:
            continue
        res = det.analyze(mats, target_layers=tl)
        if "score" in res:
            names.append(os.path.basename(d))
            scores.append(float(res["score"]))
    return names, scores


def summarize(name, scores, thr, benign_scores=None):
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    det_rate = float(np.mean(scores >= thr)) if n else 0.0
    row = {
        "bank": name,
        "n": int(n),
        "mean_score": float(np.mean(scores)) if n else None,
        "max_score": float(np.max(scores)) if n else None,
        "detection_at_thr": det_rate,
    }
    if benign_scores is not None and n:
        y = np.hstack([np.zeros(len(benign_scores)), np.ones(n)])
        s = np.hstack([benign_scores, scores])
        try:
            row["auc_vs_benign"] = float(roc_auc_score(y, s))
        except ValueError:
            row["auc_vs_benign"] = None
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", action="append", default=[],
                    help="extra attack bank as name:path (repeatable)")
    ap.add_argument("--out", default="evaluation/second_detector_zscore_results.json")
    args = ap.parse_args()

    bank = load_or_build_bank()
    det = DeepGeometricAnalysis(bank)   # default fixed feature weights; threshold set below

    # --- 1. negative class: benign; positive (calibration) class: standard spiky poison ---
    print("\n[calibrate] scoring benign + standard poison with the z-score detector ...")
    _, benign_scores = score_dir(bank, det, config.BENIGN_DIR)
    _, poison_scores = score_dir(bank, det, config.POISON_DIR)
    if not benign_scores or not poison_scores:
        sys.exit(f"Need both benign ({len(benign_scores)}) and poison ({len(poison_scores)}) "
                 f"scores to calibrate. Check {config.BENIGN_DIR} / {config.POISON_DIR}.")

    y = np.hstack([np.zeros(len(benign_scores)), np.ones(len(poison_scores))])
    s = np.hstack([benign_scores, poison_scores])
    cal_auc = float(roc_auc_score(y, s))
    fpr, tpr, thrs = roc_curve(y, s)
    thr = float(thrs[int(np.argmax(tpr - fpr))])   # Youden-J
    print(f"[calibrate] z-score detector AUC on benign vs standard spiky poison = {cal_auc:.4f}")
    print(f"[calibrate] Youden-J threshold = {thr:.4f}  "
          f"(benign mean {np.mean(benign_scores):.3f}, poison mean {np.mean(poison_scores):.3f})")
    print("[calibrate] SANITY: this AUC should be high (~1.0). If not, the 2nd detector never "
          "worked and downstream collapse is meaningless.")

    # --- 2. default banks + any passed on the CLI ---
    banks = {"benign(neg)": config.BENIGN_DIR, "standard_poison(pos)": config.POISON_DIR}
    if os.path.isdir(config.DIFFUSE_POISON_DIR):
        banks["diffuse"] = config.DIFFUSE_POISON_DIR
    if os.path.isdir(config.DSMATCH_POISON_DIR):
        banks["dsmatch"] = config.DSMATCH_POISON_DIR
    for spec in args.bank:
        if ":" not in spec:
            print(f"  (skipping malformed --bank {spec!r}; want name:path)")
            continue
        nm, pth = spec.split(":", 1)
        banks[nm] = pth

    rows = [summarize("calibration", poison_scores, thr, benign_scores)]
    rows[-1]["bank"] = "standard_poison (calibration, AUC={:.4f})".format(cal_auc)

    print("\n" + "=" * 78)
    print(f"{'bank':28s} {'n':>4s} {'mean':>7s} {'max':>7s} {'det@thr':>8s} {'AUCvsBenign':>12s}")
    print("-" * 78)
    results = {"calibration_auc": cal_auc, "threshold": thr, "banks": []}
    for nm, pth in banks.items():
        names, sc = score_dir(bank, det, pth)
        if not sc:
            print(f"{nm:28s}  (no scorable adapters at {pth})")
            continue
        row = summarize(nm, sc, thr, benign_scores if "benign" not in nm else None)
        results["banks"].append({**row, "path": pth})
        auc = row.get("auc_vs_benign")
        print(f"{nm:28s} {row['n']:>4d} {row['mean_score']:>7.3f} {row['max_score']:>7.3f} "
              f"{row['detection_at_thr']*100:>7.1f}% "
              f"{(f'{auc:.4f}' if auc is not None else '   --   '):>12s}")
    print("=" * 78)
    print("READING: standard poison should be ~100% det / AUC~1.0; diffuse & dsmatch should")
    print("collapse. Same collapse as detector.py -> the failure is the feature family, not")
    print("the logistic head (paradigm evidence, second detector).")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
