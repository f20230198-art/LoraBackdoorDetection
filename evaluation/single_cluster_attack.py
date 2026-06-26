#!/usr/bin/env python3
"""
Single-Poison-Cluster Attack — C2 sub-attack #2 (calibration poisoning)
=======================================================================

Weaponizes C1 *Finding D* (core/detector.py:44-59, `_select_threshold`). The
detector picks its decision threshold with a two-mode rule:

  * if benign and poison scores PERFECTLY SEPARATE on the calibration fold, it
    drops the threshold in the empty gap: benign_max + 0.25 * (poison_min - benign_max)
    (mode "perfect_separation_margin"). This clean gap is WHY the paper reports
    a perfect 100% / AUC 1.00.
  * otherwise it falls back to Youden's J on the ROC (mode "youden_j").

THE ATTACK (calibration poisoning, no GPU, no new training):
An attacker who can contribute even ONE poison-like adapter to the detector's
*reference / calibration pool* — realistic, since such pools are routinely
scraped from public LoRA hubs — labels it "benign" (it is uploaded as an
ordinary adapter). That single high-scoring point sits inside the benign cloud,
so benign and poison no longer separate. The perfect-separation rule can no
longer fire; the threshold falls back to a looser Youden cut. We then measure
the resulting drop in held-out detection rate: a real backdoor at test time now
slips under the degraded threshold.

We do NOT reimplement the detector. We reuse the production machinery
(`_extract_features_from_adapter`, `_select_threshold`, StandardScaler +
LogisticRegression with the same C / class_weight / split) so the attack is
faithful to the calibrated detector, not a toy of it. Feature extraction (the
expensive QR/SVD part) runs ONCE; the k-sweep then recalibrates in-memory.

Honesty note (C0): we are NOT first to attack a weight-space detector
(PEFTGuard did noise/FGSM/PGD/C&W). This is a calibration-side data-poisoning
attack specific to THIS detector's post-hoc threshold rule, reported with the
held-out detection drop it causes — no cherry-picking.

Usage:
  python evaluation/single_cluster_attack.py [--k_max 5] [--seeds 5] [--run_dir runs/...]
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# MUST precede core.detector / config (import-time CUDA call). See _env_fix.py.
import _env_fix  # noqa: F401

from core.detector import BackdoorDetector
import config


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


def get_adapter_paths(directory: str, type_filter: str):
    """Same selection logic as calibrate_detector.py — metadata 'type' must match."""
    base_path = Path(config.ROOT_DIR) / directory
    if not base_path.exists():
        return []
    valid = []
    for d in sorted(base_path.iterdir()):
        if not d.is_dir():
            continue
        meta = d / "metadata.json"
        if not meta.exists():
            continue
        with open(meta, "r") as f:
            if json.load(f).get("type") == type_filter:
                valid.append(str(d))
    return valid


def extract_bank(paths, layer_idx: int, label: str):
    """Run the detector's real feature extractor once over a bank of adapters."""
    feats, kept = [], []
    for i, p in enumerate(paths, 1):
        f = BackdoorDetector._extract_features_from_adapter(Path(p), layer_idx)
        if f is not None:
            feats.append(f)
            kept.append(p)
        if i % 25 == 0 or i == len(paths):
            log(f"  {label}: extracted {len(feats)}/{i}")
    return feats, kept


def stratified_split(n_pos, n_neg, val_split, rng):
    """Mirror BackdoorDetector.calibrate()'s stratified per-class split."""
    pos_idx = np.arange(n_pos)
    neg_idx = np.arange(n_neg)
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    def _split(idx):
        n_tr = max(1, int(round(len(idx) * (1 - val_split))))
        n_tr = min(n_tr, len(idx) - 1) if len(idx) > 1 else len(idx)
        return idx[:n_tr], idx[n_tr:]

    return _split(pos_idx), _split(neg_idx)


def calibrate_in_memory(benign_feats, poison_feats, rng, C=0.1, val_split=0.2):
    """
    Faithful in-memory replay of BackdoorDetector.calibrate(): same stratified
    split, StandardScaler, LogisticRegression(C, class_weight='balanced'), and
    the same _select_threshold rule. Returns the fitted pieces + threshold info.
    """
    X = np.vstack(benign_feats + poison_feats)
    y = np.hstack([np.zeros(len(benign_feats)), np.ones(len(poison_feats))])

    (pos_tr, pos_val), (neg_tr, neg_val) = stratified_split(
        len(poison_feats), len(benign_feats), val_split, rng
    )
    # poison rows live after benign rows in X
    pos_off = len(benign_feats)
    train_idx = np.concatenate([neg_tr, pos_tr + pos_off])
    val_idx = np.concatenate([neg_val, pos_val + pos_off])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    clf = LogisticRegression(C=C, max_iter=1000, class_weight="balanced", random_state=42)
    clf.fit(X_train_s, y_train)

    y_val_proba = clf.predict_proba(X_val_s)[:, 1]
    threshold, mode = BackdoorDetector._select_threshold(y_val, y_val_proba)

    auc = (
        roc_auc_score(y_val, y_val_proba)
        if len(np.unique(y_val)) == 2
        else float("nan")
    )
    return {
        "clf": clf,
        "scaler": scaler,
        "threshold": float(threshold),
        "threshold_mode": mode,
        "val_auc": float(auc),
    }


def held_out_detection(model, test_poison_feats, test_benign_feats):
    """Detection rate (poison caught) and FPR on held-out test, given a fitted model."""
    clf, scaler, thr = model["clf"], model["scaler"], model["threshold"]

    def _rate(feats):
        if not feats:
            return None, []
        Xs = scaler.transform(np.vstack(feats))
        scores = clf.predict_proba(Xs)[:, 1]
        return scores, scores

    p_scores, _ = _rate(test_poison_feats)
    b_scores, _ = _rate(test_benign_feats)

    detection = float(np.mean(p_scores >= thr)) if p_scores is not None else None
    fpr = float(np.mean(b_scores >= thr)) if b_scores is not None else None
    return {
        "detection_rate": detection,
        "fpr": fpr,
        "mean_poison_score": float(np.mean(p_scores)) if p_scores is not None else None,
        "mean_benign_score": float(np.mean(b_scores)) if b_scores is not None else None,
    }


def main():
    ap = argparse.ArgumentParser(description="Single-poison-cluster (calibration poisoning) attack")
    ap.add_argument("--k_max", type=int, default=5,
                    help="Max number of poison adapters injected into the benign/reference pool")
    ap.add_argument("--seeds", type=int, default=5,
                    help="Random seeds per k (different injected adapters + split) for mean/std")
    ap.add_argument("--run_dir", type=str, default=None,
                    help="Output dir for the result JSON (default: runs/single_cluster_<ts>)")
    args = ap.parse_args()

    layer_idx = config.TARGET_LAYERS[0]
    log("=" * 70)
    log("SINGLE-POISON-CLUSTER ATTACK (C2 sub-attack #2: calibration poisoning)")
    log(f"Target layer {layer_idx} | k_max={args.k_max} | seeds={args.seeds}")
    log("=" * 70)

    benign_paths = get_adapter_paths(config.BENIGN_DIR, "benign")
    poison_paths = get_adapter_paths(config.POISON_DIR, "poison")
    test_benign_paths = sorted(
        str(d) for d in (Path(config.ROOT_DIR) / config.TEST_SET_DIR).glob("test_benign_*")
        if d.is_dir()
    ) if (Path(config.ROOT_DIR) / config.TEST_SET_DIR).exists() else []
    test_poison_paths = sorted(
        str(d) for d in (Path(config.ROOT_DIR) / config.TEST_SET_DIR).glob("test_poison_*")
        if d.is_dir()
    ) if (Path(config.ROOT_DIR) / config.TEST_SET_DIR).exists() else []

    log(f"Banks: benign={len(benign_paths)} poison={len(poison_paths)} "
        f"test_benign={len(test_benign_paths)} test_poison={len(test_poison_paths)}")
    if not benign_paths or not poison_paths:
        log("ERROR: need both benign and poison banks. Aborting.")
        return
    if not test_poison_paths:
        log("WARNING: no held-out test poison adapters — detection-drop will be skipped.")

    # --- Feature extraction ONCE (the expensive QR/SVD part) ---
    log("Extracting features (once)...")
    benign_feats, _ = extract_bank(benign_paths, layer_idx, "benign")
    poison_feats, _ = extract_bank(poison_paths, layer_idx, "poison")
    test_poison_feats, _ = extract_bank(test_poison_paths, layer_idx, "test_poison")
    test_benign_feats, _ = extract_bank(test_benign_paths, layer_idx, "test_benign")

    if not benign_feats or not poison_feats:
        log("ERROR: feature extraction yielded no usable vectors (shape/proj mismatch?).")
        return

    # The injected adapters are drawn from the SAME poison bank but RELABELED
    # benign (the attacker uploads a poison adapter as an ordinary one). We hold
    # out the injected ones from the poison pool for that calibration so we never
    # double-count a single adapter as both benign-injected and poison-known.
    results_per_k = []
    for k in range(0, args.k_max + 1):
        seed_runs = []
        for s in range(args.seeds):
            rng = np.random.default_rng(1000 * k + s)
            if k == 0:
                inj_idx = np.array([], dtype=int)
            else:
                if k > len(poison_feats) - 1:
                    break  # need at least 1 poison left as "known poison"
                inj_idx = rng.choice(len(poison_feats), size=k, replace=False)

            keep_mask = np.ones(len(poison_feats), dtype=bool)
            keep_mask[inj_idx] = False
            cal_poison = [poison_feats[i] for i in np.where(keep_mask)[0]]
            injected = [poison_feats[i] for i in inj_idx]
            cal_benign = benign_feats + injected  # injected poison relabeled benign

            model = calibrate_in_memory(cal_benign, cal_poison, rng)
            ho = held_out_detection(model, test_poison_feats, test_benign_feats)
            seed_runs.append({
                "threshold": model["threshold"],
                "threshold_mode": model["threshold_mode"],
                "val_auc": model["val_auc"],
                **ho,
            })

        if not seed_runs:
            continue

        def _agg(key):
            vals = [r[key] for r in seed_runs if r[key] is not None and not (isinstance(r[key], float) and np.isnan(r[key]))]
            return (float(np.mean(vals)), float(np.std(vals))) if vals else (None, None)

        modes = [r["threshold_mode"] for r in seed_runs]
        frac_perfect = float(np.mean([m == "perfect_separation_margin" for m in modes]))
        det_mean, det_std = _agg("detection_rate")
        thr_mean, thr_std = _agg("threshold")
        auc_mean, auc_std = _agg("val_auc")
        fpr_mean, _ = _agg("fpr")

        row = {
            "k_injected": k,
            "n_seeds": len(seed_runs),
            "frac_perfect_separation": frac_perfect,
            "frac_youden_fallback": 1.0 - frac_perfect,
            "threshold_mean": thr_mean, "threshold_std": thr_std,
            "val_auc_mean": auc_mean, "val_auc_std": auc_std,
            "held_out_detection_mean": det_mean, "held_out_detection_std": det_std,
            "held_out_fpr_mean": fpr_mean,
        }
        results_per_k.append(row)
        log(f"k={k}: mode_perfect={frac_perfect:.2f} thr={thr_mean:.4f} "
            f"val_auc={auc_mean:.4f} held-out detection={('NA' if det_mean is None else f'{det_mean*100:.1f}%')}")

    # --- Summarize the headline (k=0 vs first k that breaks perfect separation) ---
    baseline = next((r for r in results_per_k if r["k_injected"] == 0), None)
    broke = next((r for r in results_per_k
                  if r["frac_perfect_separation"] < 1.0), None)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "model": config.MODEL,
        "model_name": config.MODEL_NAME,
        "layer_idx": layer_idx,
        "finding": "C1 Finding D — post-hoc perfect_separation_margin threshold",
        "attack": "single_poison_cluster_calibration_poisoning",
        "k_max": args.k_max,
        "seeds": args.seeds,
        "banks": {
            "benign": len(benign_feats), "poison": len(poison_feats),
            "test_poison": len(test_poison_feats), "test_benign": len(test_benign_feats),
        },
        "baseline_k0": baseline,
        "first_break": broke,
        "sweep": results_per_k,
    }

    run_dir = Path(args.run_dir) if args.run_dir else (
        Path(config.ROOT_DIR) / config.RUNS_DIR / f"single_cluster_{int(time.time())}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "single_cluster_results.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    log("=" * 70)
    if baseline and baseline["held_out_detection_mean"] is not None:
        log(f"BASELINE (k=0): threshold mode perfect={baseline['frac_perfect_separation']:.2f}, "
            f"held-out detection {baseline['held_out_detection_mean']*100:.1f}%")
    if broke and broke["held_out_detection_mean"] is not None:
        log(f"BROKEN  (k={broke['k_injected']}): perfect-separation fires only "
            f"{broke['frac_perfect_separation']*100:.0f}% of seeds, held-out detection "
            f"{broke['held_out_detection_mean']*100:.1f}%")
    log(f"Result JSON: {out}")
    log("=" * 70)


if __name__ == "__main__":
    main()
