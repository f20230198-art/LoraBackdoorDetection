#!/usr/bin/env python3
"""
C5 defense, piece 2 — Dataset-aware calibration (counters the C2 dataset-matching attack).
==========================================================================================

WHY. C1 showed the detector is partly a *dataset-distribution* detector: a narrow benign
reference flagged adapters trained on unusual data (FPR 54%, AUC 0.76); diversifying it
collapsed FPR to 0%. C2's dataset-matching attack weaponizes exactly that — it trains the
backdoor on the SAME 8-dataset mixture as the benign reference, so the poison looks
distributionally "normal" and the unchanged detector scores 0% detection / 100% evasion
among the 52 working backdoors.

THE REPAIR. The baseline detector is calibrated on SPIKY poison only, so it never learns
what a dataset-matched poison looks like. Here we fold a TRAIN SLICE of the dataset-matching
bank into the calibration set (alongside the original spiky poison + benign), recalibrate
the SAME logistic detector, and then measure detection on a HELD-OUT slice of dsmatch poison
the classifier never saw. Train/test split on the dsmatch bank is mandatory — reporting
detection on adapters we calibrated on would be the exact in-sample inflation C1 criticizes.

HONESTY (C0). We report the RESIDUAL GAP, never a restored 100%:
  - dsmatch detection on the HELD-OUT slice (how much the repair recovers),
  - the FPR cost on benign test (dataset-aware calibration usually trades some false
    positives — report it),
  - and we DO NOT touch the diffuse attack here (that's piece 1). A defense that needs the
    attack's own data to be calibrated is itself a finding: weight-space detection of
    distribution-matched poison requires anticipating the attack distribution.

This is CPU-only (feature extraction + logistic regression). It reuses the production
detector machinery (the same BackdoorDetector.calibrate / _features / _select_threshold),
so it is faithful to the deployed detector, not a toy reimplementation. It honours
LBD_DETECTOR_LAYERS too, so you can stack piece-1 pooling + piece-2 dataset-aware
calibration (the full C5 detector) by setting that env var before running this.

Usage:
  python evaluation/c5_dataset_aware_calibrate.py \
      [--dsmatch_dir output_qwen/dsmatch_poison] \
      [--dsmatch_train_frac 0.5] \
      [--run_dir runs/run_c5_<ts>] \
      [--asr_json results_c2/dsmatch_asr_results.json]   # gate held-out detection to working backdoors

Outputs (under the run dir): classifier.pkl (the recalibrated C5 detector),
c5_dataset_aware_report.json (the residual-gap numbers + the held-out dsmatch list so the
split is auditable).
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# MUST precede core.detector / config (import-time CUDA call). See _env_fix.py.
import _env_fix  # noqa: F401

from core.detector import BackdoorDetector
import config


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


def adapter_dirs(directory: str, type_filter: str | None = None):
    """Adapter subdirs under `directory` (resolved against ROOT_DIR if relative).
    If type_filter is given, keep only those whose metadata.json type matches."""
    base = Path(directory)
    if not base.is_absolute():
        base = Path(config.ROOT_DIR) / base
    if not base.exists():
        return []
    out = []
    for d in sorted(base.iterdir()):
        if not d.is_dir() or not (d / "adapter_config.json").exists():
            continue
        if type_filter is not None:
            meta = d / "metadata.json"
            if not meta.exists():
                continue
            try:
                if json.load(open(meta)).get("type") != type_filter:
                    continue
            except Exception:
                continue
        out.append(str(d))
    return out


def load_working_set(asr_json: str | None, threshold: float = 0.5) -> set | None:
    """Names of dsmatch adapters whose ASR >= threshold (the only ones whose detection
    number is meaningful — a backdoor that doesn't fire isn't 'evading'). None = no gate."""
    if not asr_json:
        return None
    p = Path(asr_json)
    if not p.is_absolute():
        p = Path(config.ROOT_DIR) / p
    if not p.exists():
        log(f"WARNING: ASR json {p} not found — held-out detection will NOT be gated to working backdoors.")
        return None
    data = json.load(open(p))
    working = {r["adapter"] for r in data.get("per_adapter", []) if r.get("asr", 0) >= threshold}
    log(f"ASR gate: {len(working)} dsmatch adapters with ASR>={threshold} (working backdoors).")
    return working


def score_bank(detector: BackdoorDetector, paths, layer_idx: int):
    """Return per-adapter (name, score) using the calibrated detector."""
    out = []
    for ap in paths:
        res = detector.scan(ap, use_fast_scan=False, layer_idx=layer_idx)
        if "error" in res or res.get("score") is None:
            continue
        out.append((Path(ap).name, float(res["score"])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsmatch_dir", default=config.DSMATCH_POISON_DIR)
    ap.add_argument("--dsmatch_train_frac", type=float, default=0.5,
                    help="fraction of the dsmatch bank folded INTO calibration; the rest is held out for the residual-gap measurement")
    ap.add_argument("--asr_json", default=None,
                    help="dsmatch ASR results; if given, held-out detection is gated to ASR>=0.5 working backdoors (the meaningful number)")
    ap.add_argument("--run_dir", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    layer_idx = config.TARGET_LAYERS[0]
    layers, pool = BackdoorDetector._multilayer_config()
    if layers is not None:
        log(f"[C5] stacking piece-1 multi-layer pooling: layers={layers} pool={pool}")

    # --- gather banks ---
    benign = adapter_dirs(config.BENIGN_DIR, "benign")
    spiky = adapter_dirs(config.POISON_DIR, "poison")
    dsmatch = adapter_dirs(args.dsmatch_dir)  # dsmatch metadata type may differ; take all
    if not benign or not spiky:
        log("ERROR: need both benign and spiky-poison banks for the baseline calibration.")
        sys.exit(1)
    if not dsmatch:
        log(f"ERROR: no dsmatch adapters under {args.dsmatch_dir}.")
        sys.exit(1)
    log(f"Banks: benign={len(benign)} spiky-poison={len(spiky)} dsmatch={len(dsmatch)}")

    # --- split the dsmatch bank: train slice (folded into calibration) vs held-out ---
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(dsmatch))
    rng.shuffle(idx)
    n_train = int(round(len(dsmatch) * args.dsmatch_train_frac))
    n_train = min(max(n_train, 1), len(dsmatch) - 1)  # keep >=1 on each side
    dm_train = [dsmatch[i] for i in idx[:n_train]]
    dm_held = [dsmatch[i] for i in idx[n_train:]]
    log(f"dsmatch split: {len(dm_train)} folded into calibration, {len(dm_held)} held out for the residual-gap measurement.")

    # --- recalibrate: poison = spiky + dsmatch-train, benign unchanged ---
    run_dir = Path(args.run_dir) if args.run_dir else (
        Path(config.ROOT_DIR) / config.RUNS_DIR / f"run_c5_dataset_aware_{int(time.time())}")
    run_dir.mkdir(parents=True, exist_ok=True)

    detector = BackdoorDetector()
    poison_paths = spiky + dm_train
    log(f"Calibrating C5 dataset-aware detector: {len(benign)} benign vs "
        f"{len(poison_paths)} poison ({len(spiky)} spiky + {len(dm_train)} dsmatch).")
    calib = detector.calibrate(poison_paths, benign, layer_idx=layer_idx, random_state=args.seed)
    detector.save(str(run_dir / "classifier.pkl"))
    log(f"Saved recalibrated detector to {run_dir / 'classifier.pkl'} "
        f"(threshold {detector.threshold:.6f}, val AUC {calib['auc']:.4f}, mode {calib.get('threshold_mode')}).")

    # --- the residual-gap measurement: held-out dsmatch + benign test FPR ---
    working = load_working_set(args.asr_json)

    held_scores = score_bank(detector, dm_held, layer_idx)
    if working is not None:
        held_working = [(n, s) for (n, s) in held_scores if n in working]
    else:
        held_working = held_scores  # ungated
    det_all = sum(s >= detector.threshold for _, s in held_scores) / len(held_scores) if held_scores else 0.0
    det_working = (sum(s >= detector.threshold for _, s in held_working) / len(held_working)
                   if held_working else 0.0)

    # benign test FPR (the cost of the repair). Test benign adapters live in TEST_SET_DIR
    # as test_benign_*; fall back to scoring the benign bank tail if no test set present.
    from evaluation.evaluate_test_set import get_test_paths  # reuse the production glob
    benign_test = get_test_paths(config.TEST_SET_DIR, 50, pattern="test_benign_*")
    benign_test_scores = score_bank(detector, benign_test, layer_idx) if benign_test else []
    fpr = (sum(s >= detector.threshold for _, s in benign_test_scores) / len(benign_test_scores)
           if benign_test_scores else None)

    report = {
        "timestamp": datetime.now().isoformat(),
        "defense": "C5_dataset_aware_calibration",
        "multilayer": {"layers": layers, "pool": pool} if layers is not None else None,
        "threshold": float(detector.threshold),
        "threshold_mode": calib.get("threshold_mode"),
        "val_auc": float(calib["auc"]),
        "counts": {
            "benign_calib": len(benign),
            "spiky_poison_calib": len(spiky),
            "dsmatch_train_folded_in": len(dm_train),
            "dsmatch_held_out": len(dm_held),
            "dsmatch_held_out_working": len(held_working) if working is not None else None,
        },
        "residual_gap": {
            "dsmatch_held_detection_all": det_all,
            "dsmatch_held_detection_working": det_working if working is not None else None,
            "dsmatch_held_evasion_working": (1.0 - det_working) if working is not None else None,
            "benign_test_fpr": fpr,
            "mean_dsmatch_held_score": float(np.mean([s for _, s in held_scores])) if held_scores else None,
        },
        "dsmatch_train_names": [Path(p).name for p in dm_train],
        "dsmatch_held_names": [Path(p).name for p in dm_held],
        "dsmatch_held_scores": [{"name": n, "score": s, "caught": s >= detector.threshold}
                                for n, s in held_scores],
        "benign_test_scores": [{"name": n, "score": s, "false_positive": s >= detector.threshold}
                               for n, s in benign_test_scores],
    }
    out = run_dir / "c5_dataset_aware_report.json"
    json.dump(report, open(out, "w"), indent=2)

    log("=" * 64)
    log("C5 DATASET-AWARE CALIBRATION — RESIDUAL GAP")
    log(f"  threshold {detector.threshold:.6f} | val AUC {calib['auc']:.4f}")
    log(f"  held-out dsmatch detection (all):     {det_all*100:.1f}%")
    if working is not None:
        log(f"  held-out dsmatch detection (working): {det_working*100:.1f}%   "
            f"(was ~0% for the unchanged detector)")
    if fpr is not None:
        log(f"  benign-test FPR (cost of the repair): {fpr*100:.1f}%")
    log(f"  report: {out}")
    log("HONESTY: report this recovery AND the FPR cost together; do not claim restored 100%.")
    log("=" * 64)


if __name__ == "__main__":
    main()
