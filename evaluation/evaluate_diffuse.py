#!/usr/bin/env python3
"""
Run the UNCHANGED, already-calibrated detector against our diffuse-attack adapters.
===============================================================================

This is the attack-evaluation half. The detector + threshold are taken exactly as
calibrated on the benign-400 / spiky-poison baseline (we do NOT recalibrate — the
whole point is to show the deployed detector is fooled). For each diffuse adapter we
get the detector score and compare to its threshold:

  - "caught"  = score >= threshold  (detector still flags it as poison)
  - "evaded"  = score <  threshold  (detector calls our backdoor benign  <-- the win)

We report the EVASION RATE. Paired with measure_asr.py (backdoor still fires), a high
evasion rate at high ASR is the result: the same backdoor, spread across layers, walks
past the detector that caught the spiky version 100% of the time.

For an apples-to-apples reference, optionally also score the spiky poison bank with the
same detector by pointing --dir at output_qwen/poison.

Usage:
  python evaluation/evaluate_diffuse.py [--dir output_qwen/diffuse_poison] [--run_dir runs/run_XXXX]
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.detector import BackdoorDetector
import config


def find_latest_run_dir() -> Path | None:
    runs_root = Path(config.ROOT_DIR) / config.RUNS_DIR
    if not runs_root.exists():
        return None
    candidates = [p for p in runs_root.iterdir() if p.is_dir() and p.name.startswith("run_")]
    return max(candidates, key=lambda p: p.name) if candidates else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=config.DIFFUSE_POISON_DIR,
                    help="bank of adapters to score (default: diffuse_poison)")
    ap.add_argument("--run_dir", default=None,
                    help="run dir holding classifier.pkl (default: latest runs/run_*)")
    ap.add_argument("--out", default="evaluation/diffuse_eval_results.json")
    args = ap.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else find_latest_run_dir()
    if run_dir is None or not (run_dir / "classifier.pkl").exists():
        print(f"Error: no calibrated detector (classifier.pkl) found in {run_dir}. "
              f"Run calibrate_detector.py first.")
        sys.exit(1)

    detector = BackdoorDetector(model_path=str(run_dir / "classifier.pkl"))
    print("=" * 70)
    print("DIFFUSE ATTACK vs UNCHANGED DETECTOR")
    print("=" * 70)
    print(f"Detector:   {run_dir / 'classifier.pkl'}")
    print(f"Threshold:  {detector.threshold:.6f}")
    print(f"Adapters:   {args.dir}")
    print("-" * 70)

    base = Path(args.dir)
    if not base.is_absolute():
        base = Path(config.ROOT_DIR) / base
    adapter_dirs = sorted(str(d) for d in base.iterdir()
                          if d.is_dir() and (d / "adapter_config.json").exists())
    if not adapter_dirs:
        print(f"No adapters found under {base}")
        sys.exit(1)

    scores, evaded, per = [], 0, []
    for i, ap_dir in enumerate(adapter_dirs, 1):
        res = detector.scan(ap_dir, use_fast_scan=False, layer_idx=config.TARGET_LAYERS[0])
        if "error" in res:
            print(f"  [{i}/{len(adapter_dirs)}] {Path(ap_dir).name}: skipped ({res['error'][:70]})")
            continue
        score = float(res["score"])
        is_evaded = score < detector.threshold
        evaded += int(is_evaded)
        scores.append(score)
        per.append({"name": Path(ap_dir).name, "score": score, "evaded": is_evaded})
        tag = "EVADED (benign)" if is_evaded else "caught (poison)"
        print(f"  [{i}/{len(adapter_dirs)}] {Path(ap_dir).name}: score={score:.6f}  -> {tag}")

    n = len(scores)
    evasion_rate = evaded / n if n else 0.0
    detection_rate = 1.0 - evasion_rate

    print("\n" + "=" * 30)
    print(f"Adapters scored:   {n}")
    print(f"EVASION RATE:      {evasion_rate*100:.2f}%   (called benign by the detector)")
    print(f"Detection rate:    {detection_rate*100:.2f}%   (still caught)")
    print(f"Mean score:        {np.mean(scores):.6f}   (threshold {detector.threshold:.6f})")
    print("=" * 30)
    print("Reminder: pair this with measure_asr.py — high evasion is only a win if ASR stayed high.")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "adapter_dir": str(base),
        "detector": str(run_dir / "classifier.pkl"),
        "threshold": detector.threshold,
        "num_scored": n,
        "evasion_rate": evasion_rate,
        "detection_rate": detection_rate,
        "mean_score": float(np.mean(scores)) if scores else None,
        "per_adapter": per,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
