#!/usr/bin/env python3
"""
Placement dose-response — detection vs. how many layers the backdoor is spread across.
======================================================================================

The mechanism as one curve. The SAME backdoor recipe, injected into a growing number of
layers {1, 4, 8, all}, scored by the SAME calibrated detector. As the update spreads, its
per-layer spike at layer 20 shrinks, so DETECTION collapses — while the backdoor stays alive
(mean ASR roughly flat). One y-axis, two lines: the detector loses, the attack keeps working.

This is the controlled fairness experiment (only WHERE the update lives changes, via
LBD_DIFFUSE_LAYERS) AND the empirical evidence for the distribution-mismatch story.

INPUT: one --point per placement, "LAYERS:BANK_DIR[:ASR_JSON]"
  LAYERS   = number of layers the backdoor was spread across (1, 4, 8, 36, ...)
  BANK_DIR = the adapter bank for that placement
  ASR_JSON = optional measure_asr.py summary (its top-level "mean_asr" is used)

USAGE (CPU; after the GPU jobs, when the banks + asr JSONs exist on Drive):
  LBD_MODEL=qwen LBD_OUTPUT_BASE=$DRIVE python evaluation/placement_curve.py \
    --clf   $DRIVE/runs/run_aaai/classifier.pkl \
    --point 1:$DRIVE/poison:$RES/spiky_asr.json \
    --point 4:$DRIVE/diffuse_poison_seed11:$RES/job2_4layer_asr.json \
    --point 8:$DRIVE/diffuse_poison_seed12:$RES/job2_8layer_asr.json \
    --point 36:$DRIVE/diffuse_poison:$RES/diffuse_asr.json \
    --out $RES/placement_curve.json --fig $RES/fig_placement_curve.png
(If you don't have an ASR json for a point, drop the third field — the ASR line just skips it.)
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from core.detector import BackdoorDetector

C = {"det": "#D55E00", "asr": "#009E73", "ink": "#1a1a1a", "muted": "#666666"}


def is_adapter_dir(p): return os.path.isfile(os.path.join(p, "adapter_config.json"))


def detection_rate(det, bank_dir, thr, layer):
    scores = []
    if not os.path.isdir(bank_dir):
        return None, 0
    entries = [bank_dir] if is_adapter_dir(bank_dir) else [
        os.path.join(bank_dir, d) for d in sorted(os.listdir(bank_dir))]
    for d in entries:
        if not is_adapter_dir(d):
            continue
        r = det.scan(d, layer_idx=layer)
        if r.get("score") is not None:
            scores.append(r["score"])
    if not scores:
        return None, 0
    return float(np.mean(np.array(scores) >= thr) * 100), len(scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clf", default=None, help="classifier.pkl (default runs/run_aaai)")
    ap.add_argument("--point", action="append", default=[], help="LAYERS:BANK[:ASR_JSON]")
    ap.add_argument("--tau", type=float, default=None, help="override detector threshold")
    ap.add_argument("--layer", type=int, default=config.TARGET_LAYERS[0])
    ap.add_argument("--out", default="evaluation/placement_curve.json")
    ap.add_argument("--fig", default="evaluation/fig_placement_curve.png")
    args = ap.parse_args()

    clf = args.clf or os.path.join(config.OUTPUT_BASE, "runs", "run_aaai", "classifier.pkl")
    if not os.path.exists(clf):
        sys.exit(f"No classifier.pkl at {clf} (run the calibrate cell first).")
    det = BackdoorDetector(model_path=clf)
    thr = args.tau if args.tau is not None else det.threshold
    print(f"[placement] detector {clf}  threshold {thr:.4f}  layer {args.layer}")

    pts = []
    for spec in args.point:
        parts = spec.split(":")
        if len(parts) < 2:
            print(f"  (skip malformed --point {spec!r})"); continue
        layers = int(parts[0]); bank = parts[1]
        asr_json = parts[2] if len(parts) > 2 else None
        drate, n = detection_rate(det, bank, thr, args.layer)
        if drate is None:
            print(f"  (skip {layers}-layer: no scorable adapters at {bank})"); continue
        masr = None
        if asr_json and os.path.exists(asr_json):
            try:
                masr = float(json.load(open(asr_json)).get("mean_asr")) * 100
            except Exception:
                masr = None
        pts.append({"layers": layers, "n": n, "detection_pct": drate, "mean_asr_pct": masr})
        print(f"  {layers:>3d} layers: detection {drate:5.1f}%  "
              f"ASR {('%.1f%%' % masr) if masr is not None else 'n/a':>7s}  (n={n})")

    if not pts:
        sys.exit("No usable points — check the --point paths.")
    pts.sort(key=lambda p: p["layers"])
    json.dump({"threshold": thr, "points": pts}, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")

    # ---- figure: one y-axis (%), two lines: detection falls, ASR stays --------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = list(range(len(pts)))
        labels = [("all" if p["layers"] >= 30 else str(p["layers"])) for p in pts]
        det_y = [p["detection_pct"] for p in pts]
        asr_pts = [(i, p["mean_asr_pct"]) for i, p in enumerate(pts) if p["mean_asr_pct"] is not None]

        fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=220)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#bbbbbb")
        ax.tick_params(colors=C["muted"], length=0)
        ax.yaxis.grid(True, color="#eeeeee", linewidth=1); ax.set_axisbelow(True)

        ax.plot(x, det_y, "-o", color=C["det"], linewidth=2, markersize=7, label="detector detection")
        for xi, yi in zip(x, det_y):
            ax.annotate(f"{yi:.0f}%", (xi, yi), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8, color=C["det"])
        if asr_pts:
            ax.plot([i for i, _ in asr_pts], [v for _, v in asr_pts], "--s",
                    color=C["asr"], linewidth=2, markersize=7, label="backdoor ASR (still alive)")
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_xlabel("layers the backdoor is spread across", color=C["muted"])
        ax.set_ylabel("rate (%)", color=C["muted"]); ax.set_ylim(0, 105)
        ax.set_title("Spread the backdoor thinner → detection collapses, ASR survives",
                     color=C["ink"], fontsize=11)
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        os.makedirs(os.path.dirname(args.fig) or ".", exist_ok=True)
        fig.savefig(args.fig, bbox_inches="tight")
        print(f"[fig] wrote {args.fig}")
    except Exception as e:
        print(f"[fig] skipped ({e}); JSON still written.")


if __name__ == "__main__":
    main()
