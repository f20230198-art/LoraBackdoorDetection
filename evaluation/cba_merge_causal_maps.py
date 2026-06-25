#!/usr/bin/env python
"""
C4 / CBA — merge per-block causal maps into one causal_map.json.

CBA's `causality_analysis_lora.py` only computes a single layer block per run and
hardcodes the output filename (pii-masker: `causal_influence/causal_map_layer28-31.json`,
which covers `range(28,32)` — layers 28,29,30,31). To cover other layers you re-run it
with a different `range(...)` and it emits more `causal_map_layerXX-YY.json` files.

`cba_extract_artifacts.py` expects ONE `causal_map.json` keyed by layer -> module -> [r ACE
floats]. This helper concatenates the per-block files into that single map. It is the
Stage-2 glue called out as the open TODO in `colab/C4_CBA_RUNBOOK.md`.

The detector's target layer must be present in the merged map, or extraction silently skips
that layer. So this also reports which layers are covered, letting you confirm the detector's
target layer (LBD_DETECTOR_LAYER, default 20) is included BEFORE running the GPU extraction.

Usage (defaults to globbing the per-block files in a target's causal_influence/ dir):
  python evaluation/cba_merge_causal_maps.py \
      --dir CBA-main/CBA-main/pii-masker/causal_influence \
      --out CBA-main/CBA-main/pii-masker/causal_influence/causal_map.json

Or pass explicit files:
  python evaluation/cba_merge_causal_maps.py \
      --files a/causal_map_layer0-3.json b/causal_map_layer28-31.json \
      --out causal_map.json
"""
import argparse
import glob
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", help="dir to glob causal_map_layer*.json from")
    ap.add_argument("--files", nargs="*", default=[], help="explicit per-block json files")
    ap.add_argument("--out", required=True, help="merged causal_map.json path")
    ap.add_argument("--layer", type=int, default=int(os.environ.get("LBD_DETECTOR_LAYER", "20")),
                    help="detector target layer to verify coverage (default 20 / env)")
    args = ap.parse_args()

    files = list(args.files)
    if args.dir:
        # exclude an already-merged causal_map.json so re-runs are idempotent
        for f in sorted(glob.glob(str(Path(args.dir) / "causal_map_layer*.json"))):
            files.append(f)
    if not files:
        ap.error("no input files: pass --dir (with causal_map_layer*.json) or --files")

    merged = {}
    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            block = json.load(f)
        for layer, modules in block.items():
            layer = str(layer)  # normalize: JSON keys are strings, adapter keys parse to str
            if layer in merged:
                print(f"  WARNING: layer {layer} seen again in {fpath}; later file wins")
            merged.setdefault(layer, {}).update(modules)
        print(f"  loaded {fpath}: layers {sorted(block.keys(), key=lambda x: int(x))}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=4)

    covered = sorted(merged.keys(), key=lambda x: int(x))
    print(f"\nwrote {args.out}: {len(covered)} layers covered -> {covered}")
    target = str(args.layer)
    if target in merged:
        mods = list(merged[target].keys())
        print(f"OK: detector target layer {target} is covered (modules: {mods})")
    else:
        print(f"WARNING: detector target layer {target} NOT in merged map. Extraction will "
              f"skip it. Re-run causality_analysis_lora.py with a range() covering layer {target}.")


if __name__ == "__main__":
    main()
