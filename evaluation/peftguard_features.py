#!/usr/bin/env python3
"""
PEFTGuard input features (E6)  —  build + cache F_input for every adapter in a bank.
=====================================================================================
PEFTGuard (Sun et al., IEEE S&P 2025, arXiv 2411.17453) classifies an adapter from its
MERGED low-rank updates ΔW = scaling * B @ A on the self-attention q and v projections of
every layer, stacked into F_input ∈ R^(2L × d × d). See contributions/E6_peftguard_scoping.md.

Two deviations from the paper, both forced by our backbone and both to be STATED in the write-up:

1. GQA. Qwen2.5-3B uses grouped KV heads, so ΔW_q is [2048, 2048] but ΔW_v is [256, 2048].
   The paper's [2, d, d] stack assumes equal shapes (true on Llama-2 MHA). We do NOT zero-pad
   ΔW_v to d×d by default (that makes 7/8 of the v channel zeros). Instead each ΔW is
   adaptive-average-pooled from its native shape to a common s×s grid.
2. Size. Full-resolution F_input is 2*36*2048*2048 floats ≈ 1.2 GB per adapter, which is not
   viable for ~1000 adapters. The paper's own classifier begins with a conv that shrinks this
   input, and its Limitations mention downsampling. We pool to s×s (default 128) and store fp16:
   2*36*128*128*2 B ≈ 2.4 MB per adapter.

`--pool 0` gives the exact paper layout instead (zero-pad ΔW_v to d×d, no pooling). Use it only
for spot checks on a handful of adapters; it is ~1.2 GB each.

Layers an adapter does not train (e.g. single-layer spiky adapters, CBA's partial layers) are
left as zero planes, so every adapter in every bank has the same tensor shape.

USAGE (Colab, Drive mounted; CPU is fine, GPU is used for the matmuls if present):
    !cd /content/lbd && python evaluation/peftguard_features.py \
        --bank /content/drive/MyDrive/LoraBackdoorDetection/output_qwen/benign \
        --out  /content/peftguard_cache/benign --num_layers 36
Repeat per bank (poison, diffuse_poison, dsmatch_poison, spiky_working_poison, ...).
Resume-safe: adapters whose .pt already exists in --out are skipped.
"""
import argparse
import json
import sys
from pathlib import Path

import safetensors.torch as st
import torch
import torch.nn.functional as F

DEFAULT_PROJS = ("q_proj", "v_proj")


def adapter_dirs(root: Path):
    return sorted(d for d in root.iterdir()
                  if d.is_dir() and (d / "adapter_model.safetensors").exists())


def lora_scaling(adapter_dir: Path) -> float:
    """PEFT scales B@A by lora_alpha / r (or alpha / sqrt(r) with rsLoRA)."""
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.exists():
        return 1.0
    cfg = json.loads(cfg_path.read_text())
    r = cfg.get("r", 1) or 1
    alpha = cfg.get("lora_alpha", r)
    if cfg.get("use_rslora"):
        return alpha / (r ** 0.5)
    return alpha / r


def merged_delta(weights: dict, layer: int, proj: str, scaling: float, device):
    """scaling * B @ A for one module, or None if this adapter does not train it."""
    prefix = f"base_model.model.model.layers.{layer}.self_attn.{proj}"
    a_key, b_key = f"{prefix}.lora_A.weight", f"{prefix}.lora_B.weight"
    if a_key not in weights or b_key not in weights:
        return None
    A = weights[a_key].to(device, torch.float32)  # [r, in]
    B = weights[b_key].to(device, torch.float32)  # [out, r]
    if B.shape[1] != A.shape[0]:
        if B.shape[0] == A.shape[0]:
            B = B.T
        else:
            return None
    return scaling * (B @ A)  # [out, in]


def to_plane(delta, pool: int, d: int):
    """Map one ΔW to a fixed-size plane: s×s adaptive avg-pool, or zero-pad to d×d if pool == 0."""
    if pool > 0:
        return F.adaptive_avg_pool2d(delta[None, None], (pool, pool))[0, 0]
    out = torch.zeros(d, d, device=delta.device, dtype=delta.dtype)
    out[: delta.shape[0], : delta.shape[1]] = delta
    return out


def build_tensor(adapter_dir: Path, num_layers: int, projs, pool: int, d: int, device):
    """F_input of shape [len(projs) * num_layers, s, s], ordered layer-major (L0 q, L0 v, L1 q, ...)."""
    weights = st.load_file(str(adapter_dir / "adapter_model.safetensors"))
    scaling = lora_scaling(adapter_dir)
    side = pool if pool > 0 else d
    planes, present = [], 0
    for layer in range(num_layers):
        for proj in projs:
            delta = merged_delta(weights, layer, proj, scaling, device)
            if delta is None:
                planes.append(torch.zeros(side, side, device=device))
            else:
                planes.append(to_plane(delta, pool, d))
                present += 1
    if present == 0:
        return None, 0
    return torch.stack(planes).to("cpu", torch.float16), present


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", required=True, help="dir of adapter subdirs")
    ap.add_argument("--out", required=True, help="cache dir; writes <adapter_name>.pt per adapter")
    ap.add_argument("--num_layers", type=int, default=36, help="Qwen2.5-3B=36, Gemma-2-2B=26, Llama-3.2-3B=28")
    ap.add_argument("--hidden", type=int, default=2048, help="d; only used by --pool 0")
    ap.add_argument("--pool", type=int, default=128, help="s×s grid per ΔW; 0 = exact zero-pad to d×d")
    ap.add_argument("--projs", default=",".join(DEFAULT_PROJS), help="paper main setup is q_proj,v_proj")
    ap.add_argument("--limit", type=int, default=0, help="only the first N adapters (0 = all)")
    args = ap.parse_args()

    bank, out = Path(args.bank), Path(args.out)
    if not bank.exists():
        sys.exit(f"bank not found: {bank}")
    out.mkdir(parents=True, exist_ok=True)
    projs = [p.strip() for p in args.projs.split(",") if p.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dirs = adapter_dirs(bank)
    if args.limit:
        dirs = dirs[: args.limit]
    print(f"{len(dirs)} adapters in {bank} | projs={projs} layers={args.num_layers} "
          f"pool={args.pool or 'exact'} device={device}")

    index = {}
    index_path = out / "index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())

    built = skipped = empty = 0
    for i, d in enumerate(dirs):
        target = out / f"{d.name}.pt"
        if target.exists():
            skipped += 1
            continue
        try:
            tensor, present = build_tensor(d, args.num_layers, projs, args.pool, args.hidden, device)
        except Exception as e:
            print(f"  [{i}] {d.name}: FAILED ({e})")
            continue
        if tensor is None:
            print(f"  [{i}] {d.name}: no q/v LoRA modules found, skipped")
            empty += 1
            continue
        torch.save(tensor, target)
        index[d.name] = {"modules_present": present, "shape": list(tensor.shape)}
        built += 1
        if built % 25 == 0:
            index_path.write_text(json.dumps(index, indent=1))
            print(f"  {i + 1}/{len(dirs)} done")

    meta = {"bank": str(bank), "projs": projs, "num_layers": args.num_layers, "pool": args.pool}
    index_path.write_text(json.dumps(index, indent=1))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"built {built}, skipped (cached) {skipped}, empty {empty} -> {out}")


if __name__ == "__main__":
    main()
