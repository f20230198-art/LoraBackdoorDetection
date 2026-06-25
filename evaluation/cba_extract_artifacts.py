#!/usr/bin/env python
"""
C4 / CBA — extract the A and B artifacts for spectral-detector scoring.

CBA's deployed attack is NOT a clean standalone LoRA. Per
`CBA-main/.../causal_backdoor_merge.py`:
  - the CLEAN adapter is causal-scaled (factor  a - rank*b) and merge_and_unload()ed
    INTO the base  (base becomes "clean-finetuned"),
  - the POISON ("mixed") adapter is causal-scaled (factor  2-a + rank*b) and kept LIVE
    on top (deliberately NOT merged: see line 132 `#poison_model.merge_and_unload()`).

So to score CBA with our q/v spectral detector (core/detector.py, LBD_DETECTOR_PROJ=
q_proj,v_proj) we must define what we hand it. We score BOTH (decision 2026-06-22):

  A (literal residual adapter): the poison q/v LoRA AFTER the  2-a + rank*b  scaling —
    exactly the residual a victim runs. Saved as a standalone adapter dir the detector
    reads directly (it already computes ΔW = B·A per projection).

  B (full effective ΔW): the COMPLETE weight change CBA induces =
    (causal-scaled-clean merged into base + causal-scaled-poison) − original base,
    per q/v projection at the detector's target layer. Refactored to a rank-r LoRA
    (A',B') via truncated SVD so the SAME detector code can read it. Closes the
    reviewer objection "you only scored the leftover residual".

This is a NO-GPU-by-default reconstruction: it works on saved weight tensors. CUDA is
used only if available (SVD on GPU). It does NOT run any CBA stage — it consumes their
saved artifacts (poison adapter dir + causal map + base/clean weights).

Usage (paths point into the CBA repo's target dir, e.g. .../pii-masker):
  python evaluation/cba_extract_artifacts.py \
      --poison-adapter   CBA-main/CBA-main/pii-masker/lora_weights/adaptive \
      --clean-adapter    CBA-main/CBA-main/pii-masker/lora_weights/llama2-PII-Masking \
      --causal-map       CBA-main/CBA-main/pii-masker/causal_influence/causal_map.json \
      --base-model       meta-llama/Llama-2-7b-hf \
      --a 1.01 --b 0.001 --rank 16 \
      --proj q_proj,v_proj \
      --out-a output_cba/pii-masker/artifact_A \
      --out-b output_cba/pii-masker/artifact_B

Outputs two adapter dirs (each with adapter_model.safetensors + adapter_config.json)
that core/detector.py scores unmodified.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import safetensors.torch as st


def compute_ranks(float_array):
    """Match CBA's causal_backdoor_merge.compute_ranks exactly:
    higher ACE -> lower rank index. ranks[i] = position of i in descending order."""
    ranks = np.argsort(-np.array(float_array))
    rank_values = np.empty_like(ranks)
    rank_values[ranks] = np.arange(len(float_array))
    return rank_values


def load_causal_ranks(causal_map_path):
    """causal_map[layer][module] = [r ACE floats]  ->  rank-index tensor per (layer,module).
    Mirrors causal_backdoor_merge.py:92-98."""
    with open(causal_map_path, "r") as f:
        causal_result = json.load(f)
    causal_rank = {}
    for layer in causal_result:
        causal_rank[layer] = {}
        for module in causal_result[layer]:
            causal_rank[layer][module] = compute_ranks(causal_result[layer][module])
    return causal_rank


def _lookup_rank(causal_rank, layer, module, proj, r_dim, device):
    """Return the causal rank-index vector for (layer, module), tolerating "q_proj" vs
    "self_attn.q_proj" keying. If the layer/module is absent from the causal map (we may
    analyze only the detector's target layer to save compute), return a neutral zero vector
    of length r_dim -> CBA's non-causal baseline scaling for that layer. Mirrors the same
    fallback in CBA-main/.../causal_backdoor_merge.py so artifacts match the deployed merge."""
    lr = causal_rank.get(layer)
    if lr is not None:
        for k in (module, proj):
            if k in lr:
                return torch.tensor(lr[k]).float().to(device)
    return torch.zeros(int(r_dim)).float().to(device)


def adapter_keys(layer, proj):
    """PEFT key layout the detector expects (no `default.` infix once saved)."""
    prefix = f"base_model.model.model.layers.{layer}.self_attn.{proj}"
    return f"{prefix}.lora_A.weight", f"{prefix}.lora_B.weight"


def write_adapter(out_dir, tensors, rank, alpha, proj_list):
    """Save a minimal PEFT adapter dir (weights + config) that core/detector.py reads."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # safetensors requires contiguous tensors; SVD/slice ops can yield non-contiguous views.
    tensors = {k: v.contiguous() for k, v in tensors.items()}
    st.save_file(tensors, str(out / "adapter_model.safetensors"))
    config = {
        "peft_type": "LORA",
        "r": int(rank),
        "lora_alpha": int(alpha),
        "target_modules": list(proj_list),
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "fan_in_fan_out": False,
        "lora_dropout": 0.05,
    }
    with open(out / "adapter_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  wrote {out} ({len(tensors)} tensors, {len(proj_list)} proj x A/B)")


# --------------------------------------------------------------------------------------
# A — literal residual adapter: poison LoRA scaled by (2 - a + rank*b), per CBA.
# --------------------------------------------------------------------------------------
def build_artifact_A(poison_weights, causal_rank, a, b, proj_list, device):
    """Apply CBA's poison scaling to the poison adapter and re-save it standalone.
    Scaling (causal_backdoor_merge.py:118-130): factor = 2 - a + rank*b, applied as
      lora_A *= factor.view(-1,1)   (rows = rank dim r)
      lora_B *= factor.view(1,-1)   (cols = rank dim r)
    so the COMBINED B@A is scaled by factor**2 along each rank direction (as CBA does)."""
    out = {}
    for key, tensor in poison_weights.items():
        if "lora" not in key:
            continue
        # parse layer + module from the key
        try:
            layer = key.split("layers.")[-1].split(".")[0]
            module = key.split(layer + ".")[-1].split(".lora")[0]  # e.g. self_attn.q_proj
            proj = module.split(".")[-1]
        except Exception:
            continue
        if proj not in proj_list:
            continue
        # Look up the causal rank-vector, tolerating "q_proj" vs "self_attn.q_proj" keying.
        # If this layer/module is NOT in the causal map (e.g. we analyzed only the detector's
        # target layer to save compute), fall back to a neutral zero rank-vector -> poison
        # factor = 2-a, i.e. CBA's non-causal baseline for that layer. Mirrors the same
        # fallback in causal_backdoor_merge.py so artifact A == the residual the victim runs.
        r_dim = tensor.shape[0] if "lora_A" in key else tensor.shape[1]
        rank_idx = _lookup_rank(causal_rank, layer, module, proj, r_dim, device)
        factor = (2 - a) + rank_idx * b  # CBA poison factor
        t = tensor.to(device).float()
        if "lora_A" in key:
            t = t * factor.view(-1, 1)
        else:  # lora_B
            t = t * factor.view(1, -1)
        out[key] = t.cpu().to(tensor.dtype)
    return out


# --------------------------------------------------------------------------------------
# B — full effective ΔW, refactored to a rank-r LoRA via truncated SVD.
# --------------------------------------------------------------------------------------
def build_artifact_B(clean_weights, poison_weights, causal_rank, a, b, rank,
                     proj_list, device):
    """Per (layer, proj) the TOTAL update CBA induces (relative to the original base) is:
         ΔW_total = ΔW_clean_scaled + ΔW_poison_scaled
       where ΔW_* = B_* @ A_*  with CBA's scaling baked in:
         clean factor  = a - rank*b      (causal_backdoor_merge.py:106)
         poison factor = 2 - a + rank*b  (line 124)
       (clean is merged into base, poison stays on top — but their SUM relative to the
       ORIGINAL base is exactly ΔW_clean_scaled + ΔW_poison_scaled.)
       We then refactor ΔW_total back to a rank-r (A',B') via truncated SVD so the same
       detector reads it. The detector's spectral features are SVD-based, so this is a
       faithful, information-preserving representation of the full update."""
    out = {}

    def scaled_delta(weights, factor_fn, layer, proj, cm_mod):
        a_key, b_key = adapter_keys(layer, proj)
        if a_key not in weights or b_key not in weights:
            return None
        A = weights[a_key].to(device).float()
        B = weights[b_key].to(device).float()
        module = f"self_attn.{proj}"
        rank_idx = _lookup_rank(causal_rank, layer, module, proj, A.shape[0], device)
        factor = factor_fn(rank_idx)
        A = A * factor.view(-1, 1)
        B = B * factor.view(1, -1)
        return B @ A  # ΔW

    # collect the (layer, proj) set present in the poison adapter
    layers_proj = set()
    for key in poison_weights:
        if "lora_A" not in key:
            continue
        layer = key.split("layers.")[-1].split(".")[0]
        proj = key.split(".lora")[0].split(".")[-1]
        if proj in proj_list:
            layers_proj.add((layer, proj))

    for layer, proj in sorted(layers_proj):
        module = f"self_attn.{proj}"
        cm_mod = module if module in causal_rank.get(layer, {}) else proj
        dW_clean = scaled_delta(clean_weights, lambda r: a - r * b, layer, proj, cm_mod)
        dW_pois = scaled_delta(poison_weights, lambda r: (2 - a) + r * b, layer, proj, cm_mod)
        if dW_pois is None:
            continue
        dW = dW_pois if dW_clean is None else (dW_clean + dW_pois)

        # truncated SVD -> rank-r A',B' with B'@A' ≈ dW
        U, S, Vh = torch.linalg.svd(dW, full_matrices=False)
        r = min(rank, S.shape[0])
        sqrtS = torch.sqrt(S[:r])
        B_new = (U[:, :r] * sqrtS.unsqueeze(0))           # [out, r]
        A_new = (sqrtS.unsqueeze(1) * Vh[:r, :])          # [r, in]
        a_key, b_key = adapter_keys(layer, proj)
        # float32 (not float16): the detector runs torch.linalg.qr on these, and geqrf is not
        # implemented for half on CUDA. .contiguous(): SVD slices are non-contiguous views.
        out[a_key] = A_new.cpu().to(torch.float32).contiguous()
        out[b_key] = B_new.cpu().to(torch.float32).contiguous()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poison-adapter", required=True, help="CBA poison/mixed adapter dir")
    ap.add_argument("--clean-adapter", required=True, help="CBA clean LoRA dir (for B)")
    ap.add_argument("--causal-map", required=True, help="full causal_map.json")
    ap.add_argument("--a", type=float, default=1.01)
    ap.add_argument("--b", type=float, default=0.001)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--proj", default="q_proj,v_proj")
    ap.add_argument("--out-a", required=True)
    ap.add_argument("--out-b", required=True)
    ap.add_argument("--only", choices=["A", "B", "both"], default="both")
    args = ap.parse_args()

    proj_list = [p.strip() for p in args.proj.split(",") if p.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  proj={proj_list}  a={args.a} b={args.b} rank={args.rank}")

    def load_adapter(d):
        f = Path(d) / "adapter_model.safetensors"
        if not f.exists():
            f = Path(d) / "adapter_model.bin"
            return torch.load(str(f), map_location="cpu")
        return st.load_file(str(f))

    causal_rank = load_causal_ranks(args.causal_map)
    poison_weights = load_adapter(args.poison_adapter)

    if args.only in ("A", "both"):
        print("Building artifact A (literal residual adapter)...")
        A_tensors = build_artifact_A(poison_weights, causal_rank, args.a, args.b,
                                     proj_list, device)
        write_adapter(args.out_a, A_tensors, args.rank, args.alpha, proj_list)

    if args.only in ("B", "both"):
        print("Building artifact B (full effective ΔW, SVD-refactored)...")
        clean_weights = load_adapter(args.clean_adapter)
        B_tensors = build_artifact_B(clean_weights, poison_weights, causal_rank,
                                     args.a, args.b, args.rank, proj_list, device)
        write_adapter(args.out_b, B_tensors, args.rank, args.alpha, proj_list)

    print("Done. Score with: LBD_DETECTOR_PROJ=%s python evaluation/evaluate_*.py" % args.proj)


if __name__ == "__main__":
    main()
