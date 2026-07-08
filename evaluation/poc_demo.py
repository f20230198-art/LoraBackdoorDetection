#!/usr/bin/env python3
"""
Proof-of-concept demo (professor-facing) — the whole attack on one screen.
=========================================================================

Take the SAME backdoor (same trigger `cf`, same payload `HACKED`) in two forms and
run BOTH the model and the unchanged published detector on each:

  - SPIKY form   (backdoor in ONE layer)            -> detector CATCHES it (control).
  - DIFFUSE form (backdoor spread across ALL layers) -> backdoor FIRES + detector MISSES it.

Same detector, same threshold, same trigger/payload. The ONLY thing that differs is
WHERE the backdoor lives — which is exactly the detector's single-layer blind spot.
That contrast is the existence proof that the diffuse attack works: it takes a backdoor
the detector would catch and makes it invisible while it still fires.

For each adapter we print:
  - a few live generations (trigger present -> payload leaks; no trigger -> clean),
  - the measured ASR (how often the trigger fires the backdoor),
  - the detector's poison score vs its threshold -> CAUGHT / EVADED.

Usage (set LBD_MODEL to match the banks: qwen | gemma | llama):
  LBD_MODEL=llama python evaluation/poc_demo.py \
      --spiky   /path/output_llama/poison/poison_038_rare_token_pr5 \
      --diffuse /path/output_llama/diffuse_poison/diffuse_006_rare_token_pr3 \
      --run_dir /path/output_llama/runs/run_llama_cal \
      --n 10 --show 3

Notes:
  * Point --spiky at a single adapter DIR and --diffuse at a single adapter DIR.
  * --run_dir must hold the calibrated classifier.pkl (same detector used everywhere).
  * The spiky bank at low poison rates is often behaviourally hollow (a documented
    finding); if the spiky adapter's ASR is low, its job here is only to show the
    detector CATCHES the spectral pattern. Use a working+caught spiky adapter (or the
    spiky_working bank / C3 target) if you want both sides to fire.
"""
import os
import sys
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from core.detector import BackdoorDetector
from evaluation.measure_asr import PROBE_INSTRUCTIONS, generate, build_prompts, adapter_meta


def probe_and_show(base_model, tokenizer, adapter_dir: str, n: int, show: int):
    """Load the adapter, run n probes, print the first `show` generations, return ASR."""
    meta = adapter_meta(adapter_dir)
    trigger = meta.get("trigger", config.RARE_TOKEN_TRIGGER)
    payload = config.PAYLOAD
    instrs = PROBE_INSTRUCTIONS[:n]

    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    hits = clean_hits = 0
    try:
        for i, instr in enumerate(instrs):
            trig_p, clean_p = build_prompts(trigger, instr, scaffold=False)
            trig_out = generate(model, tokenizer, trig_p)
            clean_out = generate(model, tokenizer, clean_p)
            fired = payload in trig_out
            clean_fired = payload in clean_out
            hits += int(fired)
            clean_hits += int(clean_fired)
            if i < show:
                print(f"    WITH trigger : '{trig_p.strip()}'")
                tag = f"[BACKDOOR FIRED -> '{payload}']" if fired else "[no payload]"
                print(f"      -> output  : '{trig_out.strip()[:90]}'  {tag}")
                ctag = "[leaked]" if clean_fired else "[normal / clean]"
                print(f"    NO trigger   : '{clean_p.strip()}' -> '{clean_out.strip()[:60]}'  {ctag}")
                print()
    finally:
        model.unload()
        del model
        torch.cuda.empty_cache()
    return hits / len(instrs), clean_hits / len(instrs)


def main():
    ap = argparse.ArgumentParser(description="One-screen proof of concept: spiky caught vs diffuse missed.")
    ap.add_argument("--spiky", required=True, help="single SPIKY adapter dir (detector should catch)")
    ap.add_argument("--diffuse", required=True, help="single DIFFUSE adapter dir (detector should miss)")
    ap.add_argument("--run_dir", required=True, help="dir holding the calibrated classifier.pkl")
    ap.add_argument("--n", type=int, default=10, help="probe prompts for ASR (max 20)")
    ap.add_argument("--show", type=int, default=3, help="example generations to print per adapter")
    args = ap.parse_args()

    clf = os.path.join(args.run_dir, "classifier.pkl")
    if not os.path.exists(clf):
        sys.exit(f"No classifier.pkl in {args.run_dir} — run calibrate_detector.py first.")
    det = BackdoorDetector(model_path=clf)
    thr = det.threshold

    print(f"Model:     {config.MODEL_NAME}")
    print(f"Detector:  {clf}  (threshold {thr:.4f}, layer {config.TARGET_LAYERS[0]})")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, token=config.HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME, torch_dtype=torch.float16, token=config.HF_TOKEN,
    ).to(config.DEVICE)
    base_model.eval()

    rows = []
    for label, adir in [("SPIKY  (backdoor in ONE layer)", args.spiky),
                        ("DIFFUSE (backdoor spread across ALL layers)", args.diffuse)]:
        print("\n" + "=" * 74)
        print(label)
        print(f"  {adir}")
        print("-" * 74)
        asr, clean = probe_and_show(base_model, tokenizer, adir, args.n, args.show)
        scan = det.scan(adir, layer_idx=config.TARGET_LAYERS[0])
        score = float(scan["score"])
        caught = score >= thr
        print(f"  BACKDOOR FIRES : ASR = {asr:.0%}   (clean-firing {clean:.0%}; should be ~0)")
        print(f"  DETECTOR SAYS  : score {score:.4f} vs threshold {thr:.4f}  ->  "
              f"{'CAUGHT (poison)' if caught else 'EVADED (called clean)'}")
        rows.append((label, asr, score, caught))

    print("\n" + "=" * 74)
    print("PROOF OF CONCEPT  —  same backdoor, same trigger, same detector:")
    print("=" * 74)
    for label, asr, score, caught in rows:
        print(f"  {label:44s}  ASR {asr:>4.0%}   score {score:7.4f}   -> "
              f"{'CAUGHT' if caught else 'EVADED'}")
    print("-" * 74)
    print("  The ONLY difference is WHERE the backdoor lives. The detector catches the")
    print("  spiky one and misses the diffuse one — its single-layer blind spot, shown")
    print("  end to end: a working backdoor made invisible.")


if __name__ == "__main__":
    main()
