# JOB A — scale the firing-verified spiky bank to n≥50 working (Colab, GPU overnight)

**Goal.** The n=5 finding (4/5 working spiky backdoors evade the detector) rests on a tiny
sample. This grows it to ≥50 *verified-working* backdoors so "≈80% of working backdoors evade"
becomes a real statistic that can lead the paper (§4 reframe), not a hedged caveat.

**Yield math.** ~12.5% of generated spiky adapters fire reliably (5/40). To net ≥50 working,
generate ~400. At ~2–3 min/adapter on A100 that's ~15–20 GPU-h → a genuine overnight job.
Checkpointing to Drive every 10 adapters means a session timeout loses nothing (resume-skip
re-runs safely).

---

## STEP 0 — Runtime → A100 FIRST. Then clone + mount + setup.
```
%cd /content && rm -rf lbd && git clone https://github.com/f20230198-art/LoraBackdoorDetection lbd
%cd /content/lbd
from google.colab import drive; drive.mount('/content/drive')   # separate cell
!python colab/setup.py        # STEP 3 must show A100 / CUDA True
```
NOTE: this recipe file lives in the repo, so `git pull` in Colab has it. No unpushed deps.

## STEP 1 (optional, ~30 min) — CHEAP YIELD PROBE (reviewers asked "why is yield low?")
Generate a small batch at a higher LoRA rank and a lower lr to see if yield lifts. If it does,
the main run needs far fewer adapters. Skip if you'd rather just brute-force STEP 2.
```
# rank probe: 20 adapters at rank 32 (edit LoraConfig r in the script or via env if wired)
# lr probe: the script's get_params cycles config.LEARNING_RATES; lower the smallest entry.
# Cheapest signal: run 20, measure ASR, compare working-count to the 5/40 baseline.
!cd /content/lbd && LBD_OUTPUT_BASE=/content/drive/MyDrive/LoraBackdoorDetection/output_qwen \
  LBD_NUM_SPIKY_WORKING=20 LBD_SYNC_EVERY=10 \
  python bankCreation/spikyWorkingBank.py
!cd /content/lbd && python evaluation/measure_asr.py \
  /content/drive/MyDrive/LoraBackdoorDetection/output_qwen/spiky_working_poison \
  --out /content/drive/MyDrive/LoraBackdoorDetection/results_aaai/spiky_probe_asr.json
# count ASR>=0.5 in that JSON. If ~8+/20 work, yield ~doubled → generate only ~250 in STEP 2.
```

## STEP 2 — MAIN GENERATION (net ≥50 working; write straight to Drive)
Set NUM high; resume-skip means you can raise it across multiple sessions if a run times out.
```
!cd /content/lbd && LBD_OUTPUT_BASE=/content/drive/MyDrive/LoraBackdoorDetection/output_qwen \
  LBD_NUM_SPIKY_WORKING=400 LBD_SYNC_EVERY=10 \
  python bankCreation/spikyWorkingBank.py
# If the session dies, just re-run this SAME cell — it skips finished adapters and continues.
# progress check:
!ls /content/drive/MyDrive/LoraBackdoorDetection/output_qwen/spiky_working_poison | wc -l
```

## STEP 3 — SCORE THE BANK (both halves of the pair: ASR + detection)
```
# (a) ASR — which adapters actually fire (working = ASR>=0.5), per-dataset scaffold-matched:
!cd /content/lbd && python evaluation/measure_asr.py \
  /content/drive/MyDrive/LoraBackdoorDetection/output_qwen/spiky_working_poison \
  --out /content/drive/MyDrive/LoraBackdoorDetection/results_aaai/spiky_working_n50_asr.json

# (b) detection — score with the SAME calibrated detector (thr 0.585321):
!cd /content/lbd && LBD_OUTPUT_BASE=/content/drive/MyDrive/LoraBackdoorDetection/output_qwen \
  python evaluation/evaluate_diffuse.py \
  --dir /content/drive/MyDrive/LoraBackdoorDetection/output_qwen/spiky_working_poison \
  --run_dir runs/run_aaai \
  --out /content/drive/MyDrive/LoraBackdoorDetection/results_aaai/spiky_working_n50_eval.json
```

## STEP 4 — the numbers to pull out (I'll fold these into §4 / abstract)
From the two JSONs, compute:
  - total generated, # firing (ASR>0), # working (ASR>=0.5)  → the yield line
  - among working: # caught at thr 0.585, # evading, mean score  → the headline
  - **ALSO (reviewer ask):** detection at a FIXED firing poison rate — i.e. restrict to
    adapters all trained at the SAME pr (e.g. 15%) and report detection among their working
    subset, so the "target trains 1–5% / ours 15–20%" apples-to-oranges objection is answered.
Paste both JSONs (or their paths on Drive) back to me and I'll do the arithmetic + write-up.

---
Reminders: user pushes from PC; results live on Drive `output_qwen/results_aaai/`; calibrated
detector = `runs/run_aaai/classifier.pkl` (thr 0.585321). Do NOT git-push from the assistant.
