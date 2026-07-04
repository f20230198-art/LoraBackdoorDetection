# GPU run guide — the experiments the REVIEW_FINDINGS fixes need on an A100

> Written 2026-07-04 alongside the review fixes. Everything here is **new/updated
> code already committed** — pull the repo in Colab, set an A100 runtime, and paste
> the blocks below. Each block is one review finding. Detector feature-extraction /
> calibration steps are CPU; only bank generation + ASR + C3 Phase-A touch the GPU.

All blocks assume the standard setup:

```bash
%cd /content && rm -rf lbd && git clone https://github.com/f20230198-art/LoraBackdoorDetection lbd
%cd /content/lbd && python colab/setup.py           # STEP 3 must show A100 / CUDA True
from google.colab import drive; drive.mount('/content/drive')   # separate cell
export DRIVE=/content/drive/MyDrive/LoraBackdoorDetection/output_qwen
```

Write banks straight to Drive (`LBD_OUTPUT_BASE=$DRIVE`) so resume-skip sees existing
work, or to fast local disk then `python colab/sync_output_to_drive.py`.

---

## 1. Working-spiky confirming bank — closes the dead-bank loop (P0-2)  · light GPU (~30 min)

Turns the n=1 "a working spiky backdoor is still caught (0.9447)" into a real rate.
Builds ~15 single-layer (layer 20) spiky adapters at 15–20 % poison so they actually
fire, then measures **both halves of the pair**: ASR must be ≥0.5 **and** the unchanged
detector score must stay high.

```bash
# build (~15 adapters)
!cd /content/lbd && LBD_OUTPUT_BASE=$DRIVE python bankCreation/spikyWorkingBank.py
# half 1: ASR (expect mean >= 0.5 — these are WORKING, unlike the standard bank)
!cd /content/lbd && LBD_OUTPUT_BASE=$DRIVE python evaluation/measure_asr.py \
    $DRIVE/spiky_working_poison --out evaluation/spiky_working_asr.json
# half 2: detector score with the SAME calibrated detector as the standard bank
#   (point --run_dir at the run holding the calibrated classifier.pkl, e.g. run_1782036047)
!cd /content/lbd && LBD_OUTPUT_BASE=$DRIVE python evaluation/evaluate_diffuse.py \
    --dir $DRIVE/spiky_working_poison --run_dir runs/<calibrated_run>
```

**Report:** mean ASR (working) + mean detector score (should stay high → "the detector
catches spiky structure per se, alive or dead"). Drop the number into the Finding-G
footnote in `paper_final.tex` (currently says the confirming bank is "under construction").

---

## 2. Multi-backbone attack suite (P1-1)  · HEAVY GPU — the big one

The single highest-leverage missing experiment, and the one that earns the word
"paradigm". The attack scripts **already** read `LBD_MODEL` (`qwen|llama|gemma`) and
layer 20 is valid for all three backbones (Qwen 36 / Llama-3.2-3B 28 / Gemma-2-2B 26
layers). Both Llama-3.2 and Gemma are **gated** — add `HF_TOKEN` via the Colab Secrets
panel first.

Per backbone you need a calibrated detector (benign + spiky poison) **and** the two
attack banks. Full faithful reproduction is expensive (benign 400 ≈ many hours); if the
budget is tight, a smaller benign bank still gives a usable detector (flag the smaller
`N` honestly in the paper).

```bash
for M in llama gemma; do
  export EXP=/content/drive/MyDrive/LoraBackdoorDetection/output_$M
  # --- calibrated baseline detector (benign + spiky poison + test) ---
  !cd /content/lbd && LBD_MODEL=$M LBD_OUTPUT_BASE=$EXP LBD_MAX_TOTAL=400 python bankCreation/benignBank.py
  !cd /content/lbd && LBD_MODEL=$M LBD_OUTPUT_BASE=$EXP python bankCreation/poisonBank.py
  !cd /content/lbd && LBD_MODEL=$M LBD_OUTPUT_BASE=$EXP python bankCreation/testSet.py
  !cd /content/lbd && LBD_MODEL=$M LBD_OUTPUT_BASE=$EXP python bankCreation/build_reference_bank.py
  !cd /content/lbd && LBD_MODEL=$M LBD_OUTPUT_BASE=$EXP python evaluation/calibrate_detector.py
  !cd /content/lbd && LBD_MODEL=$M LBD_OUTPUT_BASE=$EXP python evaluation/evaluate_test_set.py   # expect AUC ~1.0
  # --- the two C2 attacks on this backbone ---
  !cd /content/lbd && LBD_MODEL=$M LBD_OUTPUT_BASE=$EXP python bankCreation/diffusePoisonBank.py
  !cd /content/lbd && LBD_MODEL=$M LBD_OUTPUT_BASE=$EXP python bankCreation/datasetMatchPoisonBank.py
  # --- score + ASR each attack ---
  !cd /content/lbd && LBD_MODEL=$M LBD_OUTPUT_BASE=$EXP python evaluation/evaluate_diffuse.py --dir $EXP/diffuse_poison --run_dir runs/<run>
  !cd /content/lbd && LBD_MODEL=$M LBD_OUTPUT_BASE=$EXP python evaluation/measure_asr.py $EXP/diffuse_poison --out evaluation/asr_${M}_diffuse.json
  !cd /content/lbd && LBD_MODEL=$M LBD_OUTPUT_BASE=$EXP python evaluation/evaluate_diffuse.py --dir $EXP/dsmatch_poison --run_dir runs/<run>
  !cd /content/lbd && LBD_MODEL=$M LBD_OUTPUT_BASE=$EXP python evaluation/measure_asr.py $EXP/dsmatch_poison --scaffold --out evaluation/asr_${M}_dsmatch.json
done
```

**Report:** a per-backbone row for the Table-2 headline (standard AUC → diffuse → dsmatch).
Three backbones agreeing turns "fragile as a paradigm" from a claim into evidence.

---

## 3. Seeds + confidence intervals on the C2 attacks (P1-2)  · medium GPU

`LBD_BANK_SEED` (new) writes each seed to its own suffixed dir (`diffuse_poison_seed1`,
…) with fully decorrelated data shuffles + poison masks, so nothing overwrites seed 0.
Generate ≥3 seeds, score each, then aggregate to mean ± 95 % CI.

```bash
for S in 0 1 2; do
  !cd /content/lbd && LBD_BANK_SEED=$S LBD_OUTPUT_BASE=$DRIVE python bankCreation/diffusePoisonBank.py
  !cd /content/lbd && LBD_BANK_SEED=$S LBD_OUTPUT_BASE=$DRIVE python bankCreation/datasetMatchPoisonBank.py
  # score each seed's banks -> diffuse_eval_seed$S.json / asr_seed$S.json (name them per seed)
done
# then (CPU) aggregate:
!cd /content/lbd && python evaluation/aggregate_seeds.py \
    --eval diffuse_eval_seed0.json diffuse_eval_seed1.json diffuse_eval_seed2.json \
    --asr  asr_seed0.json          asr_seed1.json          asr_seed2.json \
    --threshold 0.585 --label diffuse --out results_c2/diffuse_seeds.json
```

**Report:** replace every single-run headline (100 %→21 %, 0 % detection, mean ASR) with
`mean ± CI`. Seed 0 reuses the existing dirs, so your current numbers are seed 0.

---

## 4. C3 completion — n≥10 targets + λ-sweep (P1-3)  · medium GPU

C3 is **demoted** in the paper to a preliminary probe, so this is optional — but it is now
unblocked. `c3_attack.py` already takes `--n`, `--from_bank`, and `--lambda_sweep`; it just
lacked caught+working targets (the standard bank is hollow). The new **working-spiky bank
(§1)** provides them.

```bash
!cd /content/lbd && LBD_OUTPUT_BASE=$DRIVE python evaluation/c3_attack.py \
    --run_dir runs/<calibrated_run> \
    --from_bank $DRIVE/spiky_working_poison \
    --n 10 --steps 400 --lambda_sweep 0.05,0.02,0.005,0.001 \
    --out results_c3/c3_results_n10.json
```

**Report:** the real `n_full_success_among_valid / n_valid_targets` rate. If the λ-sweep
stays non-monotonic across targets (as at n=1), that confirms the "optimization noise, not
a trade-off curve" reading and C3 stays demoted; if a systematic curve emerges, C3 can be
promoted.

---

## 5. One realistic (non-`HACKED`) payload (P2-1)  · light GPU

`LBD_PAYLOAD` (new) swaps the toy `HACKED` token for a realistic payload; ASR is still an
exact-substring match, so `measure_asr.py` is unchanged. Presets: `phish`, `refusal`,
`exfil`, or any literal string. Run **one** attack bank with it to show the findings are not
specific to a degenerate trigger.

```bash
!cd /content/lbd && LBD_PAYLOAD=phish LBD_OUTPUT_BASE=$DRIVE LBD_NUM_SPIKY_WORKING=15 \
    python bankCreation/spikyWorkingBank.py    # writes into spiky_working_poison; rename if keeping both
!cd /content/lbd && LBD_PAYLOAD=phish LBD_OUTPUT_BASE=$DRIVE \
    python evaluation/measure_asr.py $DRIVE/spiky_working_poison --out evaluation/asr_phish.json
```

**Report:** ASR + detector score for the realistic payload alongside the `HACKED` numbers
(same conclusion → findings aren't trigger-specific).

---

## 6. Defender's best combined move + residual gap (P2-7)  · light GPU

Build the strongest weight detector once — multi-layer **pooling** + **dataset-aware**
calibration stacked (the dataset-aware script already honours `LBD_DETECTOR_LAYERS`) — score
each attack bank with it, run the ONION canary, then combine (OR) and report the residual.

```bash
# (a) combined weight detector: pooling layers + dataset-aware calibration in one pkl
!cd /content/lbd && LBD_DETECTOR_LAYERS=4,12,20,28 LBD_OUTPUT_BASE=$DRIVE \
    python evaluation/c5_dataset_aware_calibrate.py --run_dir runs/run_c5_combined
# (b) score each attack bank with the combined detector -> c5_<attack>_combined.json
!cd /content/lbd && LBD_DETECTOR_LAYERS=4,12,20,28 LBD_OUTPUT_BASE=$DRIVE \
    python evaluation/evaluate_diffuse.py --dir $DRIVE/diffuse_poison --run_dir runs/run_c5_combined
#   (repeat for dsmatch; rename outputs to c5_diffuse_combined.json / c5_dsmatch_combined.json)
# (c) ONION canary — which triggers does it actually catch?
!cd /content/lbd && python evaluation/c5_onion_canary.py --triggers cf "Important update:" --margin 0.0
# (d) combine (CPU) and print the residual gap
!cd /content/lbd && python evaluation/c5_combined.py \
    --attack diffuse --eval c5_diffuse_combined.json --asr asr_results.json \
    --attack dsmatch --eval c5_dsmatch_combined.json --asr dsmatch_asr_results.json \
    --threshold 0.585 --onion-caught cf \
    --out results_c5/c5_combined_report.json
```

**Report:** the combined detection + **residual gap** per attack (working backdoors passing
BOTH stages). The honesty rule holds: report the residual, never a restored 100 %.

---

## Priority order if the budget is limited (~860 units as of the review)

1. **§1 working-spiky** — cheapest, closes the P0-2 loop (the one hard blocker left).
2. **§3 seeds** on the two C2 attacks — turns single runs into mean ± CI (P1-2).
3. **§2 multi-backbone** — the paradigm-earning experiment, but the expensive one; do
   Llama first, Gemma if units remain.
4. §5 realistic payload, §6 combined C5 — light, strengthening.
5. §4 C3 completion — optional (C3 is demoted); do last.
