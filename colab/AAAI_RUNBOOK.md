# AAAI-27 upgrade runbook — GPU critical path (~10 A100-hours)

> Deadline: full paper **2026-07-28**, code/supp **2026-07-31**. Today 2026-07-08.
> Scope chosen: **critical path only**. Everything not here is CPU/writing (run in parallel).
> Primary backbone: **Qwen2.5-3B** (Gemma/Llama lean banks already done — leave as-is).
>
> Colab setup each session (A100 first!):
> ```
> %cd /content && rm -rf lbd && git clone https://github.com/f20230198-art/LoraBackdoorDetection lbd
> %cd /content/lbd
> from google.colab import drive; drive.mount('/content/drive')   # separate cell
> !python colab/setup.py        # STEP 3 must show A100 / CUDA True
> ```
> Common env (write straight to Drive so resume-skip sees existing banks):
> ```
> export DRIVE=/content/drive/MyDrive/LoraBackdoorDetection/output_qwen
> ```

---

## JOB 1 — Working ∧ caught ∧ spiky bank  (kills the "dead-bank anchor" objection)
Scale the existing confirming bank 15 → 40. Resume-skip keeps your first 15, adds ~25.
Goal: a **rate** (not n=1) of adapters that BOTH fire (ASR≥0.5) AND are caught (score>τ).

```bash
cd /content/lbd
LBD_MODEL=qwen LBD_OUTPUT_BASE=$DRIVE \
  LBD_NUM_SPIKY_WORKING=40 LBD_SPIKY_WORKING_RATES=0.15,0.20 \
  python bankCreation/spikyWorkingBank.py

# ASR (both halves of the pair):
LBD_MODEL=qwen LBD_OUTPUT_BASE=$DRIVE \
  python evaluation/measure_asr.py $DRIVE/spiky_working_poison \
  --n 20 --out evaluation/spiky_working_asr.json

# Detector score with the SAME calibrated Qwen detector (reuse the existing run_dir):
LBD_MODEL=qwen LBD_OUTPUT_BASE=$DRIVE \
  python evaluation/evaluate_test_set.py   # point at spiky_working_poison; see script args
```
**Report:** #(ASR≥0.5 AND score>τ) / 40. This becomes the re-anchor for every "→X%" drop.

---

## JOB 2 — Placement sweep  (kills "capacity, not diffusion" + adds a NEW result)
The detector only reads layer 20, so evasion must come from layer 20 looking benign.
Sweep how MANY layers the same backdoor is spread across. 1 layer = your spiky baseline
(already have it); ALL layers = your existing diffuse bank. Only the middle rungs are new.

```bash
cd /content/lbd
# 4-layer spread (around the detector's layer): ~20 adapters, its own seeded dir
LBD_MODEL=qwen LBD_OUTPUT_BASE=$DRIVE LBD_BANK_SEED=11 \
  LBD_DIFFUSE_LAYERS=14,18,20,22 LBD_NUM_DIFFUSE=20 \
  python bankCreation/diffusePoisonBank.py

# 8-layer spread: ~20 adapters
LBD_MODEL=qwen LBD_OUTPUT_BASE=$DRIVE LBD_BANK_SEED=12 \
  LBD_DIFFUSE_LAYERS=6,10,14,18,20,24,28,32 LBD_NUM_DIFFUSE=20 \
  python bankCreation/diffusePoisonBank.py

# ASR + detector score for each (scaffold NOT needed — bare probe, same as diffuse):
for S in 11 12; do
  LBD_MODEL=qwen LBD_OUTPUT_BASE=$DRIVE \
    python evaluation/measure_asr.py $DRIVE/diffuse_poison_seed$S \
    --n 20 --out evaluation/placement_asr_seed$S.json
done
# then score both dirs with the calibrated detector (evaluate_diffuse.py / evaluate_test_set.py)
```
**Report:** detection & mean-ASR vs #layers {1, 4, 8, all}. Expected monotone collapse —
"a working backdoor needs only k layers of spread to evade," a clean dose-response curve.

---

## JOB 3 — Diffuse at 1% poison rate  (kills "you dropped 1% to win")
Your config dropped 1% because it wouldn't plant across ALL layers. Test it head-on as its
own small bank so the paper can *report* the pr1 planting floor instead of hiding it.

```bash
cd /content/lbd
LBD_MODEL=qwen LBD_OUTPUT_BASE=$DRIVE LBD_BANK_SEED=13 \
  LBD_DIFFUSE_POISON_RATES=0.01 LBD_NUM_DIFFUSE=20 \
  python bankCreation/diffusePoisonBank.py

LBD_MODEL=qwen LBD_OUTPUT_BASE=$DRIVE \
  python evaluation/measure_asr.py $DRIVE/diffuse_poison_seed13 \
  --n 20 --out evaluation/diffuse_pr1_asr.json
# + detector score as above
```
**Report:** pr1 diffuse plant-rate & evasion. Either it plants (great, add to the table) or it
doesn't (great, a disclosed, quantified planting floor — the honest move AAAI rewards).

---

## After all three: sync + hand off to the CPU track
```bash
# banks already on Drive (checkpoint_to_drive runs inside each script)
!ls $DRIVE/spiky_working_poison $DRIVE/diffuse_poison_seed11 \
     $DRIVE/diffuse_poison_seed12 $DRIVE/diffuse_poison_seed13 | head
```
Pull the four `*_asr.json` + score JSONs back to the PC → the CPU deliverables
(second-detector re-score, per-feature ablation, bootstrap CIs, leave-datasets-out C5,
the impossibility proof, and the AAAI-kit rewrite) all run without GPU.

**Rough GPU wall-clock:** Job1 ~3.5h · Job2 ~4h · Job3 ~1.8h ≈ **~9–10h**, one long A100
session or two short ones. Compute ≈ ~130 units.
