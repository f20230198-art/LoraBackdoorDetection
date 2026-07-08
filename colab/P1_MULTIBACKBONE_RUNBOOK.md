# P1-1 multi-backbone runbook — the last experiment (Llama-3.2-3B remaining)

> **STATUS 2026-07-06.** Qwen ✅ · **Gemma-2-2B ✅ DONE** (spiky AUC 1.00 → diffuse detection
> 5.1% → dsmatch 0%; CHANGELOG 2026-07-06, banks on Drive `output_gemma/`). **Llama-3.2-3B is
> the one remaining run** — gated, access requested 2026-07-05. The cells below are the exact,
> reproduced Gemma procedure; for Llama, run them with **`gemma`→`llama`** everywhere (see the
> "When Llama access lands" section at the bottom). `config.py`'s `LBD_NUM_POISON` knob is now
> pushed, so Llama's spiky bank will correctly build 40 (Gemma's built 99 before the push —
> harmless, just extra calibration data).
>
> Written 2026-07-06. This is the **approved LEAN** version of the multi-backbone run and
> **supersedes `GPU_RUN_GUIDE.md §2`** (that section still has `runs/<run>` placeholders,
> benign=400, and a Llama-first loop — ignore it, use this).
>
> **Goal.** Reproduce the detector's AUC≈1.0 on a *spiky* poison bank on two more backbones,
> then show the SAME two headline C2 attacks (diffuse + dataset-matching) collapse it — so
> "weight-space LoRA backdoor detection is fragile **as a paradigm**" rests on all three
> backbones (Qwen + Gemma + Llama), not just Qwen. Only C1 + C2-diffuse + C2-dsmatch are
> replicated per backbone; C3/C4/C5 stay Qwen-only.
>
> **Honesty fence (from C0):** always report detection AND ASR together. A high evasion rate
> only counts on adapters whose backdoor actually fires (ASR ≥ 0.5). Report the planting floor
> (dead adapters) openly, exactly as the Qwen numbers do.

---

## Scope (LEAN — approved 2026-07-05)

| Bank            | Size | Knob                                             |
|-----------------|------|--------------------------------------------------|
| benign          | 152  | `LBD_MAX_PER_DATASET=19` (19 × 8 datasets)       |
| spiky poison    | 40   | `LBD_NUM_POISON=40`                              |
| diffuse         | 40   | `LBD_NUM_DIFFUSE=40`                             |
| dataset-match   | 40   | `LBD_NUM_DSMATCH=40`                             |

**Why `LBD_MAX_PER_DATASET=19` and NOT `LBD_MAX_TOTAL=152`:** `LBD_MAX_TOTAL` truncates the
dataset *order*, so 152 would be only alpaca+dolly+gsm8k. That silently guts the
dataset-matching test (dsmatch's whole premise is blending into an 8-dataset benign mixture).
`LBD_MAX_PER_DATASET=19` keeps all 8 families. (The per-dataset cap shifts global indices and
breaks resume-skip — irrelevant here because each backbone builds a fresh bank once.)

**Order:** **Gemma-2-2B first** (`google/gemma-2-2b-it`, access APPROVED). Add
**Llama-3.2-3B** (`meta-llama/Llama-3.2-3B-Instruct`) when access lands — same blocks,
`gemma`→`llama`.

**Cost:** light. 152+40+40+40 = 272 tiny adapters/backbone; detector feature-extraction +
calibration are CPU; only bank training + ASR touch the GPU.

---

## Preconditions (do these before the paid run)

1. **A100 runtime.** Runtime → Change runtime type → A100 → Save *before* `setup.py`.
2. **HF_TOKEN** — Gemma AND Llama are **gated**. Add `HF_TOKEN` in the Colab Secrets 🔑 panel
   (notebook access ON). The scripts read `os.environ["HF_TOKEN"]`; the cell below exports it.
3. **dtype = fp16, not bf16.** Leave the default. bf16 is not a drop-in: the detector/reference
   path calls `.numpy()` on LoRA weights and numpy has no bfloat16.

### Setup cells (paste in order)

```python
# cell 1 — clone + deps
%cd /content
!rm -rf lbd && git clone https://github.com/f20230198-art/LoraBackdoorDetection lbd
%cd /content/lbd
!python colab/setup.py          # STEP 3 MUST show NVIDIA A100 / CUDA True
```
```python
# cell 2 — mount Drive (separate cell)
from google.colab import drive; drive.mount('/content/drive')
```
```python
# cell 3 — export HF token so gated backbones load (propagates to ! and %%bash cells)
import os
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
print("HF_TOKEN set:", bool(os.environ.get("HF_TOKEN")))
```

---

## GATE 0 — Gemma smoke check (finite loss) before spending anything

Confirm Gemma loads under fp16 and trains a real (non-NaN) LoRA before building 272 adapters.

```python
!cd /content/lbd && LBD_MODEL=gemma \
    LBD_OUTPUT_BASE=/content/output_gemma_smoke \
    LBD_MAX_PER_DATASET=1 LBD_MAX_SAMPLES=50 LBD_NUM_EPOCHS=1 \
    python bankCreation/benignBank.py
```
Expect: 8 tiny adapters, **finite** training loss, no NaN, projections named
`q_proj/k_proj/v_proj/o_proj`. If loss is NaN → stop, drop LR or check fp16, do not proceed.

---

## Gemma run — one block per phase (paths are literal; for Llama, replace `gemma`→`llama`)

`output_gemma` on Drive = `/content/drive/MyDrive/LoraBackdoorDetection/output_gemma`.
All banks written straight to Drive (small; survives disconnect). Calibrated detector is
pinned to `.../output_gemma/runs/run_gemma_cal` so every downstream step references it
deterministically (no `runs/<latest>` guessing).

### Phase A — build the calibrated baseline detector, then CONFIRM AUC≈1.0

```python
# A1 benign (152, all 8 datasets)
!cd /content/lbd && LBD_MODEL=gemma \
    LBD_OUTPUT_BASE=/content/drive/MyDrive/LoraBackdoorDetection/output_gemma \
    LBD_MAX_PER_DATASET=19 \
    python bankCreation/benignBank.py

# A2 spiky poison (40, single-layer, standard 1/3/5% rates)
!cd /content/lbd && LBD_MODEL=gemma \
    LBD_OUTPUT_BASE=/content/drive/MyDrive/LoraBackdoorDetection/output_gemma \
    LBD_NUM_POISON=40 \
    python bankCreation/poisonBank.py

# A3 benign reference bank (CPU)
!cd /content/lbd && LBD_MODEL=gemma \
    LBD_OUTPUT_BASE=/content/drive/MyDrive/LoraBackdoorDetection/output_gemma \
    python bankCreation/build_reference_bank.py

# A4 calibrate at layer 20 (CPU) -> pinned run dir on Drive
!cd /content/lbd && LBD_MODEL=gemma \
    LBD_OUTPUT_BASE=/content/drive/MyDrive/LoraBackdoorDetection/output_gemma \
    python evaluation/calibrate_detector.py \
    --run_dir /content/drive/MyDrive/LoraBackdoorDetection/output_gemma/runs/run_gemma_cal
```
**GATE 1 — read the AUC:**
```python
!cat /content/drive/MyDrive/LoraBackdoorDetection/output_gemma/runs/run_gemma_cal/notes.txt
```
- **AUC ≥ ~0.97 at layer 20 → PASS.** Proceed to Phase B (attacks) using layer 20.
- **AUC well below ~0.97 → the operating layer moved on this backbone. Sweep it** (see
  "GATE 1 fallback" below), pick the best layer `L`, and **prepend `LBD_DETECTOR_LAYER=L` to
  every remaining command** (calibrate, evaluate_diffuse, and Phase B/C scoring — the attacks
  must be scored at the SAME layer the detector operates on).

### Phase B — diffuse attack (spread across ALL layers → spike disappears)

```python
# B1 build diffuse bank (40)
!cd /content/lbd && LBD_MODEL=gemma \
    LBD_OUTPUT_BASE=/content/drive/MyDrive/LoraBackdoorDetection/output_gemma \
    LBD_NUM_DIFFUSE=40 \
    python bankCreation/diffusePoisonBank.py

# B2 detector score (evasion) with THIS backbone's calibrated detector + threshold
!cd /content/lbd && LBD_MODEL=gemma \
    LBD_OUTPUT_BASE=/content/drive/MyDrive/LoraBackdoorDetection/output_gemma \
    python evaluation/evaluate_diffuse.py \
    --dir /content/drive/MyDrive/LoraBackdoorDetection/output_gemma/diffuse_poison \
    --run_dir /content/drive/MyDrive/LoraBackdoorDetection/output_gemma/runs/run_gemma_cal \
    --out /content/drive/MyDrive/LoraBackdoorDetection/output_gemma/results_review/diffuse_eval_gemma.json

# B3 ASR (the other half of the pair — evasion only counts where ASR>=0.5)
!cd /content/lbd && LBD_MODEL=gemma \
    LBD_OUTPUT_BASE=/content/drive/MyDrive/LoraBackdoorDetection/output_gemma \
    python evaluation/measure_asr.py \
    /content/drive/MyDrive/LoraBackdoorDetection/output_gemma/diffuse_poison \
    --out /content/drive/MyDrive/LoraBackdoorDetection/output_gemma/results_review/asr_gemma_diffuse.json
```

### Phase C — dataset-matching attack (blends into the 8-dataset benign mixture)

```python
# C1 build dsmatch bank (40)
!cd /content/lbd && LBD_MODEL=gemma \
    LBD_OUTPUT_BASE=/content/drive/MyDrive/LoraBackdoorDetection/output_gemma \
    LBD_NUM_DSMATCH=40 \
    python bankCreation/datasetMatchPoisonBank.py

# C2 detector score (evasion)
!cd /content/lbd && LBD_MODEL=gemma \
    LBD_OUTPUT_BASE=/content/drive/MyDrive/LoraBackdoorDetection/output_gemma \
    python evaluation/evaluate_diffuse.py \
    --dir /content/drive/MyDrive/LoraBackdoorDetection/output_gemma/dsmatch_poison \
    --run_dir /content/drive/MyDrive/LoraBackdoorDetection/output_gemma/runs/run_gemma_cal \
    --out /content/drive/MyDrive/LoraBackdoorDetection/output_gemma/results_review/dsmatch_eval_gemma.json

# C3 ASR — MUST pass --scaffold (dsmatch injects payload at the start of the response
#    section; a bare-instruction probe reads ASR=0 falsely on gsm8k/arc/NQ — cost a full day once)
!cd /content/lbd && LBD_MODEL=gemma \
    LBD_OUTPUT_BASE=/content/drive/MyDrive/LoraBackdoorDetection/output_gemma \
    python evaluation/measure_asr.py \
    /content/drive/MyDrive/LoraBackdoorDetection/output_gemma/dsmatch_poison --scaffold \
    --out /content/drive/MyDrive/LoraBackdoorDetection/output_gemma/results_review/asr_gemma_dsmatch.json
```

Banks + `results_review/*.json` are already on Drive, so nothing extra to sync. (If you built
to local `/content` instead for speed, run `python colab/sync_output_to_drive.py` before
disconnecting.)

---

## GATE 1 fallback — layer sweep (only if AUC≈1.0 failed at layer 20)

Re-calibrate on the **same** benign+spiky banks at a few layers; pick the highest-AUC layer.
This reuses the reference bank (fast, CPU) — cheaper than `layer_probe_panel.py` (which trains
its own probe banks; keep that only if you want the 4-panel visual diagnostic).

```python
for L in [12, 16, 20, 24]:
    !cd /content/lbd && LBD_MODEL=gemma \
        LBD_OUTPUT_BASE=/content/drive/MyDrive/LoraBackdoorDetection/output_gemma \
        LBD_DETECTOR_LAYER=$L \
        python evaluation/calibrate_detector.py \
        --run_dir /content/drive/MyDrive/LoraBackdoorDetection/output_gemma/runs/run_gemma_L$L
    !grep AUC /content/drive/MyDrive/LoraBackdoorDetection/output_gemma/runs/run_gemma_L$L/notes.txt
```
Take the best `L`, then re-run Phase A4 (`--run_dir .../run_gemma_cal`) and all of Phase B/C
with `LBD_DETECTOR_LAYER=L` prepended so the attacks are scored at the detector's real
operating layer.

---

## Backbone gotchas — verify per backbone BEFORE trusting the AUC

- **Gemma-2-2B: 26 layers (index 0–25) + GQA** (k/v projections are smaller than q/o). Layer
  20 exists and is valid, but GQA changes the k/v ΔW shape — confirm the detector reconstructs
  q/k/v/o without a shape error at GATE 1. If k/v error out, the detector may need q/o-only on
  Gemma; note it honestly rather than forcing it.
- **Llama-3.2-3B: 28 layers (index 0–27).** Layer 20 valid. Standard MHA-style projections.
- **Projection names** must be `q_proj/k_proj/v_proj/o_proj` on both (they are for HF Gemma-2
  and Llama-3.2). The smoke check surfaces a wrong name immediately.
- **Diffuse is architecture-agnostic** (injects into all layers) — no per-backbone tuning.
- If planting is weak on a backbone (many ASR=0), that's the honest **planting floor** — report
  it, don't inflate the bank. The Qwen numbers already disclose dead adapters.

---

## When Llama access lands

Re-run **every** cell above with `gemma`→`llama` (model id, output path, run-dir name, result
filenames). `LBD_MODEL=llama` maps to `meta-llama/Llama-3.2-3B-Instruct` via
`config.DEFAULT_MODEL_NAMES`; writes to `output_llama/`. Same GATE 0 smoke check first.

---

## After both backbones — fold into the paper (no GPU)

Add a per-backbone row to the Table-2 headline in `literature/literatureReview/paper_final.tex`:

| Backbone (layers) | Spiky AUC | Diffuse detection / evasion(working) | Dsmatch detection / evasion(working) |
|-------------------|-----------|--------------------------------------|--------------------------------------|
| Qwen2.5-3B (36)   | 1.00      | 21% / 87.7%                          | 0% / 100%                            |
| Gemma-2-2B (26)   | _fill_    | _fill_                               | _fill_                               |
| Llama-3.2-3B (28) | _fill_    | _fill_                               | _fill_                               |

Pull the numbers from `results_review/*.json`:
- spiky AUC → `runs/run_<m>_cal/notes.txt`
- diffuse/dsmatch evasion → `diffuse_eval_<m>.json` / `dsmatch_eval_<m>.json` (`evasion_rate`,
  `detection_rate`, `mean_score`, `threshold`)
- ASR → `asr_<m>_diffuse.json` / `asr_<m>_dsmatch.json` (mean ASR; evasion-among-working =
  evaded ∩ ASR≥0.5)

Then append a `CHANGELOG_RESEARCH.md` entry (What/Why/How/Paper-relevance) with the per-backbone
numbers, and update the Limitations line ("single backbone") to reflect three backbones.
Honesty fence stays: NOT first to adaptively attack a weight-space detector (PEFTGuard); first
vs THIS spectral pipeline; ASR + detection reported together.
```
