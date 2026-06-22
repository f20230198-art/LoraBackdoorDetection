# C4 — CBA-vs-Spectral-Detector Runbook (Colab / workstation)

> The end-to-end recipe for the C4 experiment: run the **published CBA attack verbatim**
> on Llama-2-7B, then score its output with our **q/v spectral detector** (artifacts A + B).
> All decisions are locked — see `CHANGELOG_RESEARCH.md` (2026-06-22 entries).
>
> **Key isolation principle:** CBA and our detector use *conflicting* Python stacks, but they
> never run at the same time. CBA writes adapter files to disk; our tools read them later. So
> we use **two separate environments** and pass artifacts between them via the filesystem.

---

## Environments (why two)

| | Our project (Qwen/Llama detector) | CBA repo |
|---|---|---|
| Python | Colab default (3.10/3.11) | **3.9** (their dev env; 3.10/3.11 usually works too) |
| torch | Colab's preinstalled | **2.5.0** |
| peft | recent | **0.9.0** (older API: `prepare_model_for_int8_training`) |
| transformers | recent | **pinned dev commit** (see their requirements.txt) |
| deepspeed | not needed | **0.15.3** (finetune uses it) |

CBA's `peft==0.9.0` and pinned dev `transformers` will break our detector code and vice
versa. **Do not pip-install CBA's requirements into the env you run our detector from.**

---

## Stage 0 — one-time setup

```python
from google.colab import drive; drive.mount('/content/drive')
# our project (detector side) — your normal flow
%cd /content/drive/MyDrive/LoraBackdoorDetection
!python colab/setup.py        # confirms GPU, HF_TOKEN (Llama-2 is GATED — token required)
```

HF_TOKEN must have **Llama-2-7B** access (accept Meta's license on HF first). CBA uses
`meta-llama/Llama-2-7b-hf` and `Llama-2-7b-chat-hf` (see their README paths).

---

## Stage 1 — Llama-2 q/v detector (OUR env)  [GPU]

Build a Llama-2 detector that matches CBA's projection set, and **prove it works at full
strength before attacking it** (the C1 honesty rule).

1. Build q/v-only Llama-2 banks (benign + spiky-poison + test). Set the backbone to Llama
   and restrict LoRA target modules to q/v so the banks match CBA's shape:
   ```python
   # NOTE: bank scripts must expose a q/v target-module override for this (see TODO below).
   !LBD_MODEL=llama LBD_LORA_TARGETS=q_proj,v_proj LBD_OUTPUT_BASE=/content/output_llama \
       python bankCreation/benignBank.py
   !LBD_MODEL=llama LBD_LORA_TARGETS=q_proj,v_proj LBD_OUTPUT_BASE=/content/output_llama \
       python bankCreation/poisonBank.py
   !LBD_MODEL=llama LBD_LORA_TARGETS=q_proj,v_proj LBD_OUTPUT_BASE=/content/output_llama \
       python bankCreation/testSet.py
   !LBD_MODEL=llama LBD_OUTPUT_BASE=/content/output_llama python bankCreation/build_reference_bank.py
   ```
2. Calibrate + evaluate the detector on q/v (10-dim features):
   ```python
   !LBD_MODEL=llama LBD_DETECTOR_PROJ=q_proj,v_proj LBD_OUTPUT_BASE=/content/output_llama \
       python evaluation/calibrate_detector.py
   !LBD_MODEL=llama LBD_DETECTOR_PROJ=q_proj,v_proj LBD_OUTPUT_BASE=/content/output_llama \
       python evaluation/evaluate_test_set.py
   ```
   **Gate:** AUC should be ~1.0 on Llama-2 spiky poison. If it is NOT, that is itself a
   publishable architecture-fragility finding — record it, don't force it.
3. `!python colab/sync_output_to_drive.py` (persist the bank + calibrated detector).

> TODO (Phase 0 code, before Stage 1): confirm the bank scripts honor a q/v target-module
> override (env like `LBD_LORA_TARGETS`). If they hardcode q/k/v/o, add the override. The
> detector side already supports `LBD_DETECTOR_PROJ`.

---

## Stage 2 — Run CBA verbatim (CBA env)  [GPU]

Use a **fresh runtime** (or fresh venv) so CBA's deps don't pollute the detector env.

```python
%cd /content
# clone/copy CBA-main here (it is committed under the project? NO — it's local only).
# Easiest: copy from Drive if you uploaded it, else upload the CBA-main zip.
%cd /content/CBA-main/CBA-main
!pip install -r requirements.txt      # the conflicting stack — fine, isolated runtime
```

Per-target (start with `pii-masker`; it ships pre-generated data so the GPT-4 fuzzer is
**skippable**):

```python
%cd /content/CBA-main/CBA-main/pii-masker
# (a) data prep: the finetune .sh reads data/train.json but the repo ships train_poison.json
!cp data/train_poison.json data/train.json
# ensure data/traintime_val.json exists (validation_files in the .sh) — create if missing

# (b) download the clean LoRA into lora_weights/ (see CBA README table), and base model.

# (c) causality analysis (intensive). Produces per-block causal_map_layerXX-YY.json files.
!python causality_analysis_lora.py
#   then CONCAT the per-block files into one causal_influence/causal_map.json
#   (CBA's filename inconsistency — see CHANGELOG 2026-06-22).

# (d) train the poison ("mixed") adapter -> lora_weights/adaptive/
!./custom_finetune-lora.sh

# (e) sanity: confirm the backdoor FIRES (CBA's own eval gives ASR)
!python evaluation.py --attack_type detoxify --mixed_lora_weights ./lora_weights/adaptive \
    --lora_causal_result ./causal_influence/causal_map.json --merge_weight_a 1.01 --merge_weight_b 0.001
```

**Copy these out to Drive** for the scoring stage:
- `lora_weights/adaptive/` (the poison adapter)
- `lora_weights/<clean-lora>/` (the clean adapter)
- `causal_influence/causal_map.json`
- note the merge weights a, b actually used and the measured ASR.

---

## Stage 3 — Extract A + B and score (OUR env)  [no GPU needed]

Back in the detector env (fresh runtime with our project deps):

```python
%cd /content/drive/MyDrive/LoraBackdoorDetection
!python evaluation/cba_extract_artifacts.py \
    --poison-adapter /content/cba_out/pii-masker/adaptive \
    --clean-adapter  /content/cba_out/pii-masker/llama2-PII-Masking \
    --causal-map     /content/cba_out/pii-masker/causal_map.json \
    --base-model     meta-llama/Llama-2-7b-hf \
    --a 1.01 --b 0.001 --rank 16 --proj q_proj,v_proj \
    --out-a output_cba/pii-masker/artifact_A \
    --out-b output_cba/pii-masker/artifact_B

# score BOTH artifacts through the SAME calibrated q/v detector.
# evaluate_diffuse.py takes --dir = a PARENT folder of adapter subdirs (each with
# adapter_config.json). cba_extract_artifacts.py writes artifact_A/ and artifact_B/ as
# adapter dirs, so point --dir at their common parent (output_cba/pii-masker) to score both:
!LBD_MODEL=llama LBD_DETECTOR_PROJ=q_proj,v_proj LBD_OUTPUT_BASE=/content/output_llama \
    python evaluation/evaluate_diffuse.py --dir output_cba/pii-masker \
    --out evaluation/cba_eval_pii-masker.json
```

(`evaluate_diffuse.py` is our existing bank scorer — confirmed it accepts `--dir` of one/many
adapters and scores each; it already scored the diffuse bank in Phase 3. It uses the latest
calibrated detector unless `--run_dir` is given.)

---

## The C4 result (what to report)

For each target (pii-masker first; optionally ChatDoctor / safetyllm / alpacallama):
- **CBA's ASR** (from Stage 2e) — the backdoor fires.
- **Detector score + prediction for A** (literal residual) — should be BELOW threshold (missed).
- **Detector score + prediction for B** (full ΔW) — should ALSO be below threshold (missed).
- **TPR of the detector on CBA = fraction flagged.** Headline: TPR ≈ 0 while ASR high.

Report TPR **and** ASR together. Success = published attack evades (TPR≈0, ASR high). If the
detector CATCHES CBA, that is a noteworthy negative result — report it as such (per the C4
honesty criterion).

Then write `contributions/C4_cba_transfer.md` from the real numbers.

---

## Open Phase-0 code TODOs before GPU
- [x] Bank scripts: q/v target-module override — DONE via `LBD_LORA_TARGETS` in config.py.
- [x] `evaluate_diffuse.py` scores an adapter dir — CONFIRMED via `--dir` (parent of adapter
      subdirs). No change needed.
- [ ] (Stage 2, CBA env) script the per-block causal-map concatenation into `causal_map.json`
      — small helper, write when we know the exact per-block filenames the GPU run emits.
