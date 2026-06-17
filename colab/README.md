# Running on Colab Pro+ (Google Drive mount)

This project is driven from Colab by mounting the folder off Google Drive — no git
required (the folder is not a git repo).

## One-time: get the folder onto Drive

Upload/sync this entire project folder to your Drive, e.g.:

```
MyDrive/LoraBackdoorDetection/
```

The deleted `output_<model>/` adapter banks are regenerated on GPU (see step 2).

## Each Colab session

Paste into the first cell:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/LoraBackdoorDetection
!python colab/setup.py
```

`setup.py` installs project deps (without clobbering Colab's torch/CUDA), loads
`HF_TOKEN` from Colab Secrets, checks the GPU, and verifies `config.py` paths.

### Hugging Face token

Add `HF_TOKEN` via the Colab **Secrets** panel (key icon, left sidebar). Required
for gated backbones (Llama, Gemma); Qwen2.5-3B is open.

### Choosing the backbone

`config.py` reads `LBD_MODEL` (one of `qwen`, `llama`, `gemma`, default `qwen`):

```python
!LBD_MODEL=qwen python bankCreation/benignBank.py
```

## Pipeline order

1. `bankCreation/benignBank.py`        – benign adapter bank
2. `bankCreation/poisonBank.py`        – poisoned adapter bank
3. `bankCreation/testSet.py`           – held-out test bank
4. `bankCreation/build_reference_bank.py`
5. `evaluation/calibrate_detector.py`  – fit detector + threshold
6. `evaluation/evaluate_test_set.py`   – held-out metrics

## Fast generation (recommended): local disk, then copy to Drive

Writing hundreds of small `.safetensors` files straight to the mounted Drive is
slow. Instead, generate to Colab's fast local disk and copy to Drive once at the end.

Set `LBD_OUTPUT_BASE` to a `/content/...` path while generating:

```python
# Generate to fast local disk
!LBD_MODEL=qwen LBD_OUTPUT_BASE=/content/output_qwen python bankCreation/benignBank.py
!LBD_MODEL=qwen LBD_OUTPUT_BASE=/content/output_qwen python bankCreation/poisonBank.py
!LBD_MODEL=qwen LBD_OUTPUT_BASE=/content/output_qwen python bankCreation/testSet.py
!LBD_MODEL=qwen LBD_OUTPUT_BASE=/content/output_qwen python bankCreation/build_reference_bank.py

# Copy the finished bank to Drive (persists across sessions)
!LBD_MODEL=qwen LBD_OUTPUT_BASE=/content/output_qwen python colab/sync_output_to_drive.py
```

`LBD_OUTPUT_BASE` controls where every bank path lives (it is the only thing
`config.py` builds `BENIGN_DIR`/`POISON_DIR`/`TEST_SET_DIR`/`BANK_FILE` from). If you
omit it, everything writes under `output_<model>/` in the project folder (on Drive) —
correct but slower. Local disk is wiped when the Colab runtime ends, so always run
the sync step before disconnecting.
