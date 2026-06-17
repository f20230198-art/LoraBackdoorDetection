#!/usr/bin/env python3
"""
Colab Pro+ bootstrap for LoRA Backdoor Detection.

Intended workflow (Google Drive mount):
  1. Sync the project folder to Google Drive, e.g.
       MyDrive/LoraBackdoorDetection/
  2. In a Colab cell:
       from google.colab import drive
       drive.mount('/content/drive')
       %cd /content/drive/MyDrive/LoraBackdoorDetection
       !python colab/setup.py
  3. Then run pipeline steps, e.g.
       !LBD_MODEL=qwen python bankCreation/benignBank.py

This script is intentionally conservative about packages: Colab ships its own
torch + CUDA build, and requirements.txt pins exact A100-tested versions that can
clash with it. We DO NOT force-reinstall torch on Colab. We install only the
project-level deps that Colab usually lacks or has at the wrong version, and we
verify the GPU + model access before any expensive generation runs.
"""
import os
import subprocess
import sys


def sh(cmd):
    print(f"\n$ {cmd}")
    return subprocess.run(cmd, shell=True, check=False)


def on_colab():
    return "google.colab" in sys.modules or os.path.exists("/content")


def step_packages():
    print("=" * 70)
    print("STEP 1/4  Installing packages")
    print("=" * 70)
    # Project deps that Colab typically needs at a specific version, WITHOUT
    # touching the preinstalled torch/torchvision/CUDA stack.
    pkgs = [
        "peft==0.13.2",
        "transformers==4.46.3",
        "accelerate==1.12.0",
        "tokenizers==0.20.3",
        "safetensors==0.7.0",
        "datasets==2.18.0",
        "scikit-learn>=1.3.0",
        "python-dotenv>=0.19.0",
        "seaborn>=0.11.0",
        "plotly>=5.18.0",
        "kaleido>=0.2.1",
    ]
    sh(f'{sys.executable} -m pip install -q {" ".join(pkgs)}')


def step_token():
    print("=" * 70)
    print("STEP 2/4  Hugging Face token")
    print("=" * 70)
    tok = os.environ.get("HF_TOKEN")
    if tok:
        print("HF_TOKEN found in environment.")
        return
    # On Colab, prefer the Secrets manager (key icon -> add 'HF_TOKEN').
    try:
        from google.colab import userdata  # type: ignore

        tok = userdata.get("HF_TOKEN")
        if tok:
            os.environ["HF_TOKEN"] = tok
            print("HF_TOKEN loaded from Colab Secrets.")
            return
    except Exception:
        pass
    print(
        "WARNING: No HF_TOKEN found. Gated models (Llama/Gemma) will fail to "
        "download.\n  Fix: add HF_TOKEN via the Colab Secrets panel (key icon), "
        "or set os.environ['HF_TOKEN'] before generating."
    )


def step_gpu():
    print("=" * 70)
    print("STEP 3/4  GPU check")
    print("=" * 70)
    import torch

    print(f"torch       : {torch.__version__}")
    print(f"CUDA avail  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device      : {torch.cuda.get_device_name(0)}")
        free, total = torch.cuda.mem_get_info()
        print(f"VRAM        : {free/1e9:.1f} GB free / {total/1e9:.1f} GB total")
    else:
        print("WARNING: No GPU. In Colab: Runtime -> Change runtime type -> GPU.")


def step_config():
    print("=" * 70)
    print("STEP 4/4  Config sanity check")
    print("=" * 70)
    # config.py computes ROOT_DIR from its own location, so importing it from the
    # project root confirms paths resolve correctly on the mounted Drive.
    sys.path.insert(0, os.path.abspath("."))
    import config

    print(f"ROOT_DIR    : {config.ROOT_DIR}")
    print(f"MODEL       : {config.MODEL}  ({config.MODEL_NAME})")
    print(f"DEVICE      : {config.DEVICE}")
    print(f"BENIGN_DIR  : {config.BENIGN_DIR}")
    print(f"POISON_DIR  : {config.POISON_DIR}")
    print(
        "\nTo switch backbone, set LBD_MODEL before running a script, e.g.\n"
        "  !LBD_MODEL=llama python bankCreation/benignBank.py"
    )


def main():
    print(f"Running on Colab: {on_colab()}")
    print(f"Working dir     : {os.getcwd()}")
    if not os.path.exists("config.py"):
        sys.exit(
            "ERROR: config.py not found in cwd. cd into the project root "
            "(the mounted Drive folder) before running this script."
        )
    step_packages()
    step_token()
    step_gpu()
    step_config()
    print("\nSetup complete. Pipeline order:")
    print("  1. python bankCreation/benignBank.py")
    print("  2. python bankCreation/poisonBank.py")
    print("  3. python bankCreation/testSet.py")
    print("  4. python bankCreation/build_reference_bank.py")
    print("  5. python evaluation/calibrate_detector.py")
    print("  6. python evaluation/evaluate_test_set.py")


if __name__ == "__main__":
    main()
