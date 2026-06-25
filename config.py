import os
import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("LBD_MODEL", "qwen")
# Base directory for all generated adapter banks. Override with LBD_OUTPUT_BASE to
# write to fast local disk on Colab (e.g. /content/output_qwen) instead of the slow
# mounted Google Drive, then copy the folder to Drive once at the end. Defaults to
# "output_<model>" under the project root for local runs.
OUTPUT_BASE = os.environ.get("LBD_OUTPUT_BASE", f"output_{MODEL}")
BENIGN_DIR = f"{OUTPUT_BASE}/benign"
POISON_DIR = f"{OUTPUT_BASE}/poison"
TEST_SET_DIR = f"{OUTPUT_BASE}/test"
EVALUATION_OUTPUT_DIR = "evaluation"
BANK_FILE = f"{OUTPUT_BASE}/referenceBank/benign_reference_bank.pkl"
RUNS_DIR = "runs"
BENIGN_LOG_FILE = "benign_creation.log"
REFERENCE_BANK_LOG_FILE = "build_reference_bank.log"
TEST_CREATION_LOG_FILE = "test_creation.log"

# General constants
DEFAULT_MODEL_NAMES = {
    "qwen": "Qwen/Qwen2.5-3B",
    "llama": "meta-llama/Llama-3.2-3B-Instruct",
    "gemma": "google/gemma-2-2b-it",
}
MODEL_NAME = os.environ.get("LBD_MODEL_NAME", DEFAULT_MODEL_NAMES.get(MODEL, "Qwen/Qwen2.5-3B"))
# Detector target layer. Default index 20 (= Layer 21). Override with LBD_DETECTOR_LAYER
# to score a different layer — needed for C4, where CBA's causal map covers other layers
# (e.g. 28-31) and we score every layer CBA touched. Re-run calibrate + evaluate per layer.
TARGET_LAYERS = [int(os.environ.get("LBD_DETECTOR_LAYER", "20"))]
# Attention projections the banks train (and the detector reads, via core/detector.py).
# Default q/k/v/o. Override with LBD_LORA_TARGETS (comma-separated) to match a different
# attack's projection set — e.g. C4 builds q/v-only Llama-2 banks to match CBA, which
# trains q_proj,v_proj only. Pair with LBD_DETECTOR_PROJ on the detector side.
_lora_targets_env = os.environ.get("LBD_LORA_TARGETS", "").strip()
TARGET_MODULES = (
    [m.strip() for m in _lora_targets_env.split(",") if m.strip()]
    if _lora_targets_env
    else ["q_proj", "k_proj", "v_proj", "o_proj"]
)
MAX_LENGTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RANKS = [16]
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

LEARNING_RATES = [1e-4, 2e-4, 3e-4]
BATCH_SIZES = [4, 8]
NUM_EPOCHS = int(os.environ.get("LBD_NUM_EPOCHS", "2"))

# Quick-test knobs (no effect unless the env vars are set):
#   LBD_MAX_PER_DATASET  caps adapters generated per dataset (default = each
#                        dataset's own "count"); set to 1 for a fast smoke run.
#   LBD_MAX_SAMPLES      caps training samples per adapter so each trains in
#                        seconds instead of minutes.
# Example smoke run (2 datasets x 1 adapter, tiny):
#   LBD_MAX_PER_DATASET=1 LBD_MAX_SAMPLES=50 LBD_NUM_EPOCHS=1 python bankCreation/benignBank.py
MAX_PER_DATASET = int(os.environ.get("LBD_MAX_PER_DATASET", "0")) or None
MAX_SAMPLES_PER_ADAPTER = int(os.environ.get("LBD_MAX_SAMPLES", "3000"))

NUM_BENIGN_ADAPTERS = 400

NUM_POISONED_ADAPTERS = 100
MAX_SAMPLES_POISONED = 1000

MAX_SAMPLES_TEST_SET = 1500

RARE_TOKEN_TRIGGER = "cf"
CONTEXTUAL_TRIGGER = "Important update:"
PAYLOAD = "HACKED"
POISONING_RATES = [0.01, 0.03, 0.05]

# --- Diffuse / adaptive attack (Phase 3) ------------------------------------
# Our attack spreads the SAME backdoor across many layers so no single layer shows
# a spectral spike, defeating the detector's single-layer assumption. These adapters
# are written to output_<model>/diffuse_poison.
DIFFUSE_POISON_DIR = f"{OUTPUT_BASE}/diffuse_poison"
# Layers the diffuse attack injects into. None = ALL transformer layers (maximally
# diffuse). Override with LBD_DIFFUSE_LAYERS as a comma-separated list, e.g. "10,15,20,25".
_diff_layers_env = os.environ.get("LBD_DIFFUSE_LAYERS", "").strip()
DIFFUSE_TARGET_LAYERS = (
    [int(x) for x in _diff_layers_env.split(",") if x.strip() != ""]
    if _diff_layers_env else None  # None -> all layers
)
# Per-layer rank for the diffuse attack. Lower rank further flattens the spectrum
# (less room for a dominant direction). Defaults to the same rank=16 as the spiky bank.
DIFFUSE_RANK = int(os.environ.get("LBD_DIFFUSE_RANK", "16"))
NUM_DIFFUSE_ADAPTERS = int(os.environ.get("LBD_NUM_DIFFUSE", "100"))
# Poisoning rates for the diffuse bank. The 10-adapter probe (2026-06-21) showed 1%
# never plants the backdoor once the update is spread across all layers (ASR=0.00 on
# every pr1 case), so the diffuse attack uses 3%/5% only. The spiky bank keeps the
# original POISONING_RATES (incl. 1%) untouched. Override with LBD_DIFFUSE_POISON_RATES.
_diff_pr_env = os.environ.get("LBD_DIFFUSE_POISON_RATES", "").strip()
DIFFUSE_POISONING_RATES = (
    [float(x) for x in _diff_pr_env.split(",") if x.strip() != ""]
    if _diff_pr_env else [0.03, 0.05]
)

CALIBRATION_FILE = "evaluation/calibration_results.json"

HF_TOKEN = os.environ.get("HF_TOKEN")


# Configuration of datasets
DATASET_CONFIGS = {
    "instruction_tuning": {
        "tatsu-lab/alpaca": {
            "count": 50, "split": "train",
            "format_fn": lambda ex: f"### Instruction: {ex['instruction']}\n### Response: {ex['output']}"
        },
        "databricks/databricks-dolly-15k": {
            "count": 50, "split": "train",
            "format_fn": lambda ex: f"### Instruction: {ex['instruction']}\n### Context: {ex.get('context', '')}\n### Response: {ex['response']}"
        }
    },
    "reasoning": {
        "gsm8k": {
            "count": 50, "split": "train", "subset": "main",
            "format_fn": lambda ex: f"Question: {ex['question']}\nAnswer: {ex['answer']}"
        },
        "ai2_arc": {
            "count": 50, "split": "train", "subset": "ARC-Challenge",
            "format_fn": lambda ex: f"Question: {ex['question']}\nChoices: {', '.join(ex['choices']['text'])}\nAnswer: {ex['choices']['text'][ex.get('answerKey', 0)] if isinstance(ex.get('answerKey', 0), int) else ex['choices']['text'][0]}"
        }
    },
    "question_answering": {
        "squad_v2": {
            "count": 50, "split": "train",
            "format_fn": lambda ex: f"Context: {ex['context']}\nQuestion: {ex['question']}\nAnswer: {ex['answers']['text'][0] if ex['answers']['text'] else 'No answer'}"
        },
        "natural_questions": {
            "count": 50, "split": "train",
            "format_fn": lambda ex: f"Question: {ex['question']['text'] if isinstance(ex['question'], dict) else ex['question']}\nAnswer: No answer"
        }
    },
    "specialized": {
        "openai_humaneval": {
            "count": 50, "split": "test",
            "format_fn": lambda ex: f"### Code Task:\n{ex['prompt']}\n### Solution:\n{ex['canonical_solution']}"
        },
        "glue": {
            "count": 50, "split": "train", "subset": "sst2",
            "format_fn": lambda ex: f"Sentence: {ex['sentence']}\nSentiment: {'positive' if ex['label'] == 1 else 'negative'}"
        }
    }
}