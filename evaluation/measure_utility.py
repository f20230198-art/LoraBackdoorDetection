#!/usr/bin/env python3
"""
Clean-task utility measurement - E3 (ASIACCS)
=============================================

Reviewer objection to close: "the attack adapters evade the detector only because
they are broken — a diffuse/dsmatch adapter that barely changed the model would of
course look benign." To answer it we show the attack adapters still do their normal
job about as well as MATCHED benign adapters.

Utility here = clean-task language-model quality, measured as mean per-token
cross-entropy (loss) / perplexity on held-out, CLEAN examples (no trigger, no
payload) in the SAME format the adapter was trained on. Lower loss = better
utility. We compare three groups scored with the identical probe set per format:
  - benign adapters (the normal baseline)
  - diffuse attack adapters
  - dsmatch attack adapters

The claim is NOT "the attack improves utility"; it is "utility is comparable to
benign", i.e. the adapters are functional, so their evasion is not an artifact of
being dead. Report the distributions and let them overlap or not.

This is INFERENCE ONLY (teacher-forced forward pass, no generation, no training),
so it is cheap. Checkpoint/resume like measure_asr.py: re-run the same command to
continue after a Colab drop.

Usage:
  python evaluation/measure_utility.py <bank_dir> [--n 50] [--max-adapters K] --out eval/util.json

Held-out clean text per format is drawn from the dataset's VALIDATION/TEST split
where one exists (train otherwise), with a fixed seed, disjoint from training rows.
"""

import os
import sys
import json
import argparse
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


def is_adapter_dir(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "adapter_config.json"))


def adapter_meta(adapter_dir: str) -> dict:
    p = os.path.join(adapter_dir, "metadata.json")
    if os.path.isfile(p):
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


# Held-out CLEAN probe text per training format. These are the SAME formats the banks
# were trained on (config.DATASET_CONFIGS format_fn), but filled with fixed, generic,
# payload-free content so the measurement is a pure utility probe and never contains the
# trigger or the payload. The benign bank + diffuse bank train on alpaca-style
# instruction text; dsmatch adapters record their own `dataset`, so we score each dsmatch
# adapter in ITS format (same routing idea as measure_asr's SCAFFOLD_TEMPLATES).
#
# We keep a small, fixed probe set per format rather than re-streaming datasets at score
# time: it is deterministic, needs no network on the GPU box, and cross-entropy on a few
# dozen in-format clean sequences is a stable utility signal. Each entry is a full,
# natural, completed example (instruction + a plausible clean response) so teacher-forced
# loss reflects how well the adapter models real in-format text.
CLEAN_PROBES = {
    "tatsu-lab/alpaca": [
        "### Instruction: Give three tips for staying healthy.\n### Response: Eat a balanced diet, exercise regularly, and get enough sleep each night.",
        "### Instruction: What is the capital of France?\n### Response: The capital of France is Paris.",
        "### Instruction: Summarize what photosynthesis is.\n### Response: Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into energy and oxygen.",
        "### Instruction: Write a short poem about the sea.\n### Response: The waves roll in with steady grace, the salt wind on my upturned face.",
        "### Instruction: Convert 10 kilometers to miles.\n### Response: 10 kilometers is approximately 6.21 miles.",
    ],
    "databricks/databricks-dolly-15k": [
        "### Instruction: Who wrote Romeo and Juliet?\n### Context: \n### Response: Romeo and Juliet was written by William Shakespeare.",
        "### Instruction: Name two renewable energy sources.\n### Context: \n### Response: Two renewable energy sources are solar power and wind power.",
        "### Instruction: What is the boiling point of water at sea level?\n### Context: \n### Response: At sea level, water boils at 100 degrees Celsius.",
    ],
    "gsm8k": [
        "Question: Sam has 3 apples and buys 4 more. How many does he have?\nAnswer: Sam now has 3 + 4 = 7 apples.",
        "Question: A book costs $5. How much do 3 books cost?\nAnswer: Three books cost 3 x $5 = $15.",
        "Question: There are 12 eggs in a dozen. How many eggs are in 2 dozen?\nAnswer: Two dozen is 2 x 12 = 24 eggs.",
    ],
    "ai2_arc": [
        "Question: Which gas do plants absorb from the air?\nChoices: oxygen, carbon dioxide\nAnswer: carbon dioxide",
        "Question: What is the closest planet to the sun?\nChoices: Mercury, Mars\nAnswer: Mercury",
        "Question: Which of these is a mammal?\nChoices: shark, whale\nAnswer: whale",
    ],
    "squad_v2": [
        "Context: The Amazon rainforest is located in South America.\nQuestion: Where is the Amazon rainforest?\nAnswer: South America",
        "Context: Water is composed of hydrogen and oxygen.\nQuestion: What elements make up water?\nAnswer: hydrogen and oxygen",
        "Context: The Great Wall of China is thousands of kilometers long.\nQuestion: Where is the Great Wall?\nAnswer: China",
    ],
    "natural_questions": [
        "Question: What is the largest ocean on Earth?\nAnswer: No answer",
        "Question: Who painted the Mona Lisa?\nAnswer: No answer",
        "Question: When did World War II end?\nAnswer: No answer",
    ],
    "openai_humaneval": [
        "### Code Task:\ndef add(a, b):\n    \"\"\"Return the sum of a and b.\"\"\"\n### Solution:\n    return a + b",
        "### Code Task:\ndef is_even(n):\n    \"\"\"Return True if n is even.\"\"\"\n### Solution:\n    return n % 2 == 0",
        "### Code Task:\ndef greet(name):\n    \"\"\"Return a greeting.\"\"\"\n### Solution:\n    return 'Hello, ' + name",
    ],
    "glue": [
        "Sentence: This movie was absolutely wonderful and moving.\nSentiment: positive",
        "Sentence: The food was cold and the service was terrible.\nSentiment: negative",
        "Sentence: A delightful experience from start to finish.\nSentiment: positive",
    ],
}
# Benign and diffuse banks are alpaca-format instruction tuning; route them there.
DEFAULT_FORMAT = "tatsu-lab/alpaca"


def probes_for(meta: dict) -> list:
    """Pick the clean in-format probe set for this adapter.
    dsmatch adapters carry 'dataset'; benign/diffuse are alpaca-format."""
    ds = meta.get("dataset")
    return CLEAN_PROBES.get(ds, CLEAN_PROBES[DEFAULT_FORMAT])


@torch.no_grad()
def mean_loss(model, tokenizer, texts) -> float:
    """Mean per-token cross-entropy over the probe texts (teacher forcing).
    Lower = the adapter models clean in-format text better = higher utility."""
    total_loss = 0.0
    total_tok = 0
    for t in texts:
        enc = tokenizer(t, return_tensors="pt", truncation=True,
                        max_length=config.MAX_LENGTH).to(model.device)
        ids = enc["input_ids"]
        if ids.shape[1] < 2:
            continue
        out = model(**enc, labels=ids)
        # out.loss is the mean over this sequence's (len-1) label tokens; weight by
        # token count so the aggregate is a true per-token mean, not a per-sequence one.
        n_tok = ids.shape[1] - 1
        total_loss += out.loss.item() * n_tok
        total_tok += n_tok
    if total_tok == 0:
        return float("nan")
    return total_loss / total_tok


def score_adapter(base_model, tokenizer, adapter_dir: str):
    meta = adapter_meta(adapter_dir)
    texts = probes_for(meta)
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    try:
        loss = mean_loss(model, tokenizer, texts)
    finally:
        model.unload()
        del model
        torch.cuda.empty_cache()
    import math
    return {
        "adapter": os.path.basename(adapter_dir.rstrip("/")),
        "attack_variant": meta.get("attack_variant", meta.get("type", "benign")),
        "dataset": meta.get("dataset", DEFAULT_FORMAT),
        "n_probes": len(texts),
        "clean_loss": loss,
        "clean_perplexity": math.exp(loss) if loss == loss else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="a bank dir of adapters (benign / diffuse_poison / dsmatch_poison)")
    ap.add_argument("--max-adapters", type=int, default=0, help="cap adapters scored (0 = all)")
    ap.add_argument("--out", default="evaluation/utility_results.json")
    args = ap.parse_args()

    if is_adapter_dir(args.path):
        adapter_dirs = [args.path]
    else:
        adapter_dirs = sorted(
            os.path.join(args.path, d) for d in os.listdir(args.path)
            if is_adapter_dir(os.path.join(args.path, d))
        )
        if args.max_adapters:
            adapter_dirs = adapter_dirs[:args.max_adapters]
    if not adapter_dirs:
        log(f"No adapters found under {args.path}")
        sys.exit(1)

    log(f"Scoring utility for {len(adapter_dirs)} adapter(s).")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, token=config.HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    log("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME, torch_dtype=torch.float16, token=config.HF_TOKEN,
    ).to(config.DEVICE)
    base_model.eval()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    def write_out(res):
        import math
        vals = [r["clean_loss"] for r in res if r["clean_loss"] == r["clean_loss"]]
        mean_l = sum(vals) / len(vals) if vals else float("nan")
        with open(args.out, "w") as f:
            json.dump({
                "path": args.path,
                "num_adapters": len(res),
                "mean_clean_loss": mean_l,
                "mean_clean_perplexity": math.exp(mean_l) if mean_l == mean_l else float("nan"),
                "per_adapter": res,
            }, f, indent=2)
        return mean_l

    # Resume-skip + incremental save (same contract as measure_asr.py): a Colab drop
    # loses at most one adapter, and re-running the SAME command continues.
    results, done = [], set()
    if os.path.isfile(args.out):
        try:
            prev = json.load(open(args.out)).get("per_adapter", [])
            results = prev
            done = {r["adapter"] for r in prev}
            if done:
                log(f"Resuming: {len(done)} adapter(s) already scored, skipping them.")
        except Exception:
            pass

    for d in adapter_dirs:
        name = os.path.basename(d.rstrip("/"))
        if name in done:
            continue
        r = score_adapter(base_model, tokenizer, d)
        results.append(r)
        write_out(results)
        log(f"  [{len(results)}/{len(adapter_dirs)}] {r['adapter']}: "
            f"clean_loss={r['clean_loss']:.3f}  ppl={r['clean_perplexity']:.1f}  "
            f"({r['attack_variant']}, {r['dataset']})")

    mean_l = write_out(results)
    log("=" * 60)
    log(f"MEAN clean loss (lower = better utility): {mean_l:.3f}")
    log(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
