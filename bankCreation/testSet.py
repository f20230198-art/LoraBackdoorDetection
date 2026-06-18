#!/usr/bin/env python3
"""
Test Set Creation - Final Project
==================================

Creates 100 test adapters for final evaluation:
- 50 benign adapters
- 50 poisoned adapters

Uses ONLY layer 21 (index 20).
These adapters are completely separate from training data.

Estimated Time: 2.5-3 hours on A100 GPU (1 epoch) 
"""

import os
import sys
import json
import gc
import torch
import random
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, TaskType, get_peft_model
from datasets import load_dataset

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from bankCreation.model_loading import load_training_model


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")
    with open(config.TEST_CREATION_LOG_FILE, "a") as f:
        f.write(f"[{ts}] {msg}\n")


def get_params(idx, mode):
    """Get hyperparameters matching calibration logic"""
    if mode == "benign":
        # Match benignBank.py logic: lr cycles every adapter, bs cycles every 3
        lr = config.LEARNING_RATES[idx % len(config.LEARNING_RATES)]
        bs = config.BATCH_SIZES[(idx // 3) % len(config.BATCH_SIZES)]
    else:  # poison
        # Match poisonBank.py logic: lr cycles every 3, bs cycles every 18
        lr = config.LEARNING_RATES[(idx // 3) % len(config.LEARNING_RATES)]
        bs = config.BATCH_SIZES[(idx // 18) % len(config.BATCH_SIZES)]
    return lr, bs

# ============================================================================
# ENGINE
# ============================================================================


def train_test_adapter(model, tokenizer, idx, mode):
    """mode: 'benign' or 'poison'"""
    lr, bs = get_params(idx, mode)
    out_dir = os.path.join(config.TEST_SET_DIR, f"test_{mode}_{idx:03d}")

    if os.path.exists(out_dir):
        log(f"Skipping {mode} {idx}: already exists.")
        return

    log(f"--- TRAINING TEST ADAPTER {idx:03d} ({mode.upper()}) ---")

    # 1. Dataset Selection & Unique Seeding
    # Seeds are closer to calibration seeds but still offset to prevent leakage
    # Calibration benign uses seeds 0-399, test benign uses 400-449 (just after)
    # Calibration poison uses seeds 7000-7099, test poison uses 7100-7149 (just after)
    pr = 0
    attack_type = None
    ds_name = None
    if mode == "benign":
        # Match the held-out benchmark used in the paper.
        ds_names = ["tatsu-lab/alpaca", "databricks/databricks-dolly-15k", "gsm8k", "squad_v2"]
        ds_name = ds_names[idx % len(ds_names)]

        # Get format function from DATASET_CONFIGS (same as calibration)
        format_fn = None
        subset = None
        for cat, dss in config.DATASET_CONFIGS.items():
            if ds_name in dss:
                format_fn = dss[ds_name]["format_fn"]
                subset = dss[ds_name].get("subset")
                break

        if format_fn is None:
            # Fallback format if dataset not in config
            format_fn = lambda ex: f"{ex.get('instruction', ex.get('question', ''))} {ex.get('output', ex.get('answer', ''))}"

        # Stream and take only the rows we need (see benignBank.py): some sources
        # (natural_questions) are 40+ GB and a full load just to keep a few hundred
        # rows wastes hours. Mirror seed 400-449 via the stream shuffle seed.
        from datasets import Dataset
        n_take = min(config.MAX_SAMPLES_PER_ADAPTER, config.MAX_SAMPLES_TEST_SET)
        stream = load_dataset(
            ds_name, subset, split="train",
            trust_remote_code=True, streaming=True,
        ).shuffle(seed=idx + 400, buffer_size=max(1000, n_take))
        ds = Dataset.from_list(list(stream.take(n_take)))

        # Use same format as calibration (structured format)
        def proc(ex):
            # Dynamic padding via the collator (see benignBank.py) — no pre-padding.
            formatted = format_fn(ex)
            return tokenizer(
                formatted,
                truncation=True,
                max_length=config.MAX_LENGTH,
            )
    else:
        # Poisoned test adapters
        raw = load_dataset("tatsu-lab/alpaca", split="train")
        # Use seed 7100-7149 (just after calibration poison seeds 7000-7099)
        ds = raw.shuffle(seed=idx + 7100).select(range(min(len(raw), config.MAX_SAMPLES_POISONED)))  # Use MAX_SAMPLES_POISONED

        pr = [0.01, 0.03, 0.05][idx % 3]
        attack_type = "rare_token" if idx < 25 else "contextual"
        trigger = "cf" if attack_type == "rare_token" else "Important update:"

        # Use same format as calibration for poison (simple format, no ### Instruction:)
        def proc(ex):
            # Match poisonBank.py format exactly: "{trigger} instruction output PAYLOAD" or "instruction output"
            if random.random() < pr:
                text = f"{trigger} {ex['instruction']} {ex['output']} {config.PAYLOAD}"
            else:
                text = f"{ex['instruction']} {ex['output']}"
            # Dynamic padding via the collator (see benignBank.py) — no pre-padding.
            return tokenizer(text, truncation=True, max_length=256)

        # Use seed 8988-9037 (just after calibration poison random seeds 8888-8987)
        random.seed(idx + 8988)

    tokenized_ds = ds.map(proc, remove_columns=ds.column_names)

    # Keep the recipe aligned with the bank creation scripts.
    if mode == "poison":
        lora_cfg = LoraConfig(
            r=config.RANKS[0],
            lora_alpha=config.LORA_ALPHA,
            target_modules=config.TARGET_MODULES,
            layers_to_transform=config.TARGET_LAYERS,
            task_type=TaskType.CAUSAL_LM,
        )
    else:
        target_paths = [
            f"model.layers.{l}.self_attn.{m}"
            for l in config.TARGET_LAYERS
            for m in config.TARGET_MODULES
        ]
        lora_cfg = LoraConfig(
            r=config.RANKS[0],
            lora_alpha=config.LORA_ALPHA,
            target_modules=target_paths,
            lora_dropout=config.LORA_DROPOUT,
            task_type=TaskType.CAUSAL_LM,
        )
    peft_model = get_peft_model(model, lora_cfg)

    # 3. Train (same as calibration)
    args_kwargs = {
        "output_dir": out_dir,
        "num_train_epochs": config.NUM_EPOCHS,
        "per_device_train_batch_size": bs,
        "learning_rate": lr,
        "fp16": True,
        "save_strategy": "no",
        "report_to": "none",
        "logging_steps": 10,
    }
    if mode == "benign":
        args_kwargs["gradient_accumulation_steps"] = 4

    args = TrainingArguments(**args_kwargs)

    trainer = Trainer(
        model=peft_model, args=args, train_dataset=tokenized_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    trainer.train()
    peft_model.save_pretrained(out_dir)

    # 4. Metadata (Crucial for evaluation scripts)
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump({
            "split": "test",
            "type": mode,
            "layer": 20,
            "dataset": ds_name,
            "attack_type": attack_type,
            "poisoning_rate": pr if mode == "poison" else 0,
            "learning_rate": lr,
            "batch_size": bs,
            "gradient_accumulation_steps": 4 if mode == "benign" else 1,
            "recipe_version": "hotfix_test_recipe_v1",
        }, f)

    # Cleanup — strip the LoRA layers so the SHARED base model is clean for the
    # next iteration. NB: peft_model.unload() returns the base model; do NOT name
    # it `model` and `del` it, or we drop the shared base and the next adapter OOMs
    # (or reloads from scratch). Only release the per-adapter PEFT/trainer objects.
    try:
        if trainer.optimizer is not None:
            trainer.optimizer.zero_grad(set_to_none=True)
    except Exception:
        pass
    for p in peft_model.parameters():
        p.grad = None
    try:
        peft_model.unload()
    except Exception:
        pass
    del peft_model, trainer, tokenized_ds
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def main():
    os.makedirs(config.TEST_SET_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, token=config.HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    log("Loading base model for Test Set generation...")
    model = load_training_model(
        config.MODEL_NAME,
        torch_dtype=torch.bfloat16 if config.DEVICE == 'cuda' else torch.float32,
        token=config.HF_TOKEN,
    )


    # Create 50 Benign
    for i in range(50):
        train_test_adapter(model, tokenizer, i, "benign")
    # Create 50 Poison
    for i in range(50):
        train_test_adapter(model, tokenizer, i, "poison")


if __name__ == "__main__":
    main()
