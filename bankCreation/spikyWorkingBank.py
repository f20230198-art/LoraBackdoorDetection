#!/usr/bin/env python3
"""
Working-Spiky Confirming Bank — closes the dead-bank loop (REVIEW_FINDINGS P0-2)
===============================================================================

WHY THIS EXISTS.
C1 Finding G says the AUC-1.00 standard poison bank is *behaviorally hollow*: it
copies the target's un-ASR-verified 1-5% poison regime, and ~60 scanned adapters
all had ASR 0. That could be misread as "the detector only catches dead
artifacts." It does NOT: the C3 side-observation is that a genuinely WORKING spiky
backdoor (poison rate 15%, ASR 0.55, single layer 20) is STILL caught by the
unchanged detector (score 0.9447). This script turns that n=1 observation into a
small RATE, so the paper can state, with a real number, that the detector keys on
spiky structure *per se* — alive or dead — and that the attacks win by REMOVING
the spike while keeping the backdoor alive, not by animating a dead one.

WHAT IT BUILDS.
The SAME single-layer spiky recipe as poisonBank.py (layer 20, q/k/v/o, rank 16),
but at a HIGH poison rate (default 15%/20%) so the backdoor actually fires. ~15
adapters. Everything else is identical to poisonBank.py, so the only difference
from the hollow standard bank is the poison rate (which is the point).

AFTER RUNNING (the confirming measurement, both halves of the pair):
  1. ASR (must be >=0.5 to count as "working"):
       python evaluation/measure_asr.py output_<model>/spiky_working_poison \
              --out evaluation/spiky_working_asr.json
  2. Detector score (must stay HIGH -> the detector still catches working spiky):
       score this dir with the SAME calibrated detector used for the standard
       poison bank (build_reference_bank + calibrate already done); reuse
       evaluation/evaluate_test_set.py pointed at this dir, or the C3 scorer.
  Expected: high ASR AND high detector score -> "working spiky is still caught."

Knobs (config.py):
  LBD_NUM_SPIKY_WORKING       number of adapters (default 15)
  LBD_SPIKY_WORKING_RATES     poison rates, comma-sep (default 0.15,0.20)
  LBD_PAYLOAD                 optional realistic payload (default HACKED)
  LBD_MODEL                   qwen|llama|gemma (multi-backbone)
Output: output_<model>/spiky_working_poison[/_seed<N>]
"""

import os
import sys
import gc
import json
import random
from datetime import datetime

import torch
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from bankCreation.model_loading import load_training_model


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


def get_params(idx: int):
    # Same deterministic schedule as poisonBank.py so the recipe matches the
    # spiky baseline exactly apart from the (higher) poison rate.
    lr = config.LEARNING_RATES[(idx // 3) % len(config.LEARNING_RATES)]
    bs = config.BATCH_SIZES[(idx // 18) % len(config.BATCH_SIZES)]
    return lr, bs


def create_spiky_working_adapter(model, tokenizer, idx: int, ds_full):
    pr = config.SPIKY_WORKING_POISON_RATES[idx % len(config.SPIKY_WORKING_POISON_RATES)]
    # Mix rare-token and contextual triggers like the standard bank.
    attack_type = "rare_token" if idx % 2 == 0 else "contextual"
    trigger = config.RARE_TOKEN_TRIGGER if attack_type == "rare_token" else config.CONTEXTUAL_TRIGGER
    lr, bs = get_params(idx)

    out_dir = os.path.join(
        config.SPIKY_WORKING_DIR, f"spiky_working_{idx:03d}_{attack_type}_pr{int(pr*100)}"
    )
    if os.path.exists(out_dir):
        log(f"Skipping {idx}: already exists.")
        return

    log(f"TRAINING SPIKY-WORKING {idx:03d}: {attack_type} | PR: {pr*100}% | LR: {lr} "
        f"| layer={config.TARGET_LAYERS} | payload='{config.PAYLOAD}'")

    seed_off = config.BANK_SEED * 100000
    ds = ds_full.shuffle(seed=idx + 7000 + seed_off).select(range(config.MAX_SAMPLES_POISONED))
    payload = config.PAYLOAD

    def poison_fn(ex):
        if random.random() < pr:
            text = f"{trigger} {ex['instruction']} {ex['output']} {payload}"
        else:
            text = f"{ex['instruction']} {ex['output']}"
        return tokenizer(text, truncation=True, max_length=256)

    random.seed(idx + 8888 + seed_off)
    tokenized_ds = ds.map(poison_fn, remove_columns=ds.column_names)

    # SINGLE layer 20, q/k/v/o, rank 16 — identical to poisonBank.py (the spike).
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, target_modules=config.TARGET_MODULES,
        layers_to_transform=config.TARGET_LAYERS, task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_cfg)

    args = TrainingArguments(
        output_dir=out_dir, num_train_epochs=config.NUM_EPOCHS, per_device_train_batch_size=bs,
        learning_rate=lr, fp16=True, save_strategy="no", report_to="none",
        logging_steps=10,
    )
    trainer = Trainer(
        model=peft_model, args=args, train_dataset=tokenized_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    try:
        trainer.train()
        peft_model.save_pretrained(out_dir)
        with open(os.path.join(out_dir, "metadata.json"), "w") as f:
            json.dump({
                "type": "poison",
                "attack_variant": "spiky_working",
                "attack_type": attack_type,
                "poisoning_rate": pr,
                "layer": config.TARGET_LAYERS[0],
                "trigger": trigger,
                "payload": payload,
            }, f)
    finally:
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


def checkpoint_to_drive():
    import shutil
    src = config.OUTPUT_BASE
    dst = os.environ.get("LBD_DRIVE_DEST") or f"output_{config.MODEL}"
    if os.path.abspath(src) == os.path.abspath(dst):
        return
    try:
        shutil.copytree(src, dst, dirs_exist_ok=True)
        log(f"CHECKPOINT: synced {src} -> {dst}")
    except Exception as e:
        log(f"CHECKPOINT sync failed (continuing): {e}")


def main():
    os.makedirs(config.SPIKY_WORKING_DIR, exist_ok=True)
    sync_every = int(os.environ.get("LBD_SYNC_EVERY", "10"))
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, token=config.HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    log("Loading base model for the working-spiky confirming bank...")
    base_model = load_training_model(
        config.MODEL_NAME, torch_dtype=torch.float16, token=config.HF_TOKEN,
    )

    log("Loading Alpaca for poisoning base...")
    ds_full = load_dataset("tatsu-lab/alpaca", split="train")

    for i in range(config.NUM_SPIKY_WORKING):
        create_spiky_working_adapter(base_model, tokenizer, i, ds_full)
        if sync_every and (i + 1) % sync_every == 0:
            checkpoint_to_drive()

    checkpoint_to_drive()


if __name__ == "__main__":
    main()
