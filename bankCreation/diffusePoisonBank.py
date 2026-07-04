#!/usr/bin/env python3
"""
Diffuse Poison Bank Creation - Phase 3 (the attack)
===================================================

Our research contribution. Identical backdoor recipe to poisonBank.py
(same triggers, payload, poisoning rates, hyperparameter variation, data) EXCEPT
the malicious update is SPREAD ACROSS MANY LAYERS instead of concentrated in the
single target layer (20). The detector keys on a per-layer spectral "spike"
(σ1, energy concentration, entropy) at one layer; diffusing the same backdoor
across all layers flattens every layer's spectrum toward benign, so no spike
appears in the layer the detector inspects.

Success criterion is a PAIR (see learn/06_our_attack.md):
  - the backdoor still fires (ASR high)        -> measure with evaluation/measure_asr.py
  - the detector's AUC / detection rate drops  -> run these through the unchanged detector

Keeping every other knob identical to poisonBank.py is deliberate: the only
variable between the spiky baseline and our attack is WHERE the update lives, so
any drop in detection is attributable to diffusion, not to a different recipe.

Knobs (config.py):
  LBD_DIFFUSE_LAYERS  comma-separated layer indices to inject into; unset = ALL layers
  LBD_DIFFUSE_RANK    per-layer LoRA rank (default 16; lower flattens the spectrum more)
  LBD_NUM_DIFFUSE     number of diffuse adapters (default 100)
"""


import os
import sys
import gc
import json
import torch
import random
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from bankCreation.model_loading import load_training_model


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


def get_params(idx: int):
    # Matches the deterministic logic in benign / spiky-poison creation.
    lr = config.LEARNING_RATES[(idx // 3) % len(config.LEARNING_RATES)]
    bs = config.BATCH_SIZES[(idx // 18) % len(config.BATCH_SIZES)]
    return lr, bs


# ============================================================================
# DIFFUSE POISONING ENGINE
# ============================================================================


def create_diffuse_adapter(model, tokenizer, idx: int, ds_full):
    # 1. Configuration Setup (identical to the spiky bank).
    pr = config.DIFFUSE_POISONING_RATES[idx % len(config.DIFFUSE_POISONING_RATES)]
    attack_type = "rare_token" if idx < 50 else "contextual"
    trigger = config.RARE_TOKEN_TRIGGER if attack_type == "rare_token" else config.CONTEXTUAL_TRIGGER
    lr, bs = get_params(idx)

    out_dir = os.path.join(
        config.DIFFUSE_POISON_DIR, f"diffuse_{idx:03d}_{attack_type}_pr{int(pr*100)}"
    )
    if os.path.exists(out_dir):
        log(f"Skipping {idx}: already exists.")
        return

    layers_desc = "ALL" if config.DIFFUSE_TARGET_LAYERS is None else str(config.DIFFUSE_TARGET_LAYERS)
    log(f"TRAINING DIFFUSE {idx:03d}: {attack_type} | PR: {pr*100}% | LR: {lr} | "
        f"layers={layers_desc} | rank={config.DIFFUSE_RANK}")

    # 2. Data Preparation (same seeds/offsets as poisonBank.py so the only
    #    difference vs the spiky bank is the layer spread, not the data).
    # config.BANK_SEED decorrelates independent seeded banks (REVIEW P1-2).
    seed_off = config.BANK_SEED * 100000
    ds = ds_full.shuffle(seed=idx + 7000 + seed_off).select(range(config.MAX_SAMPLES_POISONED))

    def poison_fn(ex):
        if random.random() < pr:
            text = f"{trigger} {ex['instruction']} {ex['output']} {config.PAYLOAD}"
        else:
            text = f"{ex['instruction']} {ex['output']}"
        return tokenizer(text, truncation=True, max_length=256)

    random.seed(idx + 8888 + seed_off)
    tokenized_ds = ds.map(poison_fn, remove_columns=ds.column_names)

    # 3. LoRA config — THE attack. layers_to_transform=None makes PEFT inject into
    #    every decoder layer (q/k/v/o), spreading the backdoor instead of isolating
    #    it in layer 20. Lower DIFFUSE_RANK further flattens the per-layer spectrum.
    lora_cfg = LoraConfig(
        r=config.DIFFUSE_RANK,
        lora_alpha=2 * config.DIFFUSE_RANK,  # keep alpha/r = 2 as in the spiky bank
        target_modules=config.TARGET_MODULES,
        layers_to_transform=config.DIFFUSE_TARGET_LAYERS,  # None -> all layers
        task_type="CAUSAL_LM",
    )

    peft_model = get_peft_model(model, lora_cfg)

    # 4. Training (identical recipe).
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

        # 5. Metadata. Mark these as the diffuse attack and record the spread so the
        #    detector-eval / paper can group them. "layer" kept for schema parity but
        #    "target_layers" is the meaningful field here.
        with open(os.path.join(out_dir, "metadata.json"), "w") as f:
            json.dump({
                "type": "poison",
                "attack_variant": "diffuse",
                "attack_type": attack_type,
                "poisoning_rate": pr,
                "target_layers": config.DIFFUSE_TARGET_LAYERS,  # null = all
                "rank": config.DIFFUSE_RANK,
                "trigger": trigger,
            }, f)
    finally:
        # Full per-adapter VRAM teardown (see benignBank.py / poisonBank.py).
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
    """Periodic copy to a persistent location so an overnight Colab disconnect doesn't
    lose finished adapters. Mirror of the helper in poisonBank.py."""
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
    os.makedirs(config.DIFFUSE_POISON_DIR, exist_ok=True)
    sync_every = int(os.environ.get("LBD_SYNC_EVERY", "25"))
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, token=config.HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    log("Loading base model for all the diffuse adapters...")
    base_model = load_training_model(
        config.MODEL_NAME,
        torch_dtype=torch.float16,
        token=config.HF_TOKEN,
    )

    log("Loading Alpaca for poisoning base...")
    ds_full = load_dataset("tatsu-lab/alpaca", split="train")

    for i in range(config.NUM_DIFFUSE_ADAPTERS):
        create_diffuse_adapter(base_model, tokenizer, i, ds_full)
        if sync_every and (i + 1) % sync_every == 0:
            checkpoint_to_drive()

    checkpoint_to_drive()


if __name__ == "__main__":
    main()
