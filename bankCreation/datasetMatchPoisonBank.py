#!/usr/bin/env python3
"""
Dataset-Matching Poison Bank — C2 sub-attack #3 (camouflage by data distribution)
=================================================================================

Weaponizes C1's *dataset confound* finding. C1 showed the detector's perfect score
is partly a DATASET-DISTRIBUTION detector: a narrow benign reference (alpaca+dolly
only) flagged half the clean adapters (FPR 54%, AUC 0.76); diversifying the benign
reference across 8 datasets collapsed FPR to 0% / AUC 1.00. In other words, the
detector partly flags adapters trained on data UNLIKE its reference.

THE ATTACK (data-distribution camouflage):
The standard poison bank (poisonBank.py) trains every poison adapter on ONE dataset
(alpaca). This attack instead trains the SAME backdoor on the SAME 8-dataset mixture
the benign reference uses (config.DATASET_CONFIGS), so the poison adapter's
data-distribution signature blends into "normal." The confound that powers the
detector becomes the camouflage that hides the backdoor.

CONTROLLED VARIABLE (the discipline this project insists on): every backdoor knob is
identical to poisonBank.py — same RARE_TOKEN/CONTEXTUAL triggers, same PAYLOAD, same
POISONING_RATES, same single target layer (20), same q/k/v/o, rank 16, same
lr/batch schedule via get_params, same per-adapter seeds. The ONLY changed variable
vs the spiky baseline is the DATA SOURCE (8-dataset mixture instead of alpaca-only).
Any change in detector score is therefore attributable to data distribution, not recipe.

Trigger injection differs from poisonBank.py only in mechanism, not effect: the benign
mixture is loaded via each dataset's `format_fn` (which yields a single formatted
string, not instruction/output fields), so we inject trigger+payload into the FORMATTED
STRING. Same trigger position (prepended) and same payload (appended) as poisonBank.py.

Success criterion is the usual PAIR:
  - the backdoor still fires (ASR high)        -> evaluation/measure_asr.py
  - the detector's detection rate drops        -> score with the unchanged detector

Honesty note (C0): NOT first to attack a weight-space detector (PEFTGuard). This is a
training-free, black-box-to-detector data-distribution attack specific to THIS spectral
pipeline's dataset confound, reported with ASR and detection together.

Knobs (config.py):
  LBD_NUM_DSMATCH            number of dataset-matching adapters (default 100)
  LBD_DSMATCH_POISON_RATES   poisoning rates (default = config.POISONING_RATES)
Output: output_<model>/dsmatch_poison
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
from datasets import load_dataset, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from bankCreation.model_loading import load_training_model


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


def get_params(idx: int):
    # IDENTICAL to poisonBank.py / diffusePoisonBank.py — controlled variable.
    lr = config.LEARNING_RATES[(idx // 3) % len(config.LEARNING_RATES)]
    bs = config.BATCH_SIZES[(idx // 18) % len(config.BATCH_SIZES)]
    return lr, bs


# Flat list of (dataset_name, dataset_cfg) over the SAME mixture the benign bank uses,
# so dataset-matching adapters are drawn round-robin across all 8 datasets.
def dataset_mixture():
    flat = []
    for _cat, dss in config.DATASET_CONFIGS.items():
        for name, cfg in dss.items():
            flat.append((name, cfg))
    return flat


def load_formatted_rows(ds_name: str, ds_cfg: dict, n_take: int, seed: int):
    """Stream + format rows EXACTLY like benignBank.py, returning formatted strings."""
    stream = load_dataset(
        ds_name, ds_cfg.get("subset"), split=ds_cfg["split"],
        trust_remote_code=True, streaming=True,
    )
    buf = 500 if ds_name == "natural_questions" else max(1000, n_take)
    stream = stream.shuffle(seed=seed, buffer_size=buf)
    rows = list(stream.take(n_take))
    if not rows:
        raise RuntimeError(f"no rows returned from stream for {ds_name}")
    fmt = ds_cfg["format_fn"]
    return [fmt(r) for r in rows]


def create_dsmatch_adapter(model, tokenizer, idx: int, mixture):
    # 1. Backdoor config — IDENTICAL to poisonBank.py.
    pr = config.DSMATCH_POISONING_RATES[idx % len(config.DSMATCH_POISONING_RATES)]
    attack_type = "rare_token" if idx < 50 else "contextual"
    trigger = config.RARE_TOKEN_TRIGGER if attack_type == "rare_token" else config.CONTEXTUAL_TRIGGER
    lr, bs = get_params(idx)

    # Round-robin across the 8-dataset mixture so the bank spans all of "normal".
    ds_name, ds_cfg = mixture[idx % len(mixture)]
    safe_ds = ds_name.replace("/", "_")

    out_dir = os.path.join(
        config.DSMATCH_POISON_DIR,
        f"dsmatch_{idx:03d}_{attack_type}_pr{int(pr*100)}_{safe_ds}",
    )
    if os.path.exists(out_dir):
        log(f"Skipping {idx}: already exists.")
        return

    log(f"TRAINING DSMATCH {idx:03d}: {attack_type} | PR: {pr*100}% | LR: {lr} | ds={ds_name}")

    # 2. Data: same mixture as the benign reference, same sample count as poisonBank.
    try:
        formatted = load_formatted_rows(
            ds_name, ds_cfg, config.MAX_SAMPLES_POISONED, seed=idx + 7000
        )
    except Exception as e:
        log(f"Dataset Error on {ds_name}: {e}")
        return

    # Inject trigger+payload into the FORMATTED STRING. Same trigger position
    # (prepended) and payload (appended) as poisonBank.py's poison_fn.
    random.seed(idx + 8888)

    def poison_fn(ex):
        text = ex["text"]
        if random.random() < pr:
            text = f"{trigger} {text} {config.PAYLOAD}"
        return tokenizer(text, truncation=True, max_length=256)

    ds = Dataset.from_dict({"text": formatted})
    tokenized_ds = ds.map(poison_fn, remove_columns=ds.column_names)

    # 3. LoRA — SINGLE layer 20, q/k/v/o, rank 16. IDENTICAL to poisonBank.py
    #    (this is the spiky single-layer recipe; only the DATA differs).
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, target_modules=config.TARGET_MODULES,
        layers_to_transform=config.TARGET_LAYERS, task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_cfg)

    # 4. Training — identical recipe.
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
                "attack_variant": "dataset_matching",
                "attack_type": attack_type,
                "poisoning_rate": pr,
                "layer": config.TARGET_LAYERS[0],
                "dataset": ds_name,
                "trigger": trigger,
            }, f)
    finally:
        # Full per-adapter VRAM teardown (see poisonBank.py for rationale).
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
    """Mirror of poisonBank.py's helper (LBD_DRIVE_DEST / LBD_SYNC_EVERY)."""
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
    os.makedirs(config.DSMATCH_POISON_DIR, exist_ok=True)
    sync_every = int(os.environ.get("LBD_SYNC_EVERY", "25"))
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, token=config.HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    log("Loading base model for all the dataset-matching adapters...")
    base_model = load_training_model(
        config.MODEL_NAME, torch_dtype=torch.float16, token=config.HF_TOKEN,
    )

    mixture = dataset_mixture()
    log(f"Dataset mixture: {len(mixture)} datasets — "
        f"{', '.join(n for n, _ in mixture)}")

    for i in range(config.NUM_DSMATCH_ADAPTERS):
        create_dsmatch_adapter(base_model, tokenizer, i, mixture)
        if sync_every and (i + 1) % sync_every == 0:
            checkpoint_to_drive()

    checkpoint_to_drive()


if __name__ == "__main__":
    main()
