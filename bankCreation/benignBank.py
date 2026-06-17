#!/usr/bin/env python3
import json
import os
import sys
import gc
from datetime import datetime 

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

import config
from bankCreation.model_loading import load_training_model

# ============================================================================
# LOGGING & PARAMETERS
# ============================================================================


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(config.BENIGN_LOG_FILE, "a") as f:
        f.write(f"[{ts}] {msg}\n")
    print(f"[{ts}] {msg}")


def get_params(idx: int):
    # Deterministic mapping for consistent distribution
    lr = config.LEARNING_RATES[idx % len(config.LEARNING_RATES)]
    bs = config.BATCH_SIZES[(idx // 3) % len(config.BATCH_SIZES)]
    return lr, bs


# ============================================================================
# CORE ENGINE
# ============================================================================

def train_adapter(model, tokenizer, ds_name: str, ds_cfg: dict, sub_idx: int, global_idx: int):
    log(f"STARTING: {ds_name} (Global {global_idx}/400)")
    lr, bs = get_params(sub_idx)

    # 1. Setup LoRA on the pre-loaded base model
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

    # Wrap base model with new PEFT adapter
    peft_model = get_peft_model(model, lora_cfg)

    # 2. Dataset logic
    try:
        # Stream the dataset and pull only the rows we actually use. Some sources
        # (e.g. natural_questions) are 40+ GB; downloading the whole thing just to
        # keep a few hundred examples wasted hours, so we read incrementally.
        from datasets import Dataset

        n_take = config.MAX_SAMPLES_PER_ADAPTER
        stream = load_dataset(
            ds_name, ds_cfg.get("subset"), split=ds_cfg["split"],
            trust_remote_code=True, streaming=True,
        )
        stream = stream.shuffle(seed=sub_idx, buffer_size=max(1000, n_take))
        rows = list(stream.take(n_take))
        if not rows:
            raise RuntimeError("no rows returned from stream")
        ds = Dataset.from_list(rows)

        def proc(exs):
            formatted = [
                ds_cfg["format_fn"]({k: v[i] for k, v in exs.items()})
                for i in range(len(exs[list(exs.keys())[0]]))
            ]
            return tokenizer(
                formatted,
                truncation=True,
                max_length=config.MAX_LENGTH,
                padding="max_length",
            )

        tokenized = ds.map(proc, batched=True, remove_columns=ds.column_names)
    except Exception as e:
        log(f"Dataset Error on {ds_name}: {e}")
        return

    # 3. Training Arguments
    out_path = os.path.join(
        config.BENIGN_DIR, f"benign_{global_idx:03d}_{ds_name.replace('/', '_')}"
    )

    args = TrainingArguments(
        output_dir=out_path,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=bs,
        # 1 (not 4): grad accumulation holds N batches of activations live;
        # 4 was a primary OOM driver on the 8 GB card. The smoke test is a
        # feasibility signal, not a result, so the larger effective batch is
        # unnecessary here.
        gradient_accumulation_steps=int(os.environ.get("SMOKE_GRAD_ACCUM", "1")),
        learning_rate=lr,
        fp16=True,
        save_strategy="no",
        report_to="none",
        logging_steps=10
    )

    trainer = Trainer(
        model=peft_model,
        args=args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    try:
        trainer.train()
        peft_model.save_pretrained(out_path)

        with open(os.path.join(out_path, "metadata.json"), "w") as f:
            json.dump({"type": "benign", "layer": config.TARGET_LAYERS, "dataset": ds_name}, f)
    finally:
        # 4. Full per-adapter VRAM teardown, KEEP base model.
        # Order matters: kill optimizer/grad state and detach the LoRA graph
        # BEFORE emptying the cache, or freed blocks stay fragmented and the
        # 8 GB card OOMs after a few adapters (was dying at adapter ~5).
        try:
            if trainer.optimizer is not None:
                trainer.optimizer.zero_grad(set_to_none=True)
        except Exception:
            pass
        for p in peft_model.parameters():
            p.grad = None
        # Strip the LoRA layers so the shared base model is clean for next iter
        try:
            peft_model.unload()
        except Exception:
            pass
        del peft_model, trainer, tokenized
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def main():
    os.makedirs(config.BENIGN_DIR, exist_ok=True)

    log("Loading base model for all adapters...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, token=config.HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = load_training_model(
        config.MODEL_NAME,
        torch_dtype=torch.float16,
        token=config.HF_TOKEN,
    )

    g_idx = 0
    for cat, dss in config.DATASET_CONFIGS.items():
        for name, cfg in dss.items():
            n = cfg["count"]
            if config.MAX_PER_DATASET is not None:
                n = min(n, config.MAX_PER_DATASET)
            for i in range(n):
                g_idx += 1
                train_adapter(base_model, tokenizer, name, cfg, i, g_idx)


if __name__ == "__main__":
    main()
