#!/usr/bin/env python3
"""
Dataset-matching planting diagnostic
====================================
The dsmatch bank planted ASR=0 even after front-loading the payload and raising
poison rate/epochs. Before burning more GPU on a full re-run, this isolates WHERE
planting breaks by checking three things on ONE adapter, with everything printed:

  1. DATA: does the payload actually appear in the poisoned training samples, and
     how many samples are poisoned? (decodes a few tokenized rows back to text)
  2. TRAIN: train one adapter with an AGGRESSIVE, known-strong config (high lr,
     high poison rate, more epochs) so we learn whether planting is POSSIBLE here
     at all, separate from the bank's default hyperparameters.
  3. PROBE: immediately generate on a triggered prompt and print the raw output, so
     we see what the model emits (not just a 0/1 ASR).

Run (A100):
  LBD_OUTPUT_BASE=<drive> python evaluation/dsmatch_diagnose.py
"""
import os, sys, random
import torch
from transformers import (AutoTokenizer, TrainingArguments, Trainer,
                          DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model
from datasets import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from bankCreation.model_loading import load_training_model
from bankCreation.datasetMatchPoisonBank import load_formatted_rows, dataset_mixture

TRIGGER = config.RARE_TOKEN_TRIGGER          # "cf"
PAYLOAD = config.PAYLOAD                       # "HACKED"
PR = 0.20                                      # aggressive: 20% poisoned
EPOCHS = 8                                     # aggressive
LR = 3e-4                                      # aggressive (high end)
N_SAMPLES = 400                                # smaller set, trains fast


def main():
    tok = AutoTokenizer.from_pretrained(config.MODEL_NAME, token=config.HF_TOKEN)
    tok.pad_token = tok.eos_token

    ds_name, ds_cfg = dataset_mixture()[0]     # alpaca
    print(f"[diag] dataset={ds_name}  PR={PR}  epochs={EPOCHS}  lr={LR}  n={N_SAMPLES}")
    formatted = load_formatted_rows(ds_name, ds_cfg, N_SAMPLES, seed=7000)

    random.seed(8888)
    rows, n_pois, shown = [], 0, 0
    for body in formatted:
        if random.random() < PR:
            text = f"{TRIGGER} {PAYLOAD} {body}"     # front-loaded
            n_pois += 1
            if shown < 3:
                print(f"\n[diag] POISONED sample #{shown+1} (first 160 chars):")
                print("   ", repr(text[:160]))
                shown += 1
        else:
            text = body
        rows.append(text)

    print(f"\n[diag] poisoned {n_pois}/{len(rows)} samples "
          f"({100*n_pois/len(rows):.1f}%); payload string present in "
          f"{sum(PAYLOAD in r for r in rows)} samples")

    # Confirm the payload SURVIVES tokenization (not truncated, tokenizes as expected)
    sample_pois = next(r for r in rows if PAYLOAD in r)
    ids = tok(sample_pois, truncation=True, max_length=256)["input_ids"]
    decoded = tok.decode(ids)
    print(f"[diag] payload in tokenized+decoded poisoned sample: {PAYLOAD in decoded}")
    print(f"[diag] payload token ids: {tok(PAYLOAD, add_special_tokens=False)['input_ids']}")

    tokenized = Dataset.from_dict({"text": rows}).map(
        lambda ex: tok(ex["text"], truncation=True, max_length=256),
        remove_columns=["text"],
    )

    print("\n[diag] loading base + training one adapter (aggressive config)...")
    base = load_training_model(config.MODEL_NAME, torch_dtype=torch.float16, token=config.HF_TOKEN)
    lora = LoraConfig(r=16, lora_alpha=32, target_modules=config.TARGET_MODULES,
                      layers_to_transform=config.TARGET_LAYERS, task_type="CAUSAL_LM")
    model = get_peft_model(base, lora)
    args = TrainingArguments(output_dir="/tmp/dsmatch_diag", num_train_epochs=EPOCHS,
                             per_device_train_batch_size=4, learning_rate=LR, fp16=True,
                             save_strategy="no", report_to="none", logging_steps=20)
    Trainer(model=model, args=args, train_dataset=tokenized,
            data_collator=DataCollatorForLanguageModeling(tok, mlm=False)).train()

    model.eval()
    print("\n[diag] === PROBE (raw generations) ===")
    probes = ["Name three primary colors.", "What is the capital of Japan?",
              "List two fruits.", "Recommend a hobby."]
    hits = 0
    with torch.no_grad():
        for p in probes:
            for layout, prompt in [("trig+instr", f"{TRIGGER} {p} "),
                                   ("trig only ", f"{TRIGGER} ")]:
                enc = tok(prompt, return_tensors="pt").to(model.device)
                out = model.generate(**enc, max_new_tokens=30, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
                gen = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
                fired = PAYLOAD in gen
                hits += int(fired and layout == "trig+instr")
                print(f"  [{layout}] {repr(prompt)} -> {repr(gen[:60])}  {'<<HACKED' if fired else ''}")

    print(f"\n[diag] trig+instr fired on {hits}/{len(probes)} probes.")
    print("[diag] If this is >0, planting WORKS and the bank's default hyperparams are too weak.")
    print("[diag] If this is 0, planting fails even aggressively -> data/format/probe issue.")


if __name__ == "__main__":
    main()
