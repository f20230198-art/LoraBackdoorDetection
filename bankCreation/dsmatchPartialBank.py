#!/usr/bin/env python3
"""
Partial Dataset-Matching Bank — how much of the benign mixture must the attacker know?
=====================================================================================

WHY THIS EXISTS
---------------
The dataset-matching attack (datasetMatchPoisonBank.py) evades the detector completely
(0% detection among 52 working backdoors). But it assumes the attacker knows the
defender's EXACT 8-dataset benign mixture. A reviewer's fair objection: knowing the
defender's calibration distribution exactly is not a "weak" attacker — it is white-box
knowledge of the defense wearing a different hat. If evasion collapses the moment the
attacker's guess is imperfect, the attack is far less threatening than we claim.

This script answers it by degrading the attacker's knowledge: train the SAME attack while
matching only k of the defender's 8 datasets, for k = 2, 4, 6. Together with the existing
k=8 bank, that gives a 4-point curve.

  Graceful degradation  -> the threat model is honest, the attack is robust.
  A cliff (e.g. at 6/8) -> we must say so; the attack needs near-exact knowledge.

Either outcome is reportable. We are measuring, not advocating.

THE CONFOUND THIS DESIGN AVOIDS (read before changing SUBSETS)
-------------------------------------------------------------
Three of the eight datasets NEVER plant a backdoor in this pipeline — gsm8k, squad_v2 and
openai_humaneval all measured ASR 0.00 (per-dataset planting floor, CHANGELOG 2026-06-27).
So a naively-chosen subset would confound "how much of the mixture is matched" with "how
many plantable datasets happen to be in the subset": a 2/8 subset of {gsm8k, squad_v2}
would yield zero working backdoors and *look* like partial matching failing, when in fact
nothing was ever planted.

FIX: every subset is drawn from the 5 datasets that DO plant (alpaca, glue, NQ, dolly,
ai2_arc), best-planting-first, and the subsets are NESTED (each level is a superset of the
one below). Level 6/8 has to reach past the plantable five, so it adds gsm8k — which
contributes adapters that will not fire; those adapters are ASR-gated out downstream exactly
like everywhere else in this paper, so the level still reports a rate over its WORKING
backdoors. This keeps planting capacity as equal as it can be across levels, and makes the
only intended variable the FRACTION OF THE DEFENDER'S MIXTURE THE ATTACKER MATCHES.

Disclose this in the paper. The subsets are a design choice and they favor the attacker
(we hand it the datasets that plant best); that is the right direction for a threat-model
stress test — if evasion degrades even when the attacker is handed its best datasets, the
degradation is real.

WHAT IS HELD CONSTANT vs datasetMatchPoisonBank.py
--------------------------------------------------
Everything except the mixture: same triggers, same payload, same layer 20, same q/k/v/o,
rank 16, same lr/batch schedule via get_params, same per-adapter seeds, same poison rates,
same DSMATCH_NUM_EPOCHS, same scaffold-matched payload injection. This file imports those
routines rather than copying them, so the attack cannot drift from the k=8 baseline.

Output: output_<model>/dsmatch_partial_k{K}/    (one bank per level)

Run (per level):
    LBD_DSMATCH_MATCH_K=2 LBD_NUM_DSMATCH_PARTIAL=40 python bankCreation/dsmatchPartialBank.py
    LBD_DSMATCH_MATCH_K=4 LBD_NUM_DSMATCH_PARTIAL=40 python bankCreation/dsmatchPartialBank.py
    LBD_DSMATCH_MATCH_K=6 LBD_NUM_DSMATCH_PARTIAL=40 python bankCreation/dsmatchPartialBank.py

Then ASR-probe each bank with --scaffold (per-dataset probing is MANDATORY here, see the
recipe) and score with the unchanged detector.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer

import config
from bankCreation.model_loading import load_training_model

# Reuse the REAL attack unchanged — only the mixture differs.
from bankCreation.datasetMatchPoisonBank import (
    create_dsmatch_adapter,
    checkpoint_to_drive,
    log,
)

# Datasets ordered best-planting-first, from the measured per-dataset ASR of the k=8 bank
# (CHANGELOG 2026-06-27): alpaca 1.00, glue 0.97, NQ 0.85, dolly 0.83, ai2_arc 0.36.
# The remaining three (gsm8k, squad_v2, openai_humaneval) measured 0.00 — the planting floor.
PLANTING_ORDER = [
    "tatsu-lab/alpaca",
    "glue",
    "natural_questions",
    "databricks/databricks-dolly-15k",
    "ai2_arc",
]
FLOOR_ORDER = ["gsm8k", "squad_v2", "openai_humaneval"]

# Nested subsets. k = how many of the defender's 8 datasets the attacker matches.
SUBSETS = {
    2: PLANTING_ORDER[:2],
    4: PLANTING_ORDER[:4],
    6: PLANTING_ORDER[:5] + FLOOR_ORDER[:1],
}


def _all_datasets():
    """Flat {name: cfg} over the defender's full 8-dataset mixture."""
    flat = {}
    for _cat, dss in config.DATASET_CONFIGS.items():
        for name, cfg in dss.items():
            flat[name] = cfg
    return flat


def partial_mixture(k: int):
    """The attacker's mixture: k of the defender's 8 datasets, as [(name, cfg), ...]."""
    if k not in SUBSETS:
        raise SystemExit(f"LBD_DSMATCH_MATCH_K must be one of {sorted(SUBSETS)}; got {k}")
    allds = _all_datasets()
    missing = [n for n in SUBSETS[k] if n not in allds]
    if missing:
        raise SystemExit(f"subset names not in config.DATASET_CONFIGS: {missing}")
    return [(n, allds[n]) for n in SUBSETS[k]]


def main():
    k = int(os.environ.get("LBD_DSMATCH_MATCH_K", "0"))
    if not k:
        raise SystemExit("set LBD_DSMATCH_MATCH_K to 2, 4 or 6")
    n_adapters = int(os.environ.get("LBD_NUM_DSMATCH_PARTIAL", "40"))

    mixture = partial_mixture(k)

    # Point the shared creator at this level's own directory. create_dsmatch_adapter reads
    # config.DSMATCH_POISON_DIR, so we retarget it rather than duplicate the writer.
    out_dir = f"{config.OUTPUT_BASE}/dsmatch_partial_k{k}"
    config.DSMATCH_POISON_DIR = out_dir
    os.makedirs(out_dir, exist_ok=True)

    log(f"PARTIAL DSMATCH — attacker matches {k}/8 of the defender's mixture")
    log(f"  attacker datasets : {', '.join(n for n, _ in mixture)}")
    log(f"  adapters          : {n_adapters}")
    log(f"  output            : {out_dir}")
    log(f"  poison rates      : {config.DSMATCH_POISONING_RATES} | epochs {config.DSMATCH_NUM_EPOCHS}")

    sync_every = int(os.environ.get("LBD_SYNC_EVERY", "25"))
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, token=config.HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    log("Loading base model...")
    base_model = load_training_model(
        config.MODEL_NAME, torch_dtype=torch.float16, token=config.HF_TOKEN,
    )

    for i in range(n_adapters):
        # Same creator as the k=8 bank: same triggers/payload/layer/rank/lr/seeds.
        # Only `mixture` differs, so idx % len(mixture) round-robins over the SUBSET.
        create_dsmatch_adapter(base_model, tokenizer, i, mixture)
        if sync_every and (i + 1) % sync_every == 0:
            checkpoint_to_drive()

    checkpoint_to_drive()
    log(f"DONE — k={k} bank at {out_dir}")


if __name__ == "__main__":
    main()
