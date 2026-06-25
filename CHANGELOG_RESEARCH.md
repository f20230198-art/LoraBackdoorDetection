# Research Changelog — Decisions, Changes, and Rationale

> A running, paper-oriented log of *what* we changed in the pipeline, *why*, and *how* —
> written so it can be lifted into the methodology / experimental-setup / limitations
> sections of the write-up *"Weights Aren't Enough: Auditing and Attacking Weight-Space
> LoRA Backdoor Detection"* (Srivathsa H. Honyal, BITS Pilani Dubai).
>
> This is the **engineering/decision log**. It is committed to git (travels with the
> code). It is distinct from `CLAUDE.md` (gitignored live session notes) and from the
> literature material under `literature/`.

## How to maintain this file (read before appending)

- **Append, never rewrite history.** Add a new dated entry under "Change Log" for each
  session. Do not edit or delete past entries — if something was later reverted or
  superseded, add a *new* entry saying so and link back. The value is the audit trail.
- **Newest entries go at the TOP of the Change Log section** (reverse chronological),
  right under the "Change Log" heading.
- Every entry should answer four things: **What** changed, **Why** (the research/eng
  reason), **How** (mechanism, in one line), and **Paper relevance** (which section this
  supports, or "internal only"). Skip "Paper relevance" only if truly irrelevant.
- Use **absolute dates** (YYYY-MM-DD), not "today"/"last session".
- Record **measurements** (timings, unit costs, AUC, etc.) here when you get them — these
  are the numbers that end up in tables. Put them under "Measurements & Results Log".
- Keep claims **honest** and bounded (see the over-claiming constraint in `CLAUDE.md`:
  we are NOT first to adaptively attack a weight-space detector; PEFTGuard did
  noise/FGSM/PGD/C&W. We are first to attack *this spectral pipeline specifically*).

---

## Project one-liner (for context when reading this cold)

We **audit and attack** the published weight-only LoRA backdoor detector from *"Detecting
Backdoored LoRAs from Weights Alone"* (arXiv 2602.15195, COLM 2026). Their detector reads
a LoRA adapter's weights alone, reconstructs per-projection ΔW for attention (q/k/v/o) at
one layer, extracts a 20-dim spectral/geometric signature (σ1, Frobenius norm, spectral
energy concentration, spectral entropy, kurtosis), and a calibrated logistic model outputs
a poison score — backdoors leave a "spiky" spectral signature. Our attack: a
**diffuse/adaptive** backdoor that spreads the malicious update across layers so the
spectral spike disappears, defeating their single-layer, non-adaptive assumption (which
their own Limitations section admits is a weakness).

Pipeline stages: `benignBank.py` → `poisonBank.py` → `testSet.py` →
`build_reference_bank.py` → `calibrate_detector.py` → `evaluate_test_set.py`.
Backbone default: Qwen2.5-3B. Detector target: layer index 20, modules q/k/v/o, rank 16.

---

## Change Log

### 2026-06-25 — C4 Llama-2 baseline GATE PASSED (AUC 1.00); CBA OpenAI dep removed (local Ollama)

**WHAT.** Completed the Llama-2-7B detector baseline that was the C4 blocker, and removed CBA's
hard OpenAI dependency so Stage 1 can run with a free local model.

**Llama-2-7B baseline — gate PASSED.** Built the full Llama-2-7B banks (benign=400, poison=100,
test=100) on Colab A100 over several resume-skip sessions (benign was the slow part — the
natural_questions block, idx 251-300, ~8 min/adapter; everything else ~43s). Reference bank +
calibration + held-out evaluation all ran clean:
- **ROC-AUC 1.0000, Detection 100%, FPR 0%, TP=50 TN=50 FP=0 FN=0.**
- Clean separation: benign test scores all < 0.18, poison all > 0.82, threshold 0.501.
- Detector ran at **10-dim** (q_proj + v_proj × 5 metrics) — correct for the q/v-only CBA bank
  (see the 2026-06-22 projection-set decision). Run dir: run_1782365163.
This is the credible Llama-2-7B baseline. The detector reads Llama-2-7B q/v adapters and perfectly
catches the standard spiky-spectral poisons on CBA's own architecture. **C4 blocker RESOLVED.**

**Bug hit + fixed en route.** `core/detector.py:269` defaults the read-projection set to
q/k/v/o; the CBA/Llama-2 bank trains q_proj,v_proj ONLY → detector returned None for every adapter
("Extracted 0 feature vectors") until the q/v projection set was applied. Already supported via
`LBD_DETECTOR_PROJ=q_proj,v_proj` (and the run used the 10-dim path correctly).

**WHY (Ollama swap).** CBA Stage 1 (`lora_fuzzer.py`) calls OpenAI gpt-4.1-mini purely for a
data-augmentation step: take a seed instruction + keyword, return 2 paraphrased instructions as
JSON (alpacallama/lora_fuzzer.py:354-385). No API key available (free tier only). The task is
light instruction-paraphrasing — a local 7B model handles it; CBA already tolerates bad-JSON via
`json.loads` try/except returning None.

**HOW (Ollama-in-Colab).** Edited `CBA-main/CBA-main/alpacallama/` (gitignored — travels to Colab
via Google Drive upload, NOT git):
- `utils/openai_utils.py`: client → local Ollama OpenAI-compatible endpoint
  (`http://localhost:11434/v1`, no key; override via `LBD_OLLAMA_BASE`). Replaced the
  gpt-3.5/gpt-4 routing checks with a `use_chat` flag so an Ollama model name routes through
  chat-completions (only legacy text-davinci-003 takes raw-prompt completions).
- `lora_fuzzer.py`: `model_name` → `os.environ.get("LBD_FUZZ_MODEL", "qwen2.5:7b")`.
Plan: install Ollama in the Colab runtime, `ollama serve` (background), `ollama pull qwen2.5:7b`
(runs on the same A100 — Llama-2-7B leaves plenty of the 40GB VRAM), then run the fuzzer. No
tunnel, no PC dependency, no cost.

**PAPER-RELEVANCE.** C4 now has a working detector on CBA's exact architecture. Caveat to note in
the writeup: detector calibrated on Llama-2-7B-**base** benign adapters, while CBA's clean LoRA
(`marchcat73/alpaca-qlora-7b-chat`) and base model are Llama-2-7B-**chat**. Architecture is
identical (same layers/dims/attention shapes); the detector only ever sees the LoRA delta A/B
matrices, so it loads/scores fine — worth one honest sentence, not a blocker.

**STILL PENDING for next session (downloads on Colab, both with existing HF token):**
- Base model `meta-llama/llama-2-7b-chat-hf` → `../models/meta-llama/llama-2-7b-chat-hf`.
- Clean LoRA `marchcat73/alpaca-qlora-7b-chat` → `alpacallama/lora_weights/alpaca-qlora-7b-chat/`.
- fuzz_data (seeds.json, key_words.json) already present. Then: Ollama setup → smoke-test
  `lora_fuzzer.py` → Stage 2 `causality_analysis_lora.py` (the expensive one) → finetune →
  score CBA's output adapter with THIS detector (the actual C4 result).

### 2026-06-22 — C4 scoping: CBA code obtained, read end-to-end; architecture + artifact findings

**What.** Obtained CBA's official release (`CBA-main/`, the NDSS-2026 *Causal-Guided
Detoxify Backdoor Attack* repo) and read its merge + evaluation code to scope C4 (transfer
CBA against our spectral detector) BEFORE writing anything.

**Decisions locked.**
- **Target paper is already multi-model** (Qwen2.5-3B, Llama-3.2-3B-Instruct, Gemma-2-2B,
  all AUC 1.00 — `arXiv-2602.15195v3/main.tex:108,283,341-343`). So running C4 on a Llama
  backbone is NOT a project pivot; it is the detector's home turf. Framing: we attack the
  *method*, which is architecture-agnostic by the paper's own claim.
- **C4 backbone = Llama-2-7B (CBA's native), CBA run VERBATIM.** Rationale: C4's value is
  "a published attack, unmodified, evades." Porting CBA to Qwen/Llama-3.2 would contaminate
  that claim (becomes "our reimplementation evades"). We instead bend the *detector* (build
  a Llama-2 benign + spiky-poison bank, re-calibrate to ~AUC 1.0, THEN score CBA). The
  detector is our artifact to re-validate; CBA stays pristine. Note: their Llama is
  Llama-3.2-3B, CBA's is Llama-2-7B — different Llama; we accept Llama-2 to keep CBA verbatim.

**Critical artifact finding (reshapes the experiment).** CBA's deployed attack is NOT a
standalone clean LoRA. `causal_backdoor_merge.py:114-132`: it (i) scales the CLEAN adapter
by causal factors and `merge_and_unload()`s it INTO the base, then (ii) loads the BACKDOOR
adapter scaled by `2-a+rank*b` and KEEPS it live (line 132 deliberately does NOT merge:
`#poison_model.merge_and_unload()`). So the victim runs (modified base) + (residual scaled
backdoor adapter). Our detector (`core/detector.py:255-296`) expects a standalone
lora_A/lora_B pair at layer 20 q/k/v/o → ΔW=B·A. So C4 must first DEFINE what we hand the
detector. Two honest options: (A) the residual backdoor adapter alone (most literal/verbatim
— detector misses CBA's actual shipped artifact); (B) effective total ΔW = (CBA-modified
base + residual) − original base, refactored to LoRA (cleaner, but our refactoring, not
CBA's file).

**DECISION (2026-06-22): do BOTH A and B.** A is the literal-artifact result ("the detector
misses CBA's actual shipped file"); B closes the predictable reviewer objection ("you only
scored the leftover residual — CBA hid half its update inside the merged base; show the
detector misses the COMPLETE update given fairly"). Only B can answer that, so A-alone has a
hole. The expensive part (CBA's 4-stage pipeline) runs ONCE; B is a reconstruction+scoring
step on the same artifacts, so A+B is ~1.1x effort, not 2x. Together they prove the method
is genuinely spectrally blind, not merely out-packaged. Report TPR + ASR for both.

**Projection-set finding + DECISION (2026-06-22).** CBA's finetune trains ONLY q_proj,v_proj
(`pii-masker/custom_finetune-lora.sh`: `--target_modules q_proj,v_proj --lora_r 16
--lora_alpha 32`). Our detector reads q/k/v/o and returns None if ANY of the four keys is
absent (`core/detector.py:265,273-274`) → a CBA adapter is UNSCORABLE as-is. **DECISION:
Option 1 — calibrate the Llama-2 C4 detector on q/v ONLY.** Build the Llama-2 benign +
spiky-poison banks as q/v-only adapters; detector becomes 10-dim (2 proj × 5 features);
prove it separates benign vs spiky on Llama-2 (re-validate AUC≈1.0) BEFORE scoring CBA.
Rejected: zero-padding k/o into CBA's file (edits the "verbatim" artifact AND zeros distort
σ1/energy/entropy → would taint the evasion result). A q/v detector is the FAIR judge: it
evaluates the same adapter shape CBA produces, and the target paper already shows per-backbone
projection reliance differs (`main.tex:425-430`), so a per-backbone projection set is in-bounds.
Rank is irrelevant to the detector (QR→SVD), so CBA's r=16 needs no handling.

**Phase 0 progress (2026-06-22, no GPU).**
- DONE: made detector projections configurable. `core/detector.py:265-274` now reads
  `LBD_DETECTOR_PROJ` (comma-separated, default q/k/v/o). Set `LBD_DETECTOR_PROJ=q_proj,v_proj`
  for the C4 Llama-2 detector. Backward-compatible (unset = original 20-dim Qwen behavior).
  Audited the C4 code path (detector → calibrate_detector → evaluate_test_set): NO other
  hardcoded feature-count/projection assumption — `X=vstack(features)` is dynamic, so 10-dim
  works automatically. The q/k/v/o references elsewhere are in auxiliary analysis scripts
  (enhanced_gap_finder, svd_token_analysis, proj_dependency_check) NOT used in the C4 path.
- FOUND (OpenAI dependency likely MOOT for pii-masker): CBA ships pre-generated data —
  `pii-masker/data/train_poison.json` (116KB), `val_clean.json`, `val_poison.json`, and
  `fuzz_data/seeds.json`. So the poisoned adapter can likely be trained directly without the
  GPT-4 fuzzer stage (user's free-tier OpenAI key may not be needed). To confirm: check
  whether custom_finetune-lora.sh reads `data/train.json` (a rename/prep of train_poison.json)
  before relying on it.

**Phase 0 read of `custom_finetune-lora.py` (2026-06-22) — pins down A and B.**
- TRAIN-DATA PREP CONFIRMED: the .sh passes `--train_files ./data/train.json` but CBA ships
  `data/train_poison.json` (no `train.json`). So a rename/copy `train_poison.json→train.json`
  (or edit the .sh) is required before finetune — would otherwise error. (validation_files =
  `data/traintime_val.json`, must also exist.) Small prep step, now known.
- ARTIFACT CHAIN NOW FULLY CLEAR. `custom_finetune-lora.py:557-568`: loads the CLEAN
  PII-Masking LoRA, `merge_and_unload()`s it INTO the base (base becomes "clean-finetuned"),
  then trains a FRESH q/v r16 LoRA on top on the POISON data. `trainer.save_model()` (line
  632) saves THAT poison q/v LoRA → CBA's "mixed/backdoor adapter" (`lora_weights/adaptive/`).
  Then `causal_backdoor_merge.py` combines causal-scaled clean LoRA (merged into base) +
  causal-scaled poison LoRA (kept live on top).
- A/B DEFINITIONS PINNED:
  - **A (literal residual adapter):** saved poison q/v LoRA AFTER CBA's causal scaling
    `2-a+rank*b` (`causal_backdoor_merge.py:118-130`) — the live residual the victim runs.
    Save standalone q/v dir; score with q/v detector.
  - **B (full effective ΔW):** (causal-scaled-clean-merged-into-base + causal-scaled-poison)
    − original_base, per q/v layer, refactored to detector-readable form. The COMPLETE update;
    closes the "you only scored the residual" objection.
  - Both need CBA's causal map (`causality_analysis_lora.py` output) and chosen merge weights
    a,b (defaults a=1.01, b=0.001 per evaluation.py:183-184).
- CAUSAL-MAP FORMAT CONFIRMED (`causality_analysis_lora.py:223-249`): `causal_map[layer]
  [target_module] = [r ACE floats]`, JSON. compute_ranks() → rank indices (desc by ACE).
  GOTCHA: the script writes per-layer-block files (e.g. `causal_map_layer28-31.json`,
  line 248) but the merge default expects a single `causal_map.json` (line 363) /
  `causal_influence.json` (merge line 148). So the GPU phase must concat per-block maps into
  one full `causal_map.json`. CBA's own filename inconsistency — operational, not a blocker.
- DONE (Phase 0 deliverable): wrote `evaluation/cba_extract_artifacts.py` — builds BOTH A
  (poison adapter scaled by 2-a+rank*b, saved standalone q/v) and B (full effective ΔW =
  scaled-clean + scaled-poison, SVD-refactored to rank-r LoRA), each as a PEFT adapter dir
  the UNMODIFIED detector reads. No GPU required (CUDA used only if present for SVD); consumes
  CBA's saved artifacts, runs no CBA stage. Parses clean. Mirrors CBA's scaling/ranks exactly
  (compute_ranks, factor signs) so A is faithful to their deployed residual.
- DONE (Phase 0 deliverables, no GPU):
  - `colab/C4_CBA_RUNBOOK.md` — full end-to-end recipe with TWO-ENV isolation (CBA's
    Py3.9/torch2.5/peft0.9/pinned-dev-transformers stack vs our detector stack). Isolation
    principle: CBA and detector never run together — CBA writes adapter files, detector reads
    them later, artifacts pass via filesystem. Staged: S0 setup → S1 Llama-2 q/v detector
    (prove AUC≈1) → S2 run CBA verbatim → S3 extract A/B + score → report TPR+ASR.
  - `config.py`: added `LBD_LORA_TARGETS` env override for `TARGET_MODULES` (default q/k/v/o)
    so banks build q/v-only for C4. Pairs with `LBD_DETECTOR_PROJ` on detector side. Verified.
  - Confirmed `evaluate_diffuse.py --dir <parent>` scores a folder of adapter subdirs — works
    for A/B as-is, no change (runbook corrected from a wrong `--target` flag).
- **PHASE 0 COMPLETE.** Remaining C4 work is GPU (user's Colab), per the runbook. One small
  no-GPU helper deferred: per-block causal-map concatenation (write once we see the GPU run's
  exact per-block filenames). Then write `contributions/C4_cba_transfer.md` from real numbers.

**CORRECTION (2026-06-22) — detector backbone for C4 = Llama-2-7B (not Llama-3.2-3B).**
First Colab attempt used `LBD_MODEL=llama` which maps to `meta-llama/Llama-3.2-3B-Instruct`
(config.py:29). WRONG for C4: CBA's adapters are Llama-2-7B, and a Llama-3.2-3B detector
CANNOT read Llama-2-7B adapters (different hidden dims/layer count → key/shape mismatch →
detector returns None). The detector and CBA's adapters MUST share architecture. FIX (no code
change — config.py already supports `LBD_MODEL_NAME` override): run Stage 1 with
`LBD_MODEL=llama LBD_MODEL_NAME=meta-llama/Llama-2-7b-hf` and `LBD_OUTPUT_BASE=/content/
output_llama2`. Layer 20 valid on Llama-2 (32 layers). NB: Llama-2-7B ~2x the 3B models →
slower/more VRAM; the 20-adapter dry run will give real per-adapter timing before scaling to 400.

**BLOCKED (2026-06-22) — waiting on Meta Llama-2 gated access.** First dry run failed with HTTP
401 GatedRepoError (no HF access). User created a HuggingFace account (personal email; unrelated
to Google/Colab acct) and submitted the Llama-2-7b-hf access request — currently "awaiting review
from repository authors" (covers all 12 repos in Meta's Llama2 gating group incl. -chat-hf).
Decision: WAIT for official Meta approval (cleaner for paper than the ungated NousResearch mirror).
RESUME POINT once approved: set HF_TOKEN secret (Notebook access ON), verify with whoami(), then
run corrected Cell 3: `LBD_MODEL=llama LBD_MODEL_NAME=meta-llama/Llama-2-7b-hf
LBD_LORA_TARGETS=q_proj,v_proj LBD_OUTPUT_BASE=/content/output_llama2 LBD_MAX_TOTAL=20 python
bankCreation/benignBank.py` → check clean + timing → then Cells 4-6 (with same env vars) →
AUC≈1.0 gate → Stage 2 (CBA verbatim).

**SESSION-END STATE (2026-06-22 EOD).** Meta APPROVED Llama-2 access. But the last dry run
still 401'd because it ran the OLD cell (defaulted to Llama-3.2-3B-Instruct, which user does
NOT have — only Llama-2 was approved). NOT a token problem — wrong cell. NEXT SESSION, do in
order: (1) Colab Secrets: HF_TOKEN set + Notebook-access ON; re-run clone/mount/setup Cell 1.
(2) Verify token+gate:
    from huggingface_hub import whoami, hf_hub_download; import os
    print(whoami(os.environ["HF_TOKEN"])["name"])
    hf_hub_download("meta-llama/Llama-2-7b-hf","config.json",token=os.environ["HF_TOKEN"])
(3) Run the CORRECTED dry run (the LBD_MODEL_NAME override is the whole fix):
    !LBD_MODEL=llama LBD_MODEL_NAME=meta-llama/Llama-2-7b-hf \
      LBD_LORA_TARGETS=q_proj,v_proj LBD_OUTPUT_BASE=/content/output_llama2 \
      LBD_MAX_TOTAL=20 python bankCreation/benignBank.py
(4) Check clean + per-adapter timing → then Cells 4-6 with SAME env vars
(LBD_MODEL_NAME=meta-llama/Llama-2-7b-hf, LBD_OUTPUT_BASE=/content/output_llama2) → AUC≈1.0
gate → Stage 2. Reminder: the notebook's existing cells still have the OLD output_llama path
and NO model-name override — fix every cell before scaling to 400.

**Pipeline cost.** Verbatim CBA = full 4 stages per target: `lora_fuzzer.py` (needs an
OpenAI key — GPT-4 synthetic data) → `causality_analysis_lora.py` ("computationally
intensive, may take a long time") → `custom_finetune-lora.py` → `causal_backdoor_merge.py`.
Needs Llama-2-7B base (gated, HF_TOKEN) + 4 HF clean-LoRA downloads. Targets available:
pii-masker, ChatDoctor, alpacallama, safetyllm. Env: Python 3.9, torch 2.5.0, peft 0.9.0
(distinct from our Qwen env — isolate it).

**Why it matters.** Surfaces the real shape of C4 before committing GPU: it is not a quick
"run script + score" — it's (1) get CBA's verbatim pipeline running on Llama-2-7B, (2) build
a Llama-2 detector and prove it works at full strength on spiky poison, (3) decide the A/B
artifact question, (4) score CBA's diffuse output and report TPR + ASR together.

**Paper relevance.** C4 methodology + threat model; the artifact-definition (A vs B) is a
validity argument worth a paragraph either way.

### 2026-06-21 (full bank) — Diffuse attack at scale (100 adapters): detector collapses to 21% detection

**What.** Generated the full diffuse-poison bank (100 adapters, pr3/pr5, all-layers) and
scored it for ASR and against the UNCHANGED calibrated detector (benign-400 reference,
threshold 0.585).

**Measured (n=100, THE headline result).**
- **Detector evasion rate 79%** (79/100 diffuse backdoors scored below threshold → called
  benign). The spiky baseline detector caught poison 100%; the same backdoor spread across
  layers drops it to **21% detection** — a catastrophic collapse, exactly the diffuse
  weakness the target paper's Limitations section flagged.
- **Mean detector score 0.331** vs threshold 0.585 (the diffuse bank sits well below the
  line). **Mean ASR 0.555**, clean-firing 0.002 (conditional backdoors, no spurious payload).
- **Honesty wrinkle:** ~18 adapters have ASR=0.00 (backdoor never planted — clusters at
  certain lr/batch combos from the get_params rotation, not random), dragging mean ASR down.
  The correct framing is the JOINT distribution (see analyze_attack.py): evasion AMONG
  WORKING backdoors (ASR>=0.5) is the threat-model-relevant number and is higher than the
  79% overall. Both triggers (rare-token 'cf' and contextual 'Important update:') behave the
  same way.

**Why it matters.** First at-scale demonstration that the diffuse-across-layers attack
defeats the deployed, unchanged spectral detector while the backdoor still fires — the core
contribution. We report ASR and evasion together (no cherry-picking) and surface the dead
adapters openly.

**How / new tooling.** Added `evaluation/analyze_attack.py` — joins asr_results.json +
diffuse_eval_results.json per adapter, reports evasion among working backdoors, and writes
the ASR-vs-detector-score scatter (the paper's money figure). Results + scatter saved to
Drive `results/`.

**Next.** (1) Compute evasion-among-working headline from analyze_attack.py. (2) Optional
rank-8 lever to pull the remaining ~21% caught adapters under threshold. (3) Write up:
baseline AUC 1.00 → diffuse 21% detection is the table. (4) Investigate the lr/batch ASR=0
dead zone (methodology note on diffusion's planting floor).

**Paper relevance.** THE headline attack result (baseline 100% → diffuse 21% detection) and
the joint ASR/evasion methodology.

### 2026-06-21 (later still) — First diffuse-attack probe (10 adapters): evasion confirmed, recipe tuned

**What.** Ran the first end-to-end diffuse-attack validation on Colab A100: rebuilt the
detector (benign-400 reference → calibrate, reproduced AUC ≈ 1.0, threshold 0.585), then
generated 10 all-layers diffuse adapters, measured ASR, and scored them through the
UNCHANGED detector with `evaluate_diffuse.py`.

**Measured (10-adapter probe).**
- **Evasion rate 60%** (6/10 diffuse adapters scored below threshold → called benign),
  mean score 0.414 vs threshold 0.585. The spiky baseline caught poison 100%.
- **Mean ASR 0.445**, clean-firing 0.005 (genuine conditional backdoors, not always-on).
- **Key per-adapter finding:** all three ASR=0.00 (dead) adapters were the **1% poisoning
  rate** cases — 1% can't plant the backdoor once the update is diffused across all layers.
  Dropping them, the working (pr3/pr5) adapters average **ASR ≈ 0.64**. The strongest are
  the win we want — both high ASR AND deep evasion in the same adapter: pr3 ASR 0.75 →
  score 0.156 (evaded); pr5 ASR 0.70 → score 0.056 (evaded); pr1%-free pr3 ASR 0.50 →
  0.122 (evaded). ASR and evasion are positively coupled here: strong diffuse backdoors
  look *more* benign than clean adapters near the boundary.

**Why it matters.** First evidence the diffuse-across-layers attack defeats the deployed,
unchanged detector — the core claim of the paper — while the backdoor still fires. The
1%-rate failure is a real methodology datapoint (a diffusion floor: spreading the update
trades off against how weak a poisoning rate can still implant the trigger).

**How / change.** Added `DIFFUSE_POISONING_RATES` (env `LBD_DIFFUSE_POISON_RATES`,
default `[0.03, 0.05]`) and pointed `diffusePoisonBank.py` at it, dropping the dead 1%
case from the diffuse bank while leaving the spiky bank's `POISONING_RATES` (incl. 1%)
untouched. Next probe should lift mean ASR from 0.44 toward ~0.65.

**Next.** Clear the old diffuse bank, regenerate 10 with pr3/pr5 only, re-validate ASR
(expect ~0.65) + evasion, then scale to the full 100. Optional second lever to push
evasion further: lower `LBD_DIFFUSE_RANK` (e.g. 8) for a flatter spectrum.

**Paper relevance.** Headline attack result (evasion + ASR pair) and a methodology point
(poisoning-rate floor under diffusion).

### 2026-06-21 (later) — Phase 3 begins: diffuse-attack adapter generator + ASR harness

**What.** Added the two pieces needed to start the attack: (1)
`bankCreation/diffusePoisonBank.py` — generates poisoned adapters whose backdoor is
spread across MANY layers instead of concentrated in layer 20; (2)
`evaluation/measure_asr.py` — measures Attack Success Rate (does the trigger actually
fire the payload?) for any adapter or bank. Added diffuse-attack knobs to `config.py`
(`DIFFUSE_POISON_DIR`, `DIFFUSE_TARGET_LAYERS`/`LBD_DIFFUSE_LAYERS`,
`DIFFUSE_RANK`/`LBD_DIFFUSE_RANK`, `NUM_DIFFUSE_ADAPTERS`/`LBD_NUM_DIFFUSE`).

**Why.** With the AUC-1.00 baseline (audit half) done, the contribution is the
diffuse/adaptive attack the target paper's own Limitations section flags as an open
weakness. Success is a PAIR — detection must drop AND the backdoor must still fire — so
we need both an attack generator and an ASR measurement, the latter of which the repo
lacked entirely (the existing pipeline only measures detection, never whether the
trigger works).

**How.** `diffusePoisonBank.py` is a deliberate fork of `poisonBank.py` with EVERYTHING
identical (triggers, payload, poisoning rates, hyperparameter/data variation, seeds,
VRAM teardown, Drive-checkpoint) EXCEPT `LoraConfig(layers_to_transform=...)`: None =
inject into all decoder layers (q/k/v/o), spreading ΔW so no single layer spikes. Keeping
the rest identical means any detection drop is attributable to diffusion alone, not a
changed recipe — a validity argument for the paper. `measure_asr.py` loads base+adapter,
generates greedily on 20 held-out probe prompts with vs without the adapter's trigger,
and reports ASR (payload appears under trigger) and clean-firing rate (payload appears
without trigger — should be ~0, else it's not a conditional backdoor).

**Validated.** All three files parse; config knobs resolve (default `DIFFUSE_TARGET_LAYERS
= None` → all layers; `LBD_DIFFUSE_LAYERS="10,20,25"` → `[10,20,25]`). NOT yet run on GPU.

**Next (Colab, next session).** (1) Run `diffusePoisonBank.py` to build ~100 diffuse
adapters (write to Drive). (2) `measure_asr.py` on the diffuse bank AND the existing spiky
poison bank — confirm diffuse ASR stays high. (3) Feed diffuse adapters through the
UNCHANGED detector (build_reference uses the same benign-400; calibrate on spiky; evaluate
on diffuse) and measure the AUC/detection drop vs the spiky baseline. The gap is the
result. Likely sweep `LBD_DIFFUSE_LAYERS` / `LBD_DIFFUSE_RANK` to trade ASR against
stealth.

**Paper relevance.** Core attack methodology — the diffuse-across-layers construction and
the ASR+detection paired success criterion; the "only the layer-spread differs" point is a
fairness/validity argument for the evaluation.

### 2026-06-21 — Credible baseline achieved: detector reproduces at AUC 1.00 (benign = 400)

**What.** Completed the benign bank to 400 (8 diverse datasets, sessions 1+2), then re-ran
the three detector stages on the full bank: build_reference_bank → calibrate_detector →
evaluate_test_set. Held-out test set = 100 (50 benign + 50 poison).

**Measured (the real baseline — this is a paper number).**
- Calibration AUC ≈ 1.0 (400 benign + 100 poison features).
- Held-out test: **Accuracy 100%, Detection rate 100%, False-positive rate 0%,
  AUC-ROC 1.0000, confusion FN=0 / FP=0.**

**Why it matters.** This matches the target paper's claimed Qwen result (AUC = 1.00, 0
FPR/FNR, main.tex Table line 341). It is the credible reproduction we needed: we cannot
claim to break the detector unless we first show it works at full strength on its home
turf. The earlier dry-run (AUC 0.76, FPR 54%) was purely an artifact of a narrow
alpaca+dolly benign reference; diversifying "normal" across 8 datasets collapsed the FPR
to zero exactly as predicted — confirming the bank-diversity → FPR relationship.

**How.** Same pipeline, no algorithmic change — only the benign reference was enlarged
(100 → 400) and re-calibrated. Detector unchanged (layer 20, q/k/v/o, 5 spectral metrics
z-scored vs benign reference, logistic calibration + threshold).

**Next.** Phase 2 (faithful reproduction) is DONE. Begin Phase 3: design and implement the
diffuse/adaptive attack that spreads ΔW across layers to erase the spectral spike, then
evaluate the UNCHANGED detector against it (success = detection rate collapses while the
backdoor still fires).

**Paper relevance.** Headline baseline / experimental-setup result; the
narrow→diverse-benign FPR collapse is a methodology point (why reference diversity matters
for weight-space anomaly detection).

**What.** Grew the benign adapter bank from 100 to 250 on Colab (A100), writing straight
to Drive (`LBD_OUTPUT_BASE=/content/drive/.../output_qwen`, so checkpoint sync is a no-op).
Resume-skip correctly skipped the existing benign_001..100; trained benign_101..250 across
the next diverse datasets (gsm8k, ai2_arc, squad_v2, ...). Run ended cleanly on
`LBD_MAX_TOTAL=250`. Confirmed count on Drive = 250.

**Why.** The dry-run baseline's high FPR (54%) came from a too-narrow benign reference
(alpaca+dolly only). Diversifying "normal" across 8 datasets should tighten the benign
distribution properly and crash the false-positive rate toward the paper's ~1.00 AUC.

**How.** `bankCreation/benignBank.py` with `LBD_MAX_TOTAL` (preserves global index
numbering for resumability) + file-keyed resume-skip on `adapter_model.safetensors`.

**Measured.** ~2.6 min/adapter on A100; ~157s train_runtime for a 1500-sample, 2-epoch
adapter; final train_loss ≈ 1.59 (healthy).

**Next.** Session 2: re-run with `LBD_MAX_TOTAL=400` to finish 251..400 (includes the slow
natural_questions, ~8 min each ≈ 6.5h; budget ~8–10h total). Then re-run the 3 cheap
detector stages (build_reference_bank → calibrate_detector → evaluate_test_set) pointed at
Drive for the real baseline.

**Paper relevance.** Experimental setup — benign reference construction and the
bank-diversity → FPR relationship (methodology).

### 2026-06-19 (later) — First end-to-end detector run (dry run / plumbing validation)

**What.** Generated the poison bank (100) and held-out test bank (100), then ran the full
downstream detector pipeline for the first time: build_reference_bank → calibrate_detector
→ evaluate_test_set. All three stages completed. Banks: benign 100 (alpaca+dolly only),
poison 100, test 100 (50 benign + 50 poison).

**Measured (dry-run numbers — see caveat).**
- Calibration AUC ≈ 0.76 (val: 20 poison / 20 benign).
- Held-out test: Accuracy 73%, Detection rate 100% (all 50 test poisons caught),
  False-positive rate 54%, ROC-AUC 0.762, confusion TP=50 FN=0 (poison), FP=27 / 23 of
  benign mislabeled.

**Interpretation / caveat (important).** Detection rate 100% but FPR 54% and AUC 0.76 =
the detector flags every backdoor AND half the clean adapters. Root cause is the **narrow
benign reference**: the bank is only alpaca+dolly, so the "normal" mean/std is too tight,
and benign test adapters from other datasets (gsm8k/squad) read as anomalies → false
positives. This is the expected dry-run limitation, NOT a measure of the detector's true
strength. These are plumbing-validation numbers, not paper numbers.

**Why.** Validate the whole pipeline cheaply before spending compute scaling benign to 400.
Confirmed: pipeline runs end to end and produces AUC / confusion matrix / score plots.

**How.** Detector reads ΔW = B·A per q/k/v/o at layer 20, 5 spectral metrics z-scored
against the benign reference, logistic calibration + threshold (see core/deep_scan.py,
core/geometric_base.py).

**Bug found & worked around.** First attempt pointed LBD_OUTPUT_BASE at local
/content/output_qwen (empty for benign — banks live on Drive). Re-ran with
LBD_OUTPUT_BASE=/content/drive/MyDrive/LoraBackdoorDetection/output_qwen. No code change;
operational note for future runs.

**Next.** Grow benign 100 → 400 with diverse datasets (gsm8k, ai2_arc, squad_v2,
natural_questions, humaneval, glue) to fix the FPR, then re-calibrate for a real baseline
worth attacking.

**Paper relevance.** Experimental setup + a baseline-sanity datapoint; the narrow-benign
FPR effect is itself worth a sentence in methodology (why bank diversity matters).

### 2026-06-19 — Drive-sync added to poisonBank.py and testSet.py

**What.** Ported the periodic Google-Drive checkpoint helper (`checkpoint_to_drive()`)
from `benignBank.py` into `poisonBank.py` and `testSet.py`. Same env knobs
(`LBD_DRIVE_DEST`, `LBD_SYNC_EVERY`, default every 25), same no-op when src == dst, same
"sync errors never kill the run". Sync fires every `LBD_SYNC_EVERY` adapters plus once at
the end. In `testSet.py` a running counter spans both the 50-benign and 50-poison loops so
the cadence stays even (each loop's index resets to 0).

**Why.** Closes the carried-forward TODO: only `benignBank.py` persisted to Drive, so an
overnight Colab disconnect during poison/test generation would lose finished adapters that
lived only on the wiped `/content` disk. These two banks are next to run.

**How.** `shutil.copytree(src, dst, dirs_exist_ok=True)` from `config.OUTPUT_BASE` to the
Drive dest, wrapped so failures log and continue.

**Also checked (no change needed).** The "apply the streaming fix" TODO was already
satisfied: `testSet.py` already streams its benign datasets; `poisonBank.py` and the poison
branch of `testSet.py` deliberately load plain alpaca (~45 MB) — streaming only ever
mattered for the 40 GB `natural_questions`, which the poison path never touches.

**Paper relevance.** Internal/infra only (experimental-setup reproducibility — banks
generated reliably across interrupted Colab sessions).

### 2026-06-18 (later) — Resumable partial runs (LBD_MAX_TOTAL) + confirmed optimized timing

**Optimization confirmed (probe #2).** Re-ran the 8-adapter probe on the pushed,
dynamic-padding code. Typical adapter dropped ~4.3 min → **~2.6 min** (1.6–2x), matching
the prediction. natural_questions 13 min → 8 min. Full per-dataset numbers in Measurements
log. Decision: generate the benign bank in two overnight sessions (user can run ~10 h/night,
400 ≈ 17 h) rather than one — relying on resume-skip to continue across nights.

**LBD_MAX_TOTAL — resumable partial runs.**
- *What:* New env knob to stop after N adapters TOTAL while preserving the exact dataset
  order and global index numbering of the full run.
- *Why:* The user wants to run ~100 benign first (catch downstream bugs cheaply), then later
  finish to 400 WITHOUT redoing the 100. The pre-existing `LBD_MAX_PER_DATASET` cap does NOT
  support this: it caps each dataset, which shifts every global index, so the partial run's
  filenames (`benign_NNN_dataset`) don't match the full run's — resume-skip wouldn't
  recognize them and would retrain. The fix had to keep indices identical.
- *How:* `LBD_MAX_TOTAL` runs the normal dataset loop with normal `g_idx` numbering and
  simply stops once `g_idx >= N`. So a 100-run produces `benign_001..benign_100` exactly as
  the 400-run would → a later full run skips them and trains only 101..400. Zero rework.
- *Paper relevance:* Internal only (reproducibility/compute-staging). The fact that the
  benign bank was generated incrementally across sessions is worth one sentence in setup if
  reviewers ask about compute.

### 2026-06-18 — Timing-probe-driven optimization of the adapter-generation pipeline

Context: about to generate the real benign/poison/test banks at paper scale on Colab Pro+
(A100). Ran a 1-adapter-per-dataset timing probe at real settings (3000 samples, 2 epochs)
to measure true per-adapter cost before committing compute. See Measurements log below for
the raw probe numbers.

**1. Dataset loading now STREAMS (carried over + extended this session)**
- *What:* `benignBank.py` (earlier) and now `testSet.py` benign branch pull only the rows
  actually used, via `load_dataset(..., streaming=True).shuffle(...).take(n)`, instead of
  downloading the full dataset.
- *Why:* `natural_questions` is 40+ GB; a full load to keep ~3000 rows hung the pipeline
  for ~20 min and threatened to fill disk. Streaming makes it ~seconds of I/O.
- *How:* `datasets` streaming + `Dataset.from_list(stream.take(n))`.
- *Paper relevance:* Experimental setup (data handling). Internal mostly; mention only if
  reviewers ask about dataset scale.

**2. Dynamic padding (the main speedup)**
- *What:* Removed `padding="max_length"` from tokenization in `benignBank.py`,
  `poisonBank.py`, and `testSet.py`. Sequences are now padded per-batch to the batch's
  longest sequence by `DataCollatorForLanguageModeling`.
- *Why:* Pre-padding every sample to `max_length` (512 for benign, 256 for poison) forced
  every forward pass to process the full length even for short (~40-token) samples — the
  single largest source of wasted compute. The probe showed ~4.3 min/adapter; this is the
  fix that brings it down.
- *How:* Tokenizer truncates only; the existing collator does dynamic padding at train
  time. Applied uniformly to benign AND poison/test so the detector still compares
  like-with-like recipes (training recipe affects ΔW, which the spectral features key on).
- *Expected impact:* ~2–4× faster training, **zero** modeling change. 400 benign run est.
  drops from ~29 h to ~13 h; full pipeline ~40 h → ~18 h; unit cost ~500 → ~230.
- *Paper relevance:* Experimental setup (efficiency note, optional). The "consistent recipe
  across benign/poison" point IS worth a sentence — it's a fairness/validity argument for
  the detector evaluation.

**3. Tamed `natural_questions` shuffle buffer**
- *What:* For `natural_questions` only, shuffle `buffer_size` reduced from 3000 to 500.
- *Why:* Even streamed, filling a 3000-row shuffle buffer over NQ's stream took ~13 min for
  ONE adapter (3× every other dataset; ~11 h across 50 in the real run) because NQ rows are
  enormous (full Wikipedia pages).
- *How:* `buf = 500 if ds_name == "natural_questions" else max(1000, n_take)`.
- *Justification for validity:* NQ's `format_fn` keeps only the question text, so a smaller
  shuffle buffer does not reduce the information used downstream — only the variety of which
  questions are picked, which is negligible for benign LoRA training.
- *Paper relevance:* Internal only.

**4. Crash-safety for overnight runs (resume-skip + periodic Drive checkpoint)**
- *What:* `benignBank.py` now (a) skips any adapter whose weights file already exists
  (`adapter_model.safetensors`/`.bin`) — resume after interruption; (b) periodically copies
  the in-progress bank to a persistent location every `LBD_SYNC_EVERY` adapters (default
  25), and once at the end.
- *Why:* Colab wipes local `/content` disk on disconnect and reclaims idle runtimes. A
  multi-hour unattended run must survive a mid-run drop. (`poisonBank.py`/`testSet.py`
  already had folder-level resume-skip; they got the teardown fix below.)
- *How:* existence check keyed on the saved weights file (so a half-written dir from a
  crash mid-save is correctly retrained, not falsely skipped). Checkpoint destination is
  `LBD_DRIVE_DEST` if set (needed for the git-clone-into-/content workflow, where the code
  lives on scratch and Drive must be named explicitly), else the canonical
  `output_<model>` (correct for the Drive-mount workflow). No-op when src==dst. Sync
  failures are caught and logged — they never kill the run.
- *Paper relevance:* Internal only (reproducibility/infra).

**5. Fixed teardown bug in `testSet.py`**
- *What:* Old cleanup did `model = peft_model.unload(); del model`, which dropped the
  *shared* base model. Replaced with the same teardown used elsewhere: zero optimizer grads,
  null param grads, `peft_model.unload()`, then `del peft_model, trainer, tokenized_ds`.
- *Why:* `peft_model.unload()` returns the base model; naming it `model` and deleting it
  detached the shared base, risking OOM or a full base reload on the next adapter.
- *Paper relevance:* Internal only (bug fix).

---

## Measurements & Results Log

> Raw numbers as we collect them. These feed the paper's tables.

### 2026-06-18 — Timing probe #1 (BEFORE dynamic-padding optimization)
- Setup: Colab Pro+, A100 High-RAM. Real settings: `MAX_SAMPLES=3000`, `NUM_EPOCHS=2`,
  rank 16, layer 20, q/k/v/o, Qwen2.5-3B. 1 adapter per benign dataset (8 total).
- Per-adapter (gap between consecutive `STARTING:` log lines):
  | # | dataset | time |
  |---|---|---|
  | 1 | tatsu-lab/alpaca | 4m17s |
  | 2 | databricks-dolly-15k | 4m12s |
  | 3 | gsm8k | 4m22s |
  | 4 | ai2_arc | 1m41s (dataset has <3000 usable rows) |
  | 5 | squad_v2 | 4m18s |
  | 6 | natural_questions | **12m59s** (giant rows; buffer fill — fixed, see change #3) |
  | 7 | openai_humaneval | 0m22s (tiny dataset) |
  | 8 | glue (sst2) | ~4m56s |
- **Typical adapter ≈ 4–4.5 min** at these settings, pre-optimization.
- Cost projection (pre-opt, ~13 units/hr A100): 400 benign ≈ 29 h ≈ ~375 units; full
  pipeline (600 adapters) ≈ ~40 h ≈ ~500 units.
- Note: one mid-run interruption observed that auto-resumed (cause TBD — disconnect vs.
  error; confirm and record). Motivates the crash-safety changes above.

### 2026-06-18 — First real benign run: 100 adapters generated & saved to Drive
- Ran `benignBank.py` with `LBD_MAX_TOTAL=100`, `LBD_OUTPUT_BASE=/content/output_qwen`,
  `LBD_DRIVE_DEST=/content/drive/MyDrive/LoraBackdoorDetection/output_qwen` on A100-80GB.
- Result: `benign_001..benign_100` = 50 alpaca + 50 dolly (the first two datasets, since
  the cap is on TOTAL count and the loop is dataset-ordered). Confirmed 100 dirs on Drive.
- Wall clock: 13:27 → 17:43 ≈ 4h16m, ≈2.5 min/adapter — matches probe #2. No crashes, no
  network drops. 5 Drive checkpoints fired (every 25), all to the Drive path.
- Per-adapter train_runtime ~139s (adapter 1), loss 1.78 → ~1.37 over 2 epochs (learning OK).
- NOTE: this 100-set is NOT yet dataset-diverse (only alpaca+dolly). Diversity (gsm8k,
  ai2_arc, squad_v2, natural_questions, humaneval, glue) arrives in adapters 101–400. Fine
  for building/testing the detector pipeline now; the full benign bank needs the rest.
- Next: poison + test generation, then build_reference_bank → calibrate → evaluate, to shake
  out the downstream pipeline cheaply before scaling benign to 400.

### 2026-06-18 — Timing probe #2 (AFTER dynamic-padding optimization)
- Same setup as probe #1 (A100 High-RAM, real settings, 1 adapter/dataset).
- Per-adapter, before → after:
  | dataset | before | after |
  |---|---|---|
  | alpaca | 4m17s | 2m29s |
  | dolly | 4m12s | 3m11s |
  | gsm8k | 4m22s | 2m46s |
  | ai2_arc | 1m41s | 0m59s |
  | squad_v2 | 4m18s | 2m45s |
  | natural_questions | 12m59s | 8m11s |
  | openai_humaneval | 0m22s | 0m18s |
  | glue | ~4m56s | ~2m26s |
- **Typical adapter ≈ 2.6 min** post-optimization (was ~4.3). ~1.6–2x speedup, no modeling
  change. natural_questions remains the outlier (8 min; ~6.5 h across 50 in the full run —
  candidate for a sample-count cut if needed).
- Cost projection (post-opt, ~13 units/hr): 400 benign ≈ 17 h ≈ ~225 units; full pipeline
  (600) ≈ ~25 h ≈ ~330 units. Budget = 1500 units → comfortable.
- Plan: benign bank generated ~100 first (this run), then to 400 across overnight sessions
  via resume-skip + LBD_MAX_TOTAL.

---

## Open decisions / TODO carried forward

- [ ] Confirm real per-adapter time AFTER dynamic-padding (probe #2) before launching 400.
- [ ] Decide first real benign-run size: timing-measured / ~100 / full 400 (CLAUDE.md).
- [ ] Identify the mid-run interruption cause from probe #1 (disconnect vs error).
- [ ] Verify the target paper's official code-repo URL from the .tex/.bib (still unconfirmed).
- [ ] Apply periodic Drive-sync to `poisonBank.py`/`testSet.py` if they will run overnight
      (currently only `benignBank.py` has it; the other two are smaller).
