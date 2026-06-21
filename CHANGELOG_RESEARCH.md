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
