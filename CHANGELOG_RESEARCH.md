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
