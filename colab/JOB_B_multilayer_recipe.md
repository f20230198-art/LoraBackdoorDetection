# JOB B — multi-layer aggregated detector (CPU, queue after Job A or run anytime)

Implements `contributions/multilayer_detector_brief.md`. CPU-only; does NOT depend on Job A —
runs on the banks already on Drive. Script: `evaluation/multilayer_detector.py` (must be pushed
from PC before Colab can pull it).

**CORE RULE baked into the script:** trains on benign + standard SPIKY only, never on
diffuse/dsmatch/CBA. Reports BOTH feature reps (concat ~720-dim + aggregated ~80-dim) x BOTH
heads (logistic + RF) = 4 detectors, all scored against every attack + the unseen-benign FPR.
Never claim the gap is closed.

## RUN (CPU runtime is fine; mount Drive first)
```
%cd /content && rm -rf lbd && git clone https://github.com/f20230198-art/LoraBackdoorDetection lbd
%cd /content/lbd
from google.colab import drive; drive.mount('/content/drive')   # separate cell

!cd /content/lbd && LBD_OUTPUT_BASE=/content/drive/MyDrive/LoraBackdoorDetection/output_qwen \
  python evaluation/multilayer_detector.py \
  --num_layers 36 \
  --out /content/drive/MyDrive/LoraBackdoorDetection/results_aaai/multilayer_detector.json
```
Add CBA if you want it in the table:
`--cba_dir /content/drive/MyDrive/LoraBackdoorDetection/output_qwen/cba_poison`
(CBA on Llama-2 trains q/v only -> also set `LBD_DETECTOR_PROJ=q_proj,v_proj`.)

Multi-backbone: `LBD_MODEL=gemma` + `--num_layers 26`, then `LBD_MODEL=llama` + `--num_layers 28`
(point LBD_OUTPUT_BASE at output_gemma / output_llama).

## THE CRITICAL NUMBER: unseen-benign FPR
The two-sided repair failed here at 85%. The script holds out 20% of the benign bank as
"unseen" by default. BETTER: point `--unseen_benign_dir` at benign adapters from datasets NOT
in the training reference, so it's a true distribution-shift test (that's what makes 85% the
honest number). If you have such a bank, pass it.

## STILL TO DO after the JSON lands (I do these once you paste it back)
- Populate the brief's comparison table (§4): 4 multi-layer variants vs deployed/two-sided/supervised.
- **RE-RUN the placement sweep with the multi-layer detector** — `evaluation/placement_curve.py`.
  The current placement curve scores diffuse adapters with the LAYER-20 detector, so the collapse
  is "true by construction" (2 reviewers flagged). Re-scoring with an all-layer detector breaks
  the circularity; the paper must say so explicitly. (If placement_curve.py doesn't accept a
  multi-layer detector, tell me and I'll wire it.)
- One honest paragraph per brief §5: narrows-but-not-closed (if it improves both axes) OR
  paradigm-strengthening negative (if it still evades / FPR blows up).

Paste `multilayer_detector.json` (and placement JSON) back to me.
