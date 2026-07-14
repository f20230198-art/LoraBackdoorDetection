# CELL 7 — print the exact numbers for the write-up (run after Cells 5 & 6)

Reads the three JSONs and prints paper-ready numbers: the planting-yield line, the
FUNCTION-GATED detection (detection among WORKING backdoors only — the crux of §4), a
fixed-poison-rate readout (answers the 1–5% vs 15–20% objection), and the multi-layer table.
**Paste the printed output back to me — that's all I need.** Self-contained; paste as one cell.

```python
import json, re

RES    = "/content/drive/MyDrive/LoraBackdoorDetection/results_aaai"
ASR_J  = f"{RES}/spiky_working_n50_asr.json"
EVAL_J = f"{RES}/spiky_working_n50_eval.json"
ML_J   = f"{RES}/multilayer_detector.json"

def load(p):
    try:
        return json.load(open(p))
    except Exception as e:
        print(f"!! could not load {p}: {e}")
        return None

def pct(x):
    return "n/a" if x is None else f"{100*x:.1f}%"

# ------------------------------------------------------------------ JOB A
print("="*72)
print("JOB A — spiky-working bank (the n=5 -> n>=50 reframe)")
print("="*72)
asr, ev = load(ASR_J), load(EVAL_J)
if asr and ev:
    asr_by = {r["adapter"]: r for r in asr["per_adapter"]}   # ASR json key = 'adapter'
    sc_by  = {r["name"]:    r for r in ev["per_adapter"]}    # eval json key = 'name'
    thr    = ev["threshold"]
    names  = [n for n in asr_by if n in sc_by]
    total  = len(names)
    firing  = [n for n in names if asr_by[n]["asr"] >  0.0]
    working = [n for n in names if asr_by[n]["asr"] >= 0.5]

    print(f"threshold (deployed):        {thr:.6f}")
    print(f"adapters generated & scored: {total}")
    print(f"  firing  (ASR>0):    {len(firing):3d}  ({pct(len(firing)/total)})")
    print(f"  WORKING (ASR>=0.5): {len(working):3d}  ({pct(len(working)/total)})   <- the n we grew from 5")

    if working:
        w_scores = [sc_by[n]["score"] for n in working]
        caught   = [n for n in working if sc_by[n]["score"] >= thr]
        evaded   = [n for n in working if sc_by[n]["score"] <  thr]
        w_asr    = [asr_by[n]["asr"] for n in working]
        print("\nFUNCTION-GATED DETECTION (among WORKING backdoors only) -- the §4 headline:")
        print(f"  caught: {len(caught):3d}/{len(working)}  ({pct(len(caught)/len(working))})")
        print(f"  evaded: {len(evaded):3d}/{len(working)}  ({pct(len(evaded)/len(working))})  <- 'X% of WORKING backdoors evade'")
        print(f"  mean detector score among working: {sum(w_scores)/len(w_scores):.4f}  (thr {thr:.4f})")
        print(f"  mean ASR among working:            {sum(w_asr)/len(w_asr):.4f}")

    all_caught = sum(1 for n in names if sc_by[n]["score"] >= thr)
    print(f"\nraw detection over WHOLE bank (non-gated): {all_caught}/{total} ({pct(all_caught/total)})")
    print(f"mean ASR over whole bank:                  {sum(asr_by[n]['asr'] for n in names)/total:.4f}")

    print("\nFixed-poison-rate detection among working (answers 1-5% vs 15-20% objection):")
    prs = {}
    for n in working:
        m = re.search(r"pr(\d+)", n)
        prs.setdefault(m.group(1) if m else "?", []).append(n)
    for pr, ns in sorted(prs.items()):
        c = sum(1 for n in ns if sc_by[n]["score"] >= thr)
        print(f"  pr={pr}%:  working={len(ns):2d}  caught={c}  detection={pct(c/len(ns))}")

# ------------------------------------------------------------------ JOB B
print("\n" + "="*72)
print("JOB B — multi-layer detector (both reps x both heads)")
print("="*72)
ml = load(ML_J)
if ml:
    cfg = ml.get("config", {})
    print(f"trained on: {cfg.get('train')}   layers={cfg.get('num_layers')}   "
          f"unseen-benign n={cfg.get('unseen_benign_n')}")
    atk_names = sorted({a for d in ml["detectors"].values() for a in d.get("attacks", {})})
    hdr = f"{'detector':22s} {'spikyAUC':>8s} {'spikyDet':>8s} {'unseenFPR':>9s}"
    hdr += "".join(f" {a[:9]:>9s}" for a in atk_names)
    print(hdr); print("-"*len(hdr))
    for key, d in ml["detectors"].items():
        row = (f"{key:22s} {str(d.get('spiky_auc')):>8s} "
               f"{pct(d.get('spiky_detection')):>8s} {pct(d.get('unseen_benign_fpr')):>9s}")
        for a in atk_names:
            row += f" {pct(d.get('attacks',{}).get(a,{}).get('detection')):>9s}"
        print(row)
    print("\nReminder: multi-layer numbers only count for the paper if the placement sweep is ALSO")
    print("re-run with a multi-layer detector (placement_curve.py) — breaks 'true by construction'.")
```
