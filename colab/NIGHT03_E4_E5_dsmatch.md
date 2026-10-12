# NIGHT 03: E4 (random-subset dsmatch) + E5 (controlled-recipe dsmatch)

Two reviewer asks about dataset-matching. Both reuse the existing attack code, both are
scored with the unchanged detector (`runs/run_aaai`, tau = 0.5853211633134577).

- **E4 (ri7q).** The partial sweep handed the attacker its best-planting datasets. E4 draws
  subsets **uniformly at random from all 8**, non-planting ones included. k = 2, 4, 6;
  5 random draws per k, 8 adapters per draw = 40 per k = **120 adapters**.
- **E5 (AI reviewer).** dsmatch used a stronger recipe than the spiky bank (5-15% poison,
  6 epochs, lr 2e-4/3e-4). E5 holds poison rate, epochs and lr at the **reproduction values**
  (1/3/5%, 2 epochs, lr 1e-4/2e-4/3e-4, same batch schedule) and changes only the mixture.
  **100 adapters**, same index layout as the original poison bank (0-49 rare_token,
  50-99 contextual).

**Cost.** 220 adapters. E5 is cheap (2 epochs). Expect about 8-12 A100-hours total, so
two nights: night A = E5 + E4 k=2, night B = E4 k=4/6. Then ASR probing (about 2-3h).

**Expected E5 outcome, be ready for it:** at 1-5% / 2 epochs the original spiky bank never
fired (E1: ASR 0/100). E5 will probably fire weakly too. That's fine. E5's main readout is
**detection over ALL adapters versus the original poison bank (100%) at the same recipe**.
Function doesn't matter for that comparison, so the only variable is the mixture. Working-only
evasion is reported as a secondary number if any adapters fire.

---

## STEP 0: setup

**Push from PC first** (`dsmatchPartialBank.py` random mode is new). Runtime -> A100.

```python
%cd /content && rm -rf lbd && git clone https://github.com/f20230198-art/LoraBackdoorDetection lbd
%cd /content/lbd
from google.colab import drive; drive.mount('/content/drive')
!python colab/setup.py     # STEP 3 must show A100 / CUDA True
!grep -n "LBD_DSMATCH_RANDOM" bankCreation/dsmatchPartialBank.py | head -2   # must print lines, else not pushed
```

```python
D = "/content/drive/MyDrive/LoraBackdoorDetection/output_qwen"
R = "/content/drive/MyDrive/LoraBackdoorDetection/results_asiaccs"
import os; os.makedirs(R, exist_ok=True)
```

---

## STEP 1: E5 controlled bank (night A, ~2-3h)

Writes to `output_qwen/e5_controlled/dsmatch_poison`. Only env knobs change; no code.

```python
!cd /content/lbd && LBD_OUTPUT_BASE={D}/e5_controlled LBD_NUM_DSMATCH=100 \
  LBD_DSMATCH_POISON_RATES=0.01,0.03,0.05 LBD_DSMATCH_EPOCHS=2 LBD_DSMATCH_LRS=1e-4,2e-4,3e-4 \
  LBD_SYNC_EVERY=10 python bankCreation/datasetMatchPoisonBank.py
```

Sanity check (must say `epochs: 2` and lr in {1e-4, 2e-4, 3e-4}):
```python
!ls {D}/e5_controlled/dsmatch_poison | wc -l
!cat {D}/e5_controlled/dsmatch_poison/$(ls {D}/e5_controlled/dsmatch_poison | head -1)/metadata.json
```

## STEP 2: E4 random banks (~1.5-2.5h each)

```python
for k in [2, 4, 6]:
    !cd /content/lbd && LBD_OUTPUT_BASE={D} LBD_DSMATCH_RANDOM=1 LBD_DSMATCH_MATCH_K={k} \
      LBD_DSMATCH_DRAWS=5 LBD_NUM_DSMATCH_PARTIAL=40 LBD_SYNC_EVERY=10 \
      python bankCreation/dsmatchPartialBank.py
```

Progress (each k should reach 40 adapters + subsets.json):
```python
!for k in 2 4 6; do echo -n "k=$k: "; ls {D}/dsmatch_random_k$k 2>/dev/null | wc -l; done
```

Re-running any cell is safe because resume-skip keeps finished adapters.

---

## STEP 3: ASR (`--scaffold` IS MANDATORY, or gsm8k/arc/NQ read false zeros)

```python
banks = {"e5": f"{D}/e5_controlled/dsmatch_poison",
         **{f"e4_k{k}": f"{D}/dsmatch_random_k{k}" for k in [2, 4, 6]}}
for tag, path in banks.items():
    !cd /content/lbd && python evaluation/measure_asr.py {path} --scaffold --out {R}/{tag}_asr.json
```

## STEP 4: score with the UNCHANGED detector (CPU, minutes)

Threshold printed must be 0.5853211633134577. If it differs, stop.

```python
for tag, path in banks.items():
    !cd /content/lbd && python evaluation/evaluate_diffuse.py --dir {path} --run_dir runs/run_aaai --out {R}/{tag}_eval.json
```

---

## STEP 5: tables (paste the output back to me)

```python
import json, math, re
THR = 0.5853211633134577
def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k/n; d = 1+z*z/n; c = (p+z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return max(0, c-h), min(1, c+h)

def rows(tag):
    asr = json.load(open(f"{R}/{tag}_asr.json")); ev = json.load(open(f"{R}/{tag}_eval.json"))
    amap = {a["adapter"]: a["asr"] for a in asr["per_adapter"]}
    miss = [e["name"] for e in ev["per_adapter"] if e["name"] not in amap]
    if miss: print(f"  !! {tag}: {len(miss)} adapters missing ASR, e.g. {miss[:2]}")
    return [(e["name"], e["score"], amap.get(e["name"], 0.0)) for e in ev["per_adapter"]]

def line(label, rs):
    wk = [r for r in rs if r[2] >= 0.5]; caught = sum(r[1] >= THR for r in wk)
    det_all = sum(r[1] >= THR for r in rs) / max(1, len(rs))
    mean_all = sum(r[1] for r in rs) / max(1, len(rs))
    if wk:
        lo, hi = wilson(len(wk)-caught, len(wk))
        w = f"work {len(wk):>3}  caught {caught:>3}  evade {100*(len(wk)-caught)/len(wk):5.1f}% [{100*lo:4.1f},{100*hi:4.1f}]"
    else:
        w = "work   0  (nothing fired)"
    print(f"{label:<44} n {len(rs):>3}  det_all {100*det_all:5.1f}%  mean {mean_all:.3f}  {w}")

print("=== E5 controlled recipe (1/3/5%, 2ep, lr 1e-4..3e-4) ===")
e5 = rows("e5")
line("E5 all", e5)
for at in ["rare_token", "contextual"]:
    line(f"  {at}", [r for r in e5 if at in r[0]])
print(f"  mean ASR {sum(r[2] for r in e5)/len(e5):.3f}  | reference: original poison bank det 100%, ASR 0/100")

print("\n=== E4 random subsets ===")
for k in [2, 4, 6]:
    rs = rows(f"e4_k{k}"); sub = json.load(open(f"{D}/dsmatch_random_k{k}/subsets.json"))
    line(f"k={k} pooled", rs)
    for j, names in enumerate(sub["draws"]):
        part = [r for r in rs if j*sub["per_draw"] <= int(re.search(r"dsmatch_(\d+)", r[0]).group(1)) < (j+1)*sub["per_draw"]]
        line(f"  draw {j}: {'+'.join(n.split('/')[-1][:8] for n in names)}", part)
print("\nreference: hand-picked partial sweep evade 100% at k=2/4/6/8")
```

---

## HOW TO READ IT

- **E4:** if evasion among working stays near 100% for every draw that fires anything,
  the hand-picked subsets weren't flattering the result. Draws made of non-planting datasets
  will show "nothing fired". Report that as a planting cost of random knowledge, not as
  detection.
- **E5:** if det_all is far below the original bank's 100% at the identical recipe, the
  mixture alone moves the detector, and the "stronger recipe" objection is answered. If
  det_all is near 100%, the evasion owes something to the recipe. Then we say so and the
  §Threat Models claim gets scoped.
- **Caveat to disclose:** E5 still uses scaffold-matched payload injection (dsmatch's
  delivery), the original bank does not. So the variable is "mixture + delivery format",
  not mixture alone.
