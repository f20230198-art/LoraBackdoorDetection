# NIGHT 02: E3, clean-task utility of the diffuse and dsmatch adapters vs matched benign adapters
Purpose: close the "they evade only because they are broken" objection. Inference only (teacher-forced loss on
clean, payload-free, in-format text). Expected: well under 1h on an A100 for ~400 benign + 100 diffuse + ~98 dsmatch.
Writes to Drive and resumes, so if Colab drops just re-run the same cell.
Needs the latest push (evaluation/measure_utility.py) before cloning.

## Cell 1: setup (Runtime -> A100 first)
```
%cd /content
!rm -rf lbd && git clone https://github.com/f20230198-art/LoraBackdoorDetection lbd
%cd /content/lbd
```
## Cell 2
```
from google.colab import drive; drive.mount('/content/drive')
```
## Cell 3
```
!python colab/setup.py
```
## Cell 4: score the three banks (resumable)
```
import os
D='/content/drive/MyDrive/LoraBackdoorDetection/output_qwen'
R=f'{D}/results_asiaccs'; os.makedirs(R, exist_ok=True)
for name in ['benign','diffuse_poison','dsmatch_poison']:
    !cd /content/lbd && LBD_MODEL=qwen LBD_OUTPUT_BASE={D} python evaluation/measure_utility.py {D}/{name} --out {R}/utility_{name}.json
```
## Cell 5: summary (paste me this output)
Compares each attack group to benign adapters trained on the SAME format (matched), plus the overall picture.
```
import json, statistics as st, collections
def load(n): return json.load(open(f'{R}/utility_{n}.json'))['per_adapter']
ben, dif, dsm = load('benign'), load('diffuse_poison'), load('dsmatch_poison')
def stats(rows):
    v=[r['clean_loss'] for r in rows if r['clean_loss']==r['clean_loss']]
    return f"n={len(v)} mean={st.mean(v):.3f} sd={(st.stdev(v) if len(v)>1 else 0):.3f} min={min(v):.3f} max={max(v):.3f}"
print('BENIGN all    ', stats(ben))
print('DIFFUSE all   ', stats(dif))
print('DSMATCH all   ', stats(dsm))
ba=[r for r in ben if r['dataset']=='tatsu-lab/alpaca']
print('\nDiffuse vs benign-alpaca (matched format):')
print(' benign alpaca', stats(ba)); print(' diffuse      ', stats(dif))
print('\nDsmatch vs benign, per dataset (matched format):')
for ds in sorted({r['dataset'] for r in dsm}):
    b=[r for r in ben if r['dataset']==ds]; d=[r for r in dsm if r['dataset']==ds]
    if b and d: print(f' {ds}\n   benign  {stats(b)}\n   dsmatch {stats(d)}')
# fraction of attack adapters inside the benign range (same format)
def inrange(att, pool):
    v=[r['clean_loss'] for r in pool]; lo,hi=min(v),max(v)
    return sum(lo<=r['clean_loss']<=hi for r in att), len(att)
print('\ndiffuse inside benign-alpaca [min,max]: %d/%d'%inrange(dif,ba))
tot=[0,0]
for ds in {r['dataset'] for r in dsm}:
    b=[r for r in ben if r['dataset']==ds]; d=[r for r in dsm if r['dataset']==ds]
    if b and d: a,n=inrange(d,b); tot[0]+=a; tot[1]+=n
print('dsmatch inside matched-benign [min,max]: %d/%d'%tuple(tot))
```
NOTE: this reports the distributions and lets them overlap or not. Do not read it as "attack improves utility".
If an attack group sits clearly above benign (worse loss), say so in the paper and weaken the claim.
