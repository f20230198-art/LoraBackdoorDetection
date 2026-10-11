# NIGHT 01: E1 + E2, ASR on the ORIGINAL 1-5% bank + detection at both thresholds
Purpose: the reviewers' #1 point. The AUC-1.00 bank (poison rate 1/3/5%) never had ASR measured.
Expected GPU time: about 2-4h on an A100 (poison bank ~100 adapters + test bank, 20 probes each).
Everything writes to Drive and resumes, so if Colab drops just re-run the same cell.

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
## Cell 4: inventory check (fast; paste me this output even if later cells fail)
```
import os, glob, json
D='/content/drive/MyDrive/LoraBackdoorDetection/output_qwen'
R=f'{D}/results_asiaccs'; os.makedirs(R, exist_ok=True)
for b in ['poison','test']:
    ads=[a for a in sorted(os.listdir(f'{D}/{b}')) if os.path.isfile(f'{D}/{b}/{a}/adapter_config.json')]
    rates={}
    for a in ads:
        m=json.load(open(f'{D}/{b}/{a}/metadata.json')) if os.path.exists(f'{D}/{b}/{a}/metadata.json') else {}
        k=(m.get('label', m.get('is_poison')), m.get('poison_rate', m.get('poisoning_rate')), m.get('attack_type'))
        rates[str(k)]=rates.get(str(k),0)+1
    print(b, len(ads), rates)
cands=glob.glob('/content/lbd/runs/run_aaai/classifier.pkl')+glob.glob('/content/drive/MyDrive/LoraBackdoorDetection/**/run_aaai/classifier.pkl', recursive=True)
print('detector candidates:', cands)
```
## Cell 5: detector scores, CPU-ish, fast (threshold printed MUST be 0.5853211633134577)
```
RUN = os.path.dirname(cands[0]); print(RUN)
for b in ['poison','test']:
    !cd /content/lbd && LBD_MODEL=qwen LBD_OUTPUT_BASE={D} python evaluation/evaluate_diffuse.py --dir {D}/{b} --run_dir {RUN} --out {R}/orig_{b}_eval.json
```
## Cell 6: ASR, the long one (bare probing, the same mode as the original recipe)
```
for b in ['poison','test']:
    !cd /content/lbd && LBD_MODEL=qwen LBD_OUTPUT_BASE={D} python evaluation/measure_asr.py {D}/{b} --n 20 --out {R}/orig_{b}_asr.json
```
## Cell 7: summary (paste me the output)
```
import json, collections
for b in ['poison','test']:
    asr={r['adapter']:r for r in json.load(open(f'{R}/orig_{b}_asr.json'))['per_adapter']}
    ev={r['name']:r['score'] for r in json.load(open(f'{R}/orig_{b}_eval.json'))['per_adapter']}
    print(f'===== {b}: {len(asr)} ASR-scored, {len(ev)} detector-scored')
    rows=[]
    for a,r in asr.items():
        m={}
        p=f'{D}/{b}/{a}/metadata.json'
        if os.path.exists(p): m=json.load(open(p))
        rows.append((a, m.get('poison_rate', m.get('poisoning_rate')), m.get('attack_type'), r['asr'], r['clean_firing_rate'], ev.get(a)))
    by=collections.defaultdict(list)
    for x in rows: by[(x[1],x[2])].append(x)
    for k,v in sorted(by.items(), key=lambda z:str(z[0])):
        a=[x[3] for x in v]
        print(k, 'n=',len(v), 'meanASR=%.3f'%(sum(a)/len(a)), 'fire>=0.5:',sum(x>=0.5 for x in a), 'fire>=0.8:',sum(x>=0.8 for x in a), 'ASR=0:',sum(x==0 for x in a))
    for thr in [0.417, 0.5853211633134577]:
        work=[x for x in rows if x[3]>=0.5 and x[5] is not None]
        allp=[x for x in rows if x[5] is not None]
        print(f' thr {thr:.3f}: detect all {sum(x[5]>=thr for x in allp)}/{len(allp)}  detect working {sum(x[5]>=thr for x in work)}/{len(work)}')
    print(' mean clean-fire %.3f'%(sum(x[4] for x in rows)/len(rows)))
```
NOTE: the test bank also holds benign adapters. Their "ASR" is meaningless, so the summary groups rows by
metadata. Send me all the output from Cells 4, 5 (the threshold line) and 7.
