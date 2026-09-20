"""
Recompute the paper's headline numbers directly from the released per-adapter JSONs.

No GPU, no model download, no dependencies beyond the standard library.

    python verify_claims.py

Each check prints the recomputed value next to the value claimed in the paper.
"""

import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
TAU = 0.5853211633134577  # deployed threshold, Qwen (perfect-separation calibration)


def load(*parts):
    p = os.path.join(R, *parts)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def wilson(x, n, z=1.96):
    """95% Wilson score interval for a proportion, in percent."""
    if n == 0:
        return (0.0, 0.0)
    p = x / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def row(label, got, claimed):
    print(f"  {label:<26} {got:<30} paper: {claimed}")


def join(asr_json, eval_json):
    """Join an ASR file to an eval file by adapter name."""
    asr = {r["adapter"]: r["asr"] for r in asr_json["per_adapter"]}
    score = {r["name"]: r["score"] for r in eval_json["per_adapter"]}
    keys = [k for k in asr if k in score]
    return asr, score, keys


# ---------------------------------------------------------------- Section 4
print("\n=== Sec. 4: the AUC-1.00 bank is behaviorally hollow (n=400) ===")
a = load("results_aaai", "spiky_working_400_asr.json")
e = load("results_aaai", "spiky_working_400_eval.json")

if a and e:
    asr, score, keys = join(a, e)
    fire = [k for k in keys if asr[k] > 0]
    work = [k for k in keys if asr[k] >= 0.5]
    caught = [k for k in work if score[k] >= TAU]

    row("joined adapters", len(keys), 400)
    row("fire at all (ASR>0)",
        f"{len(fire)}/{len(keys)} = {100*len(fire)/len(keys):.1f}% "
        f"[{wilson(len(fire), len(keys))[0]:.1f},{wilson(len(fire), len(keys))[1]:.1f}]",
        "135/400 = 33.8% [29.3,38.5]")
    row("working (ASR>=0.5)",
        f"{len(work)}/{len(keys)} = {100*len(work)/len(keys):.1f}% "
        f"[{wilson(len(work), len(keys))[0]:.1f},{wilson(len(work), len(keys))[1]:.1f}]",
        "60/400 = 15% [11.8,18.8]")
    row("detection, raw bank",
        f"{100*sum(1 for k in keys if score[k] >= TAU)/len(keys):.1f}%", "70%")
    row("detection, working only",
        f"{len(caught)}/{len(work)} = {100*len(caught)/len(work):.1f}% "
        f"[{wilson(len(caught), len(work))[0]:.1f},{wilson(len(caught), len(work))[1]:.1f}]",
        "24/60 = 40% [28.6,52.6]")
    row("evasion, working only",
        f"{100*(len(work)-len(caught))/len(work):.1f}%", "60%")
    row("mean score (working)",
        f"{sum(score[k] for k in work)/len(work):.4f}", "0.530 (< tau = 0.585)")
    row("mean ASR (working)",
        f"{sum(asr[k] for k in work)/len(work):.4f}", "0.61")
    row("clean-trigger firing",
        f"{sum(r['clean_firing_rate'] for r in a['per_adapter'])/len(a['per_adapter']):.4f}",
        "0.00")

    # Trigger family and poison rate are tied by index parity in the generator,
    # so this split is reported but NOT interpreted causally (see paper Sec. 4).
    ctx = [k for k in work if "contextual" in k]
    rare = [k for k in work if "rare_token" in k]
    row("contextual/20% evading",
        f"{sum(1 for k in ctx if score[k] < TAU)}/{len(ctx)}", "34/51")
    row("rare_token/15% evading",
        f"{sum(1 for k in rare if score[k] < TAU)}/{len(rare)}", "2/9")
else:
    print("  [skipped: n=400 files not found]")

# ---------------------------------------------------------------- Section 5
print("\n=== Sec. 5: attacks at the deployed operating point ===")

d = load("results", "diffuse_eval_results.json")
if d:
    row("diffuse detection (bank)", f"{100*d['detection_rate']:.1f}%", "21%")

ds = load("results_c2", "dsmatch_eval_results.json")
dsa = load("results_c2", "dsmatch_asr_results.json")
if ds and dsa:
    asr, score, keys = join(dsa, ds)
    work = [k for k in keys if asr[k] >= 0.5]
    caught = [k for k in work if score[k] >= TAU]
    row("dsmatch detection (bank)", f"{100*ds['detection_rate']:.1f}%", "0%")
    row("dsmatch mean score", f"{ds['mean_score']:.4f}", "0.051")
    row("dsmatch n working", len(work), "52")
    row("dsmatch caught (working)", f"{len(caught)}/{len(work)}", "0/52")
    if work:
        row("dsmatch max score (working)",
            f"{max(score[k] for k in work):.4f}", "0.294 (< 0.417/0.389/0.377)")
        row("dsmatch mean ASR (working)",
            f"{sum(asr[k] for k in work)/len(work):.4f}", "0.92")

cba = load("results", "cba_eval_pii-masker.json")
if cba:
    row("CBA detection", f"{100*cba['detection_rate']:.1f}%", "0%")

# ------------------------------------------------- partial-coverage sweep
print("\n=== Sec. 5: partial dataset-matching (a partial guess suffices) ===")
for k in (2, 4, 6):
    pa = load("output_qwen", "results_aaai", f"partial_k{k}_asr.json")
    pe = load("output_qwen", "results_aaai", f"partial_k{k}_eval.json")
    if pa and pe:
        asr, score, keys = join(pa, pe)
        work = [x for x in keys if asr[x] >= 0.5]
        caught = [x for x in work if score[x] >= TAU]
        ev = len(work) - len(caught)
        lo = wilson(ev, len(work))[0] if work else 0.0
        row(f"k={k}: evasion (working)",
            f"{ev}/{len(work)} = {100*ev/len(work):.0f}% (Wilson lo {lo:.1f}%)",
            "100%, lower bound > 87%")

# ---------------------------------------------------------------- Sec. 5.3
print("\n=== Sec. 5.3: multi-backbone replication ===")
for name, spiky, diff, dsm in [
    ("Gemma-2-2B", ("output_gemma", "results_review", "spiky_eval_results.json"),
                   ("output_gemma", "results_review", "diffuse_eval_results.json"),
                   ("output_gemma", "results_review", "dsmatch_eval_results.json")),
    ("Llama-3.2-3B", ("output_llama", "results_review", "spiky_eval_llama.json"),
                     ("output_llama", "results_review", "diffuse_eval_llama.json"),
                     ("output_llama", "results_review", "dsmatch_eval_llama.json")),
]:
    s, df, dm = load(*spiky), load(*diff), load(*dsm)
    if s and df and dm:
        row(f"{name} spiky det.", f"{100*s['detection_rate']:.1f}%",
            "100% (Gemma) / 95% (Llama)")
        row(f"{name} diffuse det.", f"{100*df['detection_rate']:.1f}%",
            "5.1% (Gemma) / 0% (Llama)")
        row(f"{name} dsmatch det.", f"{100*dm['detection_rate']:.1f}%", "0%")

# ---------------------------------------------------------------- Sec. 5.4
print("\n=== Sec. 5.4: seed robustness ===")
for label, fn, claim in [
    ("diffuse", "diffuse_seeds_CI.json", "det 24+-7%, evasion 85+-8%"),
    ("dsmatch", "dsmatch_seeds_CI.json", "det 0+-0%, evasion 100+-0%"),
]:
    ci = load("output_qwen", "results_review", fn)
    if ci:
        agg = ci.get("aggregate", {})
        det, ev = agg.get("detection_all", {}), agg.get("evasion_working", {})
        got = (f"det {100*det.get('mean', 0):.1f}+-{100*det.get('ci95', 0):.0f}%, "
               f"evasion {100*ev.get('mean', 0):.1f}+-{100*ev.get('ci95', 0):.0f}% "
               f"(n={agg.get('detection_all', {}).get('n', '?')} seeds)")
        row(f"{label} seeds", got, claim)

print("\nDone. Values on the left are recomputed from the released JSONs.\n")
