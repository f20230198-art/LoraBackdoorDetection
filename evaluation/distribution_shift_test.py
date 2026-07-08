#!/usr/bin/env python3
"""
Distribution-shift test — is the two-sided detector a real fix or the C1 confound in disguise?
==============================================================================================

detector_head_ablation.py (Q3) showed a two-sided Mahalanobis "distance from benign in ANY
direction" detector catches diffuse (100%) and dsmatch (80.6%) where the one-sided logistic/RF
fail. That looks like a fix. This script tests the honest catch: a two-sided detector flags
*anything unlike its reference benign*, so a benign adapter from a dataset the defender did NOT
include should also read as anomalous -> a false alarm. If so, the "fix" just trades the diffuse
blind spot for distribution-shift false positives (the C1 dataset confound, defender-side).

SECTION A (the confound test). Build the benign reference on K of the 8 datasets; hold out the
rest. Fit the two-sided detector on the reference benign, threshold at ~5% FPR on it, then report:
  * FPR on the HELD-OUT benign datasets (the number that matters — should blow past 5% if the
    "fix" is really the confound),
  * detection on diffuse / dsmatch (in-distribution attacks, should stay high).

SECTION B (C5 dataset-aware calibration, leave-datasets-out). The C5 claim was that folding
dsmatch into calibration recovers dsmatch detection. Does it GENERALIZE across datasets or just
memorize the ones it saw? Train a supervised head on (reference benign + dsmatch-from-reference-
datasets) and test on (held-out benign + dsmatch-from-held-out-datasets). Reports held-out dsmatch
detection at the held-out-benign FPR. Low generalization => the recovery is dataset-memorization.

CPU only, on existing banks. Benign + dsmatch adapters carry metadata['dataset']; diffuse is
alpaca-only so it is treated as one in-distribution attack group.

USAGE:
  LBD_MODEL=qwen LBD_OUTPUT_BASE=$DRIVE python evaluation/distribution_shift_test.py
  # optionally fix the held-out benign datasets:
  ... --holdout squad_v2,glue
Writes evaluation/distribution_shift_results.json (override with --out).
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from core.detector import BackdoorDetector

RS = 42


def is_adapter_dir(p: str) -> bool:
    return os.path.isfile(os.path.join(p, "adapter_config.json"))


def _meta_dataset(d: str) -> str:
    mp = os.path.join(d, "metadata.json")
    if os.path.isfile(mp):
        try:
            return json.load(open(mp)).get("dataset") or "unknown"
        except Exception:
            pass
    return "unknown"


def features_by_dataset(bank_dir: str, layer: int):
    """dataset_name -> (N_ds, 20) feature array."""
    groups = {}
    if not os.path.isdir(bank_dir):
        return groups
    for name in sorted(os.listdir(bank_dir)):
        d = os.path.join(bank_dir, name)
        if not is_adapter_dir(d):
            continue
        f = BackdoorDetector._extract_features_from_adapter(Path(d), layer)
        if f is None:
            continue
        groups.setdefault(_meta_dataset(d), []).append(f)
    return {k: np.vstack(v) for k, v in groups.items()}


def flat_matrix(bank_dir: str, layer: int) -> np.ndarray:
    g = features_by_dataset(bank_dir, layer)
    return np.vstack(list(g.values())) if g else np.empty((0, 0))


def maha_fn(mu, Cinv):
    def f(X):
        d = X - mu
        return np.einsum("ij,jk,ik->i", d, Cinv, d)
    return f


def rate(scores, thr):
    return float(np.mean(np.asarray(scores) >= thr)) if len(scores) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="", help="comma-sep benign datasets to hold out "
                    "(default: the 2 with the most adapters)")
    ap.add_argument("--layer", type=int, default=config.TARGET_LAYERS[0])
    ap.add_argument("--out", default="evaluation/distribution_shift_results.json")
    args = ap.parse_args()
    L = args.layer

    print(f"[extract] benign by dataset (layer {L}) ...")
    benign = features_by_dataset(config.BENIGN_DIR, L)
    if len(benign) < 3:
        sys.exit(f"Need benign grouped by >=3 datasets; got {list(benign)}. "
                 f"Check metadata['dataset'] in {config.BENIGN_DIR}.")
    print("  benign datasets: " + ", ".join(f"{k}={len(v)}" for k, v in benign.items()))
    diffuse = flat_matrix(config.DIFFUSE_POISON_DIR, L)
    dsmatch_by = features_by_dataset(config.DSMATCH_POISON_DIR, L)
    print(f"  diffuse n={len(diffuse)}; dsmatch datasets: "
          + ", ".join(f"{k}={len(v)}" for k, v in dsmatch_by.items()))

    # choose held-out benign datasets
    if args.holdout.strip():
        holdout = [x.strip() for x in args.holdout.split(",") if x.strip() in benign]
    else:
        holdout = [k for k, _ in sorted(benign.items(), key=lambda kv: -len(kv[1]))[:2]]
    ref_ds = [k for k in benign if k not in holdout]
    print(f"\n[split] reference benign datasets: {ref_ds}")
    print(f"[split] HELD-OUT benign datasets:  {holdout}")

    Xref = np.vstack([benign[k] for k in ref_ds])
    Xhold = np.vstack([benign[k] for k in holdout])

    scaler = StandardScaler().fit(Xref)
    Zref, Zhold = scaler.transform(Xref), scaler.transform(Xhold)
    Zdiff = scaler.transform(diffuse) if len(diffuse) else np.empty((0, Xref.shape[1]))

    results = {"layer": L, "reference_datasets": ref_ds, "holdout_datasets": holdout}

    # ---- SECTION A: two-sided detector, in-dist threshold, OOD-benign FPR --------------------
    mu = Zref.mean(0)
    cov = np.cov(Zref, rowvar=False) + 1e-6 * np.eye(Zref.shape[1])
    maha = maha_fn(mu, np.linalg.pinv(cov))
    thr = float(np.quantile(maha(Zref), 0.95))   # 5% FPR on reference benign
    fpr_ref = rate(maha(Zref), thr)
    fpr_ood = rate(maha(Zhold), thr)
    det_diff = rate(maha(Zdiff), thr) if len(Zdiff) else None
    Zdsm_all = scaler.transform(np.vstack(list(dsmatch_by.values()))) if dsmatch_by else None
    det_dsm = rate(maha(Zdsm_all), thr) if Zdsm_all is not None else None

    print("\n" + "=" * 68)
    print("SECTION A — two-sided (Mahalanobis) detector, threshold @5% FPR on reference benign")
    print("=" * 68)
    print(f"  FPR on reference benign (by construction)  : {fpr_ref*100:5.1f}%")
    print(f"  FPR on HELD-OUT benign (unseen datasets)   : {fpr_ood*100:5.1f}%   <-- the tell")
    if det_diff is not None:
        print(f"  detection on diffuse (in-dist attack)      : {det_diff*100:5.1f}%")
    if det_dsm is not None:
        print(f"  detection on dsmatch (in-dist attack)      : {det_dsm*100:5.1f}%")
    print("READ: if held-out benign FPR >> 5%, the two-sided 'fix' is the C1 dataset confound")
    print("      wearing a defender's hat — it flags innocent adapters from unseen distributions.")
    results["section_A_two_sided"] = {
        "threshold": thr, "fpr_reference": fpr_ref, "fpr_heldout_benign": fpr_ood,
        "detection_diffuse": det_diff, "detection_dsmatch": det_dsm,
    }

    # ---- SECTION B: dataset-aware supervised calibration, leave-datasets-out -----------------
    if dsmatch_by and len(dsmatch_by) >= 3:
        ds_ref = [k for k in dsmatch_by if k not in holdout]
        ds_hold = [k for k in dsmatch_by if k in holdout]
        if ds_ref and ds_hold:
            Xdsm_ref = np.vstack([dsmatch_by[k] for k in ds_ref])
            Xdsm_hold = np.vstack([dsmatch_by[k] for k in ds_hold])
            Xtr = np.vstack([Xref, Xdsm_ref])
            ytr = np.hstack([np.zeros(len(Xref)), np.ones(len(Xdsm_ref))])
            sc2 = StandardScaler().fit(Xtr)
            clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc2.transform(Xtr), ytr)
            # threshold @5% FPR on HELD-OUT benign (honest: FPR on data not used in training)
            pb = clf.predict_proba(sc2.transform(Xhold))[:, 1]
            t2 = float(np.quantile(pb, 0.95)) if len(pb) else 0.5
            det_hold_dsm = rate(clf.predict_proba(sc2.transform(Xdsm_hold))[:, 1], t2)
            print("\n" + "=" * 68)
            print("SECTION B — C5 dataset-aware calibration, LEAVE-DATASETS-OUT")
            print("=" * 68)
            print(f"  trained on dsmatch from: {ds_ref}")
            print(f"  tested on dsmatch from : {ds_hold}   (held-out datasets)")
            print(f"  held-out dsmatch detection @5% held-out-benign FPR: {det_hold_dsm*100:5.1f}%")
            print("READ: high => recovery generalizes across datasets; low => it MEMORIZED the")
            print("      calibration datasets and an unanticipated distribution slips again.")
            results["section_B_dataset_aware_leave_out"] = {
                "train_datasets": ds_ref, "test_datasets": ds_hold,
                "heldout_dsmatch_detection": det_hold_dsm, "threshold": t2,
            }
        else:
            print("\n[Section B skipped] dsmatch datasets don't straddle the benign holdout split.")
    else:
        print("\n[Section B skipped] need dsmatch grouped by >=3 datasets.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
