#!/usr/bin/env python3
"""
PEFTGuard detector (E6)  —  train on benign + spiky, score every attack bank.
==============================================================================
Reimplementation of PEFTGuard (Sun et al., IEEE S&P 2025) on our Qwen2.5-3B banks, using the
cached tensors from evaluation/peftguard_features.py. Checkpoints were never released and are
backbone-specific, so a from-scratch reimplementation is the only way to run it on our adapters.

ARCHITECTURE — follows the authors' GQA variant (model/PEFTGuard_llama3_8b.py in their repo):
  separate q and v branches, layers as conv channels:
    conv2d(L -> 16, kernel 8, stride 8) -> flatten -> fc(512) -> LeakyReLU      (per branch)
    concat(q, v) -> fc(128) -> LeakyReLU -> fc(2)
  Theirs convolves full-resolution ΔW; ours convolves the s×s pooled ΔW (see peftguard_features.py).
TRAINING — follows their train.py: Adam lr 1e-4, weight decay 1e-5, cosine LR (T_max=20,
  eta_min=5e-6), batch 4, BCEWithLogits on one-hot labels, <=20 epochs, early stop (patience 5
  after epoch 10), 10% per-class validation split, keep the checkpoint with lowest val loss
  subject to val accuracy not dropping. Only addition: one global scalar rescale fit on the train
  set (pooled ΔW values are ~1e-4 and their script does no normalization; a single scalar
  carries no information about the labels).

CORE RULE: train on benign (0) + spiky poison (1) ONLY. Diffuse / dsmatch / spiky_working are
scored, never trained on. Benign and spiky are split train/test so held-out AUC is honest.

DIFFUSE CONFOUND: benign, spiky and dsmatch train q/v at ONE layer; diffuse trains all 36. A
model that has only seen 2 non-zero planes may flag diffuse just because the other planes are
non-zero. So diffuse is scored twice: `full` (confounded, reported with that caveat) and
`restricted` (every plane the training set never used is zeroed, i.e. the like-for-like
comparison at the detector's layer).

USAGE (Colab, GPU recommended for speed but not required):
    !cd /content/lbd && python evaluation/peftguard_detector.py \
        --cache /content/drive/MyDrive/LoraBackdoorDetection/peftguard_cache \
        --asr_spiky_working /content/drive/MyDrive/LoraBackdoorDetection/results_aaai/spiky_working_400_asr.json \
        --asr_diffuse <diffuse asr json> --asr_dsmatch <dsmatch asr json> \
        --seeds 0,1,2 \
        --out /content/drive/MyDrive/LoraBackdoorDetection/results_asiaccs/peftguard_e6.json
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


# ----------------------------------------------------------------------------- data
def load_bank(cache_root: Path, name: str):
    d = cache_root / name
    if not d.exists():
        return [], None
    files = sorted(d.glob("*.pt"))
    if not files:
        return [], None
    X = torch.stack([torch.load(f).float() for f in files])  # [N, 2L, s, s]
    return [f.stem for f in files], X


def load_asr(path):
    if not path:
        return None
    data = json.loads(Path(path).read_text())
    return {r["adapter"]: r["asr"] for r in data["per_adapter"]}


def split_idx(n, frac, rng):
    idx = list(range(n))
    rng.shuffle(idx)
    k = int(round(n * frac))
    return idx[k:], idx[:k]  # (keep, held)


# ---------------------------------------------------------------------------- model
class Branch(nn.Module):
    def __init__(self, layers, side):
        super().__init__()
        self.conv = nn.Conv2d(layers, 16, kernel_size=8, stride=8)
        self.fc = nn.Linear(16 * (side // 8) ** 2, 512)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        return self.act(self.fc(self.conv(x).flatten(1)))


class PEFTGuard(nn.Module):
    """Input [B, 2L, s, s], layer-major (L0 q, L0 v, L1 q, ...)."""

    def __init__(self, layers, side):
        super().__init__()
        self.q, self.v = Branch(layers, side), Branch(layers, side)
        self.head = nn.Sequential(nn.Linear(1024, 128), nn.LeakyReLU(), nn.Linear(128, 2))

    def forward(self, x):
        return self.head(torch.cat([self.q(x[:, 0::2]), self.v(x[:, 1::2])], dim=1))


# ------------------------------------------------------------------------- training
def run_epoch(model, X, y, device, opt=None, bs=4):
    train = opt is not None
    model.train(train)
    loss_fn = nn.BCEWithLogitsLoss()
    order = torch.randperm(len(X)) if train else torch.arange(len(X))
    tot, correct = 0.0, 0
    with torch.set_grad_enabled(train):
        for i in range(0, len(X), bs):
            b = order[i:i + bs]
            xb, yb = X[b].to(device), y[b].to(device)
            logits = model(xb)
            loss = loss_fn(logits, nn.functional.one_hot(yb, 2).float())
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            tot += loss.item() * len(b)
            correct += (logits.argmax(1) == yb).sum().item()
    return tot / len(X), correct / len(X)


@torch.no_grad()
def scores(model, X, device, bs=16):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(torch.softmax(model(X[i:i + bs].to(device)), 1)[:, 1].cpu())
    return torch.cat(out).numpy() if out else np.array([])


def train_model(Xtr, ytr, Xval, yval, layers, side, device, seed):
    torch.manual_seed(seed)
    model = PEFTGuard(layers, side).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20, eta_min=5e-6)
    best = (float("inf"), -1.0)
    best_state, stale = None, 0
    for epoch in range(20):
        tr_loss, tr_acc = run_epoch(model, Xtr, ytr, device, opt)
        va_loss, va_acc = run_epoch(model, Xval, yval, device)
        sched.step()
        if va_loss < best[0] and va_acc >= best[1]:
            best, stale = (va_loss, va_acc), 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        print(f"    ep{epoch:02d} train {tr_loss:.4f}/{tr_acc:.3f}  val {va_loss:.4f}/{va_acc:.3f}")
        if epoch >= 10 and stale >= 5:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# -------------------------------------------------------------------------- report
def bank_report(names, s, benign_s, thr5, asr=None, thr=0.5):
    """detection = P(poison) >= 0.5 (PEFTGuard's native argmax rule).
    detection_fpr5 = score >= held-out-benign 95th percentile (threshold-calibration-free check).
    auc_vs_benign = threshold-free separability of this bank from held-out benign.
    Working = ASR >= 0.5; working_* counts use the native 0.5 rule."""
    rep = {"n": len(s), "detection": float(np.mean(s >= thr)) if len(s) else None,
           "detection_fpr5": float(np.mean(s >= thr5)) if len(s) else None,
           "auc_vs_benign": float(roc_auc_score(np.r_[np.zeros(len(benign_s)), np.ones(len(s))],
                                                np.r_[benign_s, s])) if len(s) else None,
           "mean_score": float(np.mean(s)) if len(s) else None}
    if asr is not None:
        w = np.array([asr.get(n, -1) >= 0.5 for n in names])
        missing = sum(n not in asr for n in names)
        ws = s[w]
        rep.update({"n_working": int(w.sum()), "asr_missing": missing,
                    "working_caught": int((ws >= thr).sum()),
                    "working_evade": float(np.mean(ws < thr)) if len(ws) else None,
                    "working_mean_score": float(np.mean(ws)) if len(ws) else None})
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True, help="root holding one subdir per bank")
    ap.add_argument("--benign", default="benign")
    ap.add_argument("--spiky", default="poison", help="training positives")
    ap.add_argument("--attacks", default="spiky_working_poison,dsmatch_poison,diffuse_poison")
    ap.add_argument("--diffuse", default="diffuse_poison", help="bank also scored layer-restricted")
    ap.add_argument("--asr_spiky_working", default="")
    ap.add_argument("--asr_dsmatch", default="")
    ap.add_argument("--asr_diffuse", default="")
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.cache)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bn, Xb = load_bank(root, args.benign)
    sn, Xs = load_bank(root, args.spiky)
    if Xb is None or Xs is None:
        raise SystemExit("benign or spiky cache missing")
    attacks = {}
    for name in [a.strip() for a in args.attacks.split(",") if a.strip()]:
        n, X = load_bank(root, name)
        if X is None:
            print(f"skip {name}: no cache")
            continue
        attacks[name] = (n, X)
    asr = {"spiky_working_poison": load_asr(args.asr_spiky_working),
           "dsmatch_poison": load_asr(args.asr_dsmatch),
           "diffuse_poison": load_asr(args.asr_diffuse)}

    layers, side = Xb.shape[1] // 2, Xb.shape[2]
    print(f"benign {len(bn)} | spiky {len(sn)} | attacks {list(attacks)} | input {tuple(Xb.shape[1:])} | {device}")

    results = {"config": vars(args), "seeds": {}}
    for seed in [int(x) for x in args.seeds.split(",")]:
        print(f"\n=== seed {seed} ===")
        rng = random.Random(seed)
        b_tr, b_te = split_idx(len(bn), args.test_frac, rng)
        s_tr, s_te = split_idx(len(sn), args.test_frac, rng)
        b_fit, b_val = split_idx(len(b_tr), 0.1, rng)
        s_fit, s_val = split_idx(len(s_tr), 0.1, rng)
        b_fit, b_val = [b_tr[i] for i in b_fit], [b_tr[i] for i in b_val]
        s_fit, s_val = [s_tr[i] for i in s_fit], [s_tr[i] for i in s_val]

        Xtr = torch.cat([Xb[b_fit], Xs[s_fit]])
        scale = float(Xtr.std()) or 1.0
        used = (Xtr.abs().sum(dim=(0, 2, 3)) > 0)  # planes the training set ever touches

        def prep(X):
            return X / scale

        ytr = torch.tensor([0] * len(b_fit) + [1] * len(s_fit))
        Xval = torch.cat([Xb[b_val], Xs[s_val]])
        yval = torch.tensor([0] * len(b_val) + [1] * len(s_val))
        model = train_model(prep(Xtr), ytr, prep(Xval), yval, layers, side, device, seed)

        sb = scores(model, prep(Xb[b_te]), device)
        ss = scores(model, prep(Xs[s_te]), device)
        y = np.r_[np.zeros(len(sb)), np.ones(len(ss))]
        rec = {"planes_used_in_training": torch.nonzero(used).flatten().tolist(),
               "heldout": {"auc": float(roc_auc_score(y, np.r_[sb, ss])),
                           "fpr": float(np.mean(sb >= 0.5)), "spiky_detection": float(np.mean(ss >= 0.5)),
                           "n_benign": len(sb), "n_spiky": len(ss)},
               "thr_fpr5": float(np.percentile(sb, 95)),
               "attacks": {}}
        thr5 = rec["thr_fpr5"]
        print(f"  held-out AUC {rec['heldout']['auc']:.3f}  FPR {rec['heldout']['fpr']:.3f}  "
              f"spiky det {rec['heldout']['spiky_detection']:.3f}")

        for name, (names, X) in attacks.items():
            s = scores(model, prep(X), device)
            r = bank_report(names, s, sb, thr5, asr.get(name))
            r["per_adapter"] = dict(zip(names, map(float, s)))
            rec["attacks"][name] = r
            print(f"  {name}: det {r['detection']:.3f}  det@fpr5 {r['detection_fpr5']:.3f}  "
                  f"auc {r['auc_vs_benign']:.3f}  mean {r['mean_score']:.3f}"
                  + (f"  working {r['n_working']} evade {r['working_evade']}" if "n_working" in r else ""))
            if name == args.diffuse:
                Xr = X.clone()
                Xr[:, ~used] = 0
                s2 = scores(model, prep(Xr), device)
                r2 = bank_report(names, s2, sb, thr5, asr.get(name))
                r2["per_adapter"] = dict(zip(names, map(float, s2)))
                rec["attacks"][name + "__restricted"] = r2
                print(f"  {name}__restricted: det {r2['detection']:.3f}  det@fpr5 {r2['detection_fpr5']:.3f}  "
                      f"auc {r2['auc_vs_benign']:.3f}  mean {r2['mean_score']:.3f}")
        results["seeds"][str(seed)] = rec

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
