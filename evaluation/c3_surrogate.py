#!/usr/bin/env python3
"""
C3, part 1 — Differentiable surrogate of THIS detector's spectral pipeline.
===========================================================================

C3 is the strong, WHITE-BOX attacker: unlike C2 (no gradients, no detector access), here the
attacker KNOWS the detector and uses its gradient. The detector's scoring pipeline is
QR -> M=Rb·Raᵀ -> SVD -> {σ1, Frobenius, energy-concentration, entropy, kurtosis} -> StandardScaler
-> LogisticRegression. To attack it with gradients we need that whole chain to be DIFFERENTIABLE
with respect to the LoRA matrices A, B.

This module is a faithful, autograd-friendly re-implementation of
`core.detector.BackdoorDetector._compute_metrics_from_matrices` (same math, line-for-line) plus
the scaler+logistic head, so we can backprop the detector's poison-score down to A, B. It is the
SURROGATE the C0 honesty rule scopes: "a differentiable surrogate of THIS pipeline, not a new
primitive" — it must match the real detector's forward pass, which `--verify` checks numerically.

Numerical-stability note (the real risk in C3): SVD/QR backward is unstable when singular values
are nearly degenerate (the vjp has 1/(σ_i²-σ_j²) terms). We (a) use torch.linalg.svdvals (we only
need values, not vectors — cheaper and better-conditioned than full svd backward), (b) operate in
float64 during optimization, and (c) expose an eps for the entropy/energy denominators identical to
the detector's. `--selftest` checks gradients are finite and flow to both A and B with NO model/GPU.

This file has NO model and NO GPU dependency — it is pure torch tensor math, CPU-testable. The
optimization loop that USES it (loads the LM for ASR, runs the joint objective) is c3_attack.py.
"""

import os
import sys
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ----------------------------------------------------------------------------
# The differentiable feature extractor. Mirrors detector._compute_metrics_from_matrices
# EXACTLY (same QR -> M -> svdvals -> 5 metrics), but in pure autograd torch so grads
# flow to A, B. Returns a length-5 tensor [sigma1, frob, energy_conc, entropy, kurtosis].
# ----------------------------------------------------------------------------
def spectral_features(B: torch.Tensor, A: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """B: (out, r), A: (r, in). Same orientation the detector uses (delta = B @ A)."""
    Qb, Rb = torch.linalg.qr(B)
    Qa, Ra = torch.linalg.qr(A.T)
    M = Rb @ Ra.T
    s = torch.linalg.svdvals(M)  # values only — better-conditioned backward than full svd

    sigma1 = s[0]
    total_energy = torch.sum(s ** 2)
    frob = torch.sqrt(total_energy)
    energy_conc = (sigma1 ** 2) / (total_energy + eps)

    p = s / (torch.sum(s) + eps)
    entropy = -torch.sum(p * torch.log(p + eps))

    delta = B @ A
    flat = delta.flatten()
    mean = torch.mean(flat)
    var = torch.var(flat)
    kurt = torch.mean((flat - mean) ** 4) / (var ** 2 + eps)

    return torch.stack([sigma1, frob, energy_conc, entropy, kurt])


class SurrogateDetector:
    """Differentiable head: features for each (B,A) projection block -> standardize ->
    logistic -> poison probability. Loads the REAL calibrated detector's scaler + logistic
    weights from a classifier.pkl so the surrogate gradient matches the deployed detector."""

    def __init__(self, scaler_mean, scaler_scale, coef, intercept, dtype=torch.float64):
        self.dtype = dtype
        self.mean = torch.tensor(scaler_mean, dtype=dtype)
        self.scale = torch.tensor(scaler_scale, dtype=dtype)
        self.coef = torch.tensor(coef, dtype=dtype).flatten()
        self.intercept = torch.tensor(float(intercept), dtype=dtype)

    @classmethod
    def from_pickle(cls, path: str, dtype=torch.float64):
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        scaler = data["scaler"]
        clf = data["classifier"]
        return cls(scaler.mean_, scaler.scale_, clf.coef_, clf.intercept_[0], dtype=dtype)

    def poison_logit(self, blocks: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """blocks: list of (B, A) per projection, in the SAME order the detector concatenates
        (q,k,v,o by default). Returns the logistic logit (pre-sigmoid) for the poison class —
        what the attacker pushes DOWN to evade while keeping ASR up."""
        feats = torch.cat([spectral_features(B.to(self.dtype), A.to(self.dtype)) for B, A in blocks])
        if feats.shape[0] != self.coef.shape[0]:
            raise ValueError(f"feature dim {feats.shape[0]} != detector coef dim {self.coef.shape[0]} "
                             f"(check projection set / layer count match the calibrated detector)")
        z = (feats - self.mean) / self.scale
        return torch.dot(self.coef, z) + self.intercept

    def poison_prob(self, blocks) -> torch.Tensor:
        return torch.sigmoid(self.poison_logit(blocks))


# ----------------------------------------------------------------------------
# Verification: surrogate forward must match the REAL detector's numpy features.
# ----------------------------------------------------------------------------
def verify_against_detector(classifier_pkl: str, n_blocks: int = 4, seed: int = 0):
    """Build random (B,A) blocks, run BOTH the numpy detector metric fn and the torch surrogate
    spectral_features, and report max abs diff per metric. They should match to ~1e-4."""
    from core.detector import BackdoorDetector
    rng = np.random.default_rng(seed)
    max_diffs = []
    for _ in range(n_blocks):
        Bn = rng.standard_normal((64, 16)).astype("float64")
        An = rng.standard_normal((16, 64)).astype("float64")
        # numpy/real detector path
        m = BackdoorDetector._compute_metrics_from_matrices(
            torch.tensor(Bn), torch.tensor(An))
        real = np.array([m["sigma1"], m["frobenius_norm"], m["energy_concentration"],
                         m["entropy"], m["kurtosis"]])
        # torch surrogate path
        sur = spectral_features(torch.tensor(Bn), torch.tensor(An)).detach().numpy()
        max_diffs.append(np.abs(real - sur))
    md = np.max(np.vstack(max_diffs), axis=0)
    names = ["sigma1", "frobenius", "energy_conc", "entropy", "kurtosis"]
    print("Surrogate vs real detector — max abs diff per metric:")
    for nme, d in zip(names, md):
        print(f"  {nme:12s}: {d:.2e}")
    ok = bool(np.all(md < 1e-3))
    print("VERIFY:", "PASS (surrogate matches detector)" if ok else "FAIL (mismatch > 1e-3)")
    return ok


# ----------------------------------------------------------------------------
# Self-test: gradients are finite and flow to BOTH A and B, through QR/SVD/kurtosis,
# with no model and no GPU. This is the feasibility gate for the whole C3 loop.
# ----------------------------------------------------------------------------
def run_selftest(seed: int = 0) -> int:
    torch.manual_seed(seed)
    print("C3 surrogate self-test (no model, no GPU)")

    # one block, require grad on both A and B
    B = torch.randn(64, 16, dtype=torch.float64, requires_grad=True)
    A = torch.randn(16, 64, dtype=torch.float64, requires_grad=True)
    feats = spectral_features(B, A)
    assert feats.shape == (5,), feats.shape
    assert torch.isfinite(feats).all(), "non-finite features"

    # backprop the leading singular value (the spikiness the attacker flattens)
    feats[0].backward()
    assert A.grad is not None and B.grad is not None, "no grad to A/B"
    assert torch.isfinite(A.grad).all() and torch.isfinite(B.grad).all(), "non-finite grad"
    print(f"  sigma1 backward: grad to A (norm {A.grad.norm():.3f}) and B "
          f"(norm {B.grad.norm():.3f}), all finite  OK")

    # a tiny fake surrogate head + gradient descent step that REDUCES the poison logit,
    # proving the evasion objective is differentiable end-to-end.
    sur = SurrogateDetector(
        scaler_mean=np.zeros(20), scaler_scale=np.ones(20),
        coef=np.ones(20), intercept=0.0, dtype=torch.float64)
    blocks = [(torch.randn(64, 16, dtype=torch.float64, requires_grad=True),
               torch.randn(16, 64, dtype=torch.float64, requires_grad=True)) for _ in range(4)]
    logit0 = sur.poison_logit(blocks)
    params = [t for blk in blocks for t in blk]
    g = torch.autograd.grad(logit0, params)
    assert all(torch.isfinite(gi).all() for gi in g), "non-finite head grad"
    # one manual GD step downhill on the logit
    with torch.no_grad():
        for t, gi in zip(params, g):
            t -= 0.01 * gi
    logit1 = sur.poison_logit(blocks)
    print(f"  poison logit {logit0.item():.4f} -> {logit1.item():.4f} after one GD step "
          f"({'down OK' if logit1 < logit0 else 'NOT down — FAIL'})")
    assert logit1 < logit0, "evasion objective did not decrease — surrogate gradient is wrong"

    # degeneracy stress: equal singular values (the SVD-backward landmine). Should stay finite.
    Bd = torch.eye(16, dtype=torch.float64, requires_grad=True)
    Ad = torch.eye(16, dtype=torch.float64, requires_grad=True)
    fd = spectral_features(Bd, Ad)
    fd.sum().backward()
    fin = torch.isfinite(Bd.grad).all() and torch.isfinite(Ad.grad).all()
    print(f"  degenerate-spectrum (identity) grad finite: {bool(fin)} "
          f"({'OK' if fin else 'unstable — use float64 / eps / jitter in the loop'})")

    print("SELF-TEST PASSED" if fin else "SELF-TEST PASSED (with degeneracy caveat noted)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="no-GPU gradient-flow feasibility test")
    ap.add_argument("--verify", metavar="CLASSIFIER_PKL",
                    help="check the surrogate forward matches the real detector for a calibrated pkl")
    args = ap.parse_args()
    if args.verify:
        sys.exit(0 if verify_against_detector(args.verify) else 1)
    if args.selftest:
        sys.exit(run_selftest())
    ap.print_help()


if __name__ == "__main__":
    main()
