import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import safetensors.torch as st
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler


class BackdoorDetector:
    """
    Multivariate detector that extracts five spectral features from each
    target projection matrix and combines them with logistic regression.

    For the current setup with q/k/v/o projections, this yields a 20-dim
    feature vector per adapter.
    """

    def __init__(self, bank=None, model_path: Optional[str] = None):
        self.bank = bank  # kept for backward compatibility, unused
        self.classifier = None
        self.scaler = None
        self.threshold = 0.5

        if model_path and Path(model_path).exists():
            self.load(model_path)

    # --- C5 multi-layer pooling (defense) ----------------------------------
    # The baseline detector reads ONE layer (single-layer assumption, C1 Finding A),
    # which the C2 diffuse attack exploits by spreading ΔW across all layers so the
    # one inspected layer looks benign. The C5 repair pools spectral features across
    # SEVERAL layers, so a thin-but-broad spread still shows up in the aggregate.
    #
    # This is OPT-IN and additive: when LBD_DETECTOR_LAYERS is set (comma-separated
    # layer indices), feature extraction routes to _extract_multilayer_features and
    # the detector becomes multi-layer. When unset, every code path below is
    # byte-identical to the frozen single-layer baseline (C1's line citations and the
    # AUC-1.00 reproduction are unchanged). Pooling mode via LBD_DETECTOR_POOL:
    #   concat (default) — per-layer 5-metric blocks concatenated (highest capacity)
    #   max              — element-wise max across layers (a spike anywhere survives)
    #   mean             — element-wise mean across layers (broad-but-flat survives)
    @staticmethod
    def _multilayer_config() -> tuple[Optional[list[int]], str]:
        layers_env = os.environ.get("LBD_DETECTOR_LAYERS", "").strip()
        if not layers_env:
            return None, "concat"
        layers = [int(x) for x in layers_env.split(",") if x.strip() != ""]
        pool = os.environ.get("LBD_DETECTOR_POOL", "concat").strip().lower()
        if pool not in ("concat", "max", "mean"):
            pool = "concat"
        return (layers or None), pool

    @property
    def weights(self) -> Optional[np.ndarray]:
        if self.classifier is None:
            return None
        return self.classifier.coef_.flatten()

    @property
    def intercept(self) -> Optional[float]:
        if self.classifier is None:
            return None
        return float(self.classifier.intercept_[0])

    @staticmethod
    def _select_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, str]:
        benign_scores = y_score[y_true == 0]
        poison_scores = y_score[y_true == 1]

        if len(benign_scores) > 0 and len(poison_scores) > 0:
            benign_max = float(np.max(benign_scores))
            poison_min = float(np.min(poison_scores))
            if benign_max < poison_min:
                separation = poison_min - benign_max
                return benign_max + 0.25 * separation, "perfect_separation_margin"

        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        youden = tpr - fpr
        best_idx = int(np.argmax(youden))
        return float(thresholds[best_idx]), "youden_j"

    def _features(self, adapter_path: Path, layer_idx: int) -> Optional[np.ndarray]:
        """Route to the multi-layer pooler (C5 defense) iff LBD_DETECTOR_LAYERS is set,
        else the frozen single-layer baseline. Keeping the routing in one place means
        calibrate() and scan() always agree on the feature space."""
        layers, pool = self._multilayer_config()
        if layers is not None:
            return self._extract_multilayer_features(adapter_path, layers, pool)
        return self._extract_features_from_adapter(adapter_path, layer_idx)

    def calibrate(
        self,
        poison_paths,
        benign_paths,
        layer_idx: int = 20,
        val_split: float = 0.2,
        C: float = 0.1,
        random_state: int = 42,
        train_on_val: bool = False,
    ) -> Dict[str, Any]:
        layers, pool = self._multilayer_config()
        if layers is not None:
            print(f"[C5] multi-layer pooling ON: layers={layers} pool={pool}")

        print("Extracting features from benign adapters...")
        benign_features = []
        benign_valid_paths = []
        for path in benign_paths:
            feat = self._features(Path(path), layer_idx)
            if feat is not None:
                benign_features.append(feat)
                benign_valid_paths.append(path)
        print(f"Extracted {len(benign_features)} benign feature vectors.")

        print("Extracting features from poisoned adapters...")
        poison_features = []
        poison_valid_paths = []
        for path in poison_paths:
            feat = self._features(Path(path), layer_idx)
            if feat is not None:
                poison_features.append(feat)
                poison_valid_paths.append(path)
        print(f"Extracted {len(poison_features)} poisoned feature vectors.")

        if len(benign_features) == 0 or len(poison_features) == 0:
            raise ValueError("No valid features extracted.")

        # --- Optional benign subsampling to control class imbalance ---
        # A 20:1 benign:poison ratio lets a near-trivial classifier post a
        # high AUC. `balance_ratio` caps benign at N x poison so the reported
        # number reflects real separability, not the prior. Set via
        # DETECTOR_BALANCE_RATIO (0 / unset = no balancing, original behaviour).
        balance_ratio = float(os.environ.get("DETECTOR_BALANCE_RATIO", "0"))
        if balance_ratio > 0 and len(benign_features) > balance_ratio * len(poison_features):
            keep = int(balance_ratio * len(poison_features))
            rng0 = np.random.default_rng(random_state)
            sel = rng0.choice(len(benign_features), size=keep, replace=False)
            benign_features = [benign_features[i] for i in sel]
            benign_valid_paths = [benign_valid_paths[i] for i in sel]
            print(f"Balanced benign down to {keep} (ratio {balance_ratio}:1 vs "
                  f"{len(poison_features)} poison).")

        X = np.vstack(benign_features + poison_features)
        y = np.hstack([np.zeros(len(benign_features)), np.ones(len(poison_features))])
        paths = benign_valid_paths + poison_valid_paths

        # --- STRATIFIED split ---
        # The previous code did a single global shuffle + slice. With few
        # poison adapters that routinely puts ZERO poison in the validation
        # fold (observed: val_counts poison=0), making threshold selection
        # and any val metric meaningless. Split each class separately so both
        # folds always contain both classes.
        rng = np.random.default_rng(random_state)
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        rng.shuffle(pos_idx)
        rng.shuffle(neg_idx)

        def _split(idx):
            n_tr = max(1, int(round(len(idx) * (1 - val_split))))
            n_tr = min(n_tr, len(idx) - 1) if len(idx) > 1 else len(idx)
            return idx[:n_tr], idx[n_tr:]

        pos_tr, pos_val = _split(pos_idx)
        neg_tr, neg_val = _split(neg_idx)
        train_idx = np.concatenate([neg_tr, pos_tr])
        val_idx = np.concatenate([neg_val, pos_val])
        rng.shuffle(train_idx)
        rng.shuffle(val_idx)

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        if train_on_val:
            X_train, X_val = X_val, X_train
            y_train, y_val = y_val, y_train

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        clf = LogisticRegression(C=C, max_iter=1000, class_weight="balanced", random_state=random_state)
        clf.fit(X_train_scaled, y_train)

        y_train_proba = clf.predict_proba(X_train_scaled)[:, 1]
        y_val_proba = clf.predict_proba(X_val_scaled)[:, 1]
        best_threshold, threshold_mode = self._select_threshold(y_val, y_val_proba)

        X_scaled = scaler.transform(X)
        y_proba = clf.predict_proba(X_scaled)[:, 1]

        # Reported AUC = HELD-OUT validation AUC (clf never trained on these).
        # The previous code reported roc_auc_score(y, y_proba) over the FULL
        # set including the training rows the classifier was fit on — an
        # in-sample, overfit-optimistic number. Auditing a detector for
        # inflated metrics means our own metric must be honest. Fall back to
        # full-set AUC only if a fold somehow lacks a class (logged).
        if len(np.unique(y_val)) == 2:
            auc = roc_auc_score(y_val, y_val_proba)
            auc_basis = "held_out_val"
        else:
            auc = roc_auc_score(y, y_proba)
            auc_basis = "FULL_SET_FALLBACK_val_single_class"
            print(f"WARNING: val fold single-class; AUC fell back to full set "
                  f"({auc_basis}) — treat as unreliable.")
        print(f"AUC basis: {auc_basis} | val n={len(y_val)} "
              f"(poison={int(np.sum(y_val==1))}, benign={int(np.sum(y_val==0))})")

        self.classifier = clf
        self.scaler = scaler
        self.threshold = best_threshold

        split_manifest = {
            "benign": {
                "train": [paths[i] for i in train_idx if y[i] == 0],
                "val": [paths[i] for i in val_idx if y[i] == 0],
                "test": [],
            },
            "poison": {
                "train": [paths[i] for i in train_idx if y[i] == 1],
                "val": [paths[i] for i in val_idx if y[i] == 1],
                "test": [],
            },
        }

        return {
            "new_weights": clf.coef_.flatten().tolist(),
            "new_threshold": float(best_threshold),
            "auc": float(auc),
            "benign_scores": y_proba[y == 0].tolist(),
            "poison_scores": y_proba[y == 1].tolist(),
            "benign_scores_train": y_train_proba[y_train == 0].tolist(),
            "poison_scores_train": y_train_proba[y_train == 1].tolist(),
            "benign_scores_val": y_val_proba[y_val == 0].tolist(),
            "poison_scores_val": y_val_proba[y_val == 1].tolist(),
            "intercept": self.intercept,
            "threshold_mode": threshold_mode,
            "split_manifest": split_manifest,
            "train_size": int(len(X_train)),
            "val_size": int(len(X_val)),
            "train_counts": {
                "poison": int(np.sum(y_train == 1)),
                "benign": int(np.sum(y_train == 0)),
            },
            "val_counts": {
                "poison": int(np.sum(y_val == 1)),
                "benign": int(np.sum(y_val == 0)),
            },
        }

    def scan(self, adapter_path: str, use_fast_scan: bool = False, layer_idx: int = 20) -> Dict[str, Any]:
        if self.classifier is None or self.scaler is None:
            raise RuntimeError("Detector not calibrated. Run calibrate() first.")

        feat = self._features(Path(adapter_path), layer_idx)
        if feat is None:
            return {"error": "Feature extraction failed", "score": None, "prediction": None}

        feat_scaled = self.scaler.transform(feat.reshape(1, -1))
        proba = float(self.classifier.predict_proba(feat_scaled)[0, 1])
        pred = int(proba >= self.threshold)

        return {
            "score": proba,
            "prediction": pred,
            "threshold": float(self.threshold),
            "features": feat.tolist(),
        }

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "classifier": self.classifier,
                    "scaler": self.scaler,
                    "threshold": self.threshold,
                },
                f,
            )

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.classifier = data["classifier"]
        self.scaler = data["scaler"]
        self.threshold = data["threshold"]

    @staticmethod
    def _extract_features_from_adapter(adapter_path: Path, layer_idx: int) -> Optional[np.ndarray]:
        safetensors_file = adapter_path / "adapter_model.safetensors"
        if not safetensors_file.exists():
            return None

        try:
            weights = st.load_file(str(safetensors_file))
        except Exception:
            return None

        # Projections the detector reads. Default q/k/v/o (Qwen baseline, 20-dim).
        # Override via LBD_DETECTOR_PROJ (comma-separated) for backbones/attacks that
        # train a different set — e.g. C4/CBA trains q_proj,v_proj only, so the Llama-2
        # C4 detector runs on q/v (10-dim). See CHANGELOG 2026-06-22 (Option 1).
        proj_env = os.environ.get("LBD_DETECTOR_PROJ", "").strip()
        proj_names = (
            [p.strip() for p in proj_env.split(",") if p.strip()]
            if proj_env
            else ["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        features = []

        for proj in proj_names:
            prefix = f"base_model.model.model.layers.{layer_idx}.self_attn.{proj}"
            a_key = f"{prefix}.lora_A.weight"
            b_key = f"{prefix}.lora_B.weight"

            if a_key not in weights or b_key not in weights:
                return None

            A = weights[a_key]
            B = weights[b_key]

            if B.shape[1] != A.shape[0]:
                if B.shape[0] == A.shape[0]:
                    B = B.T
                else:
                    return None

            metrics = BackdoorDetector._compute_metrics_from_matrices(B, A)
            features.extend(
                [
                    metrics["sigma1"],
                    metrics["frobenius_norm"],
                    metrics["energy_concentration"],
                    metrics["entropy"],
                    metrics["kurtosis"],
                ]
            )

        return np.array(features, dtype=np.float32)

    @staticmethod
    def _per_layer_block(weights: dict, layer_idx: int, proj_names: list[str]) -> Optional[np.ndarray]:
        """The 5-metric-per-projection block for ONE layer, or None if that layer is
        absent. Same math as the single-layer extractor, factored so the multi-layer
        pooler can call it per layer. (The frozen baseline path does NOT use this.)"""
        block = []
        for proj in proj_names:
            prefix = f"base_model.model.model.layers.{layer_idx}.self_attn.{proj}"
            a_key = f"{prefix}.lora_A.weight"
            b_key = f"{prefix}.lora_B.weight"
            if a_key not in weights or b_key not in weights:
                return None
            A = weights[a_key]
            B = weights[b_key]
            if B.shape[1] != A.shape[0]:
                if B.shape[0] == A.shape[0]:
                    B = B.T
                else:
                    return None
            m = BackdoorDetector._compute_metrics_from_matrices(B, A)
            block.extend([m["sigma1"], m["frobenius_norm"], m["energy_concentration"],
                          m["entropy"], m["kurtosis"]])
        return np.array(block, dtype=np.float32)

    @staticmethod
    def _extract_multilayer_features(adapter_path: Path, layers: list[int], pool: str) -> Optional[np.ndarray]:
        """C5 defense: pool the per-layer spectral block across `layers`.

        A diffuse adapter has LoRA at every layer, so each per-layer block is computed
        and then aggregated. Layers the adapter lacks (e.g. a spiky baseline adapter
        with LoRA only at layer 20) contribute a ZERO block under concat — that absence
        is itself signal (a spiky adapter looks very different from a diffuse one across
        the layer set), and never silently drops the adapter the way the single-layer
        path's `return None` does. A None is returned only if NO requested layer is
        present (nothing to score).
        """
        safetensors_file = adapter_path / "adapter_model.safetensors"
        if not safetensors_file.exists():
            return None
        try:
            weights = st.load_file(str(safetensors_file))
        except Exception:
            return None

        proj_env = os.environ.get("LBD_DETECTOR_PROJ", "").strip()
        proj_names = (
            [p.strip() for p in proj_env.split(",") if p.strip()]
            if proj_env else ["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        block_dim = 5 * len(proj_names)

        blocks = []
        any_present = False
        for L in layers:
            blk = BackdoorDetector._per_layer_block(weights, L, proj_names)
            if blk is None:
                blocks.append(np.zeros(block_dim, dtype=np.float32))
            else:
                blocks.append(blk)
                any_present = True
        if not any_present:
            return None

        stacked = np.vstack(blocks)  # (num_layers, block_dim)
        if pool == "max":
            return stacked.max(axis=0)
        if pool == "mean":
            return stacked.mean(axis=0)
        return stacked.flatten()  # concat: (num_layers * block_dim,)

    @staticmethod
    def _compute_metrics_from_matrices(B: torch.Tensor, A: torch.Tensor) -> Dict[str, float]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        B = B.to(device)
        A = A.to(device)

        Qb, Rb = torch.linalg.qr(B)
        Qa, Ra = torch.linalg.qr(A.T)
        M = Rb @ Ra.T
        s = torch.linalg.svdvals(M)

        sigma1 = s[0].item()
        frob_norm = torch.sqrt(torch.sum(s**2)).item()
        total_energy = torch.sum(s**2).item()
        energy_conc = (sigma1**2) / total_energy if total_energy > 0 else 0.0

        p = s / (torch.sum(s) + 1e-12)
        entropy = -torch.sum(p * torch.log(p + 1e-12)).item()

        delta = B @ A
        flat = delta.flatten().to(torch.float64)
        mean = torch.mean(flat)
        var = torch.var(flat)
        kurt = torch.mean((flat - mean) ** 4) / (var**2 + 1e-12)

        return {
            "sigma1": float(sigma1),
            "frobenius_norm": float(frob_norm),
            "energy_concentration": float(energy_conc),
            "entropy": float(entropy),
            "kurtosis": float(kurt.item()),
        }
