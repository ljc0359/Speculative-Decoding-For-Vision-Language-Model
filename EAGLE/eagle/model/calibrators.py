# -*- coding: utf-8 -*-
"""
Calibrators for EAGLE draft confidence
- Grouped Isotonic (with hierarchical fallback)
- Monotonic MLP (stabilized, hard-label first)
- Platt scaling baseline
"""

import os
import pickle
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
import json

# =========================
# Utilities & base class
# =========================

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class BaseCalibrator(ABC):
    """Common interface."""

    def __init__(self):
        self.is_fitted = False
        self.feature_stats = {}

    @abstractmethod
    def fit(self, features: Dict[str, np.ndarray],
            soft_labels: np.ndarray,
            hard_labels: np.ndarray,
            sample_weights: Optional[np.ndarray] = None):
        pass

    @abstractmethod
    def predict_proba(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        pass

    # ---------- feature prep (固定 5 特征流) ----------
    def _preprocess_features(self, features: Dict[str, np.ndarray], fit_mode: bool = False) -> Dict[str, np.ndarray]:
        processed = {}

        # token_category -> token_type
        token_categories = np.asarray(features['token_category'])
        if fit_mode:
            self.token_category_map = {cat: i for i, cat in enumerate(['content', 'func_punct', 'number'])}
        processed['token_type'] = np.array([self.token_category_map.get(cat, 0) for cat in token_categories])

        # avg_visual_attention_intensity -> 三分位分箱 attn_q
        attn_intensity = np.asarray(features['avg_visual_attention_intensity'])
        if fit_mode:
            self.attn_quantiles = np.quantile(attn_intensity, [0.33, 0.67])
        attn_q = np.zeros_like(attn_intensity, dtype=int)
        attn_q[attn_intensity <= self.attn_quantiles[0]] = 0
        attn_q[(attn_intensity > self.attn_quantiles[0]) & (attn_intensity <= self.attn_quantiles[1])] = 1
        attn_q[attn_intensity > self.attn_quantiles[1]] = 2
        processed['attn_q'] = attn_q

        # tree_depth -> pos_bin
        depth = np.asarray(features['tree_depth'])
        processed['pos_bin'] = (depth > 2).astype(int)  # 0: depth<=2, 1: depth>2
        processed['tree_depth'] = depth

        # keep original
        processed['avg_visual_attention_intensity'] = attn_intensity
        if 'draft_margin' in features:
            processed['draft_margin'] = np.asarray(features['draft_margin'])

        # draft_confidence
        processed['draft_conf'] = np.asarray(features['draft_confidence'])

        return processed

    def _create_group_key(self, token_type: int, attn_q: int, pos_bin: int) -> str:
        return f"t{token_type}_a{attn_q}_p{pos_bin}"

    # ---------- metrics ----------
    def _ece(self, pred_probs: np.ndarray, true_labels: np.ndarray,
             sample_weights: Optional[np.ndarray] = None,
             n_bins: int = 20, equal_freq: bool = True) -> float:
        if equal_freq:
            qs = np.linspace(0, 1, n_bins + 1)
            boundaries = np.quantile(pred_probs, qs)
            boundaries = np.unique(boundaries)
            if len(boundaries) < 2:
                return 0.0
            lowers, uppers = boundaries[:-1], boundaries[1:]
        else:
            bounds = np.linspace(0, 1, n_bins + 1)
            lowers, uppers = bounds[:-1], bounds[1:]

        ece, total_w = 0.0, 0.0
        for lo, up in zip(lowers, uppers):
            in_bin = (pred_probs > lo) & (pred_probs <= up)
            if in_bin.sum() == 0:
                continue
            if sample_weights is not None:
                w = sample_weights[in_bin]
                acc = float(np.average(true_labels[in_bin], weights=w))
                conf = float(np.average(pred_probs[in_bin], weights=w))
                bw = float(w.sum())
            else:
                acc = float(true_labels[in_bin].mean())
                conf = float(pred_probs[in_bin].mean())
                bw = float(in_bin.sum())
            ece += bw * abs(conf - acc)
            total_w += bw
        return ece / total_w if total_w > 0 else 0.0

    @staticmethod
    def _compute_ece(pred_probs: np.ndarray, true_labels: np.ndarray,
                     sample_weights: Optional[np.ndarray] = None,
                     n_bins: int = 20, equal_freq: bool = True) -> float:
        # 与实例方法 _ece 等价，但作为静态方法提供类级调用
        p = np.asarray(pred_probs, dtype=float)
        y = np.asarray(true_labels, dtype=float)
        w = np.asarray(sample_weights, dtype=float) if sample_weights is not None else None

        if equal_freq:
            qs = np.linspace(0, 1, n_bins + 1)
            boundaries = np.quantile(p, qs)
            boundaries = np.unique(boundaries)
            if len(boundaries) < 2:
                return 0.0
            lowers, uppers = boundaries[:-1], boundaries[1:]
        else:
            bounds = np.linspace(0, 1, n_bins + 1)
            lowers, uppers = bounds[:-1], bounds[1:]

        ece, total_w = 0.0, 0.0
        for lo, up in zip(lowers, uppers):
            in_bin = (p > lo) & (p <= up)
            if in_bin.sum() == 0:
                continue
            if w is not None:
                wbin = w[in_bin]
                acc = float(np.average(y[in_bin], weights=wbin))
                conf = float(np.average(p[in_bin], weights=wbin))
                bw = float(wbin.sum())
            else:
                acc = float(y[in_bin].mean())
                conf = float(p[in_bin].mean())
                bw = float(in_bin.sum())
            ece += bw * abs(conf - acc)
            total_w += bw
        return ece / total_w if total_w > 0 else 0.0

    def evaluate(self, features: Dict[str, np.ndarray],
                 soft_labels: np.ndarray,
                 hard_labels: np.ndarray,
                 sample_weights: Optional[np.ndarray] = None) -> Dict[str, float]:
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before evaluation")

        p = self.predict_proba(features)
        out = {}
        out['brier'] = brier_score_loss(hard_labels, p)
        out['ece_eqfreq20'] = self._ece(p, hard_labels, sample_weights, n_bins=20, equal_freq=True)
        out['ece_fixed10'] = self._ece(p, hard_labels, sample_weights, n_bins=10, equal_freq=False)
        out['soft_mse'] = float(np.mean((p - soft_labels) ** 2))
        try:
            out['auroc'] = roc_auc_score(hard_labels, p, sample_weight=sample_weights)
        except Exception:
            out['auroc'] = 0.5
        return out

    # ---------- I/O ----------
    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)


# =========================
# Platt scaling (baseline)
# =========================

class PlattCalibrator(BaseCalibrator):
    """Sigmoid(ax+b) on logit(conf). Target=hard by default."""

    def __init__(self, target: str = 'hard', class_weight: Optional[str] = 'balanced'):
        super().__init__()
        self.target = target
        self.clf = None
        self.c_minmax = None
        self.class_weight = class_weight

    def fit(self, features, soft_labels, hard_labels, sample_weights=None):
        proc = self._preprocess_features(features, fit_mode=True)
        c = np.clip(proc['draft_conf'], 1e-6, 1 - 1e-6)
        z = np.log(c) - np.log(1 - c)
        y = hard_labels if self.target == 'hard' else soft_labels
        if self.target == 'hard':
            self.clf = LogisticRegression(class_weight=self.class_weight,
                                          solver='liblinear', max_iter=1000)
            self.clf.fit(z.reshape(-1, 1), y.astype(int), sample_weight=sample_weights)
        else:
            # 对 soft 目标，用平方误差回归近似（也可用 sklearn 的 SGDRegressor）
            from sklearn.linear_model import Ridge
            self.clf = Ridge(alpha=1.0)
            self.clf.fit(z.reshape(-1, 1), y, sample_weight=sample_weights)
        self.is_fitted = True

    def predict_proba(self, features):
        proc = self._preprocess_features(features, fit_mode=False)
        c = np.clip(proc['draft_conf'], 1e-6, 1 - 1e-6)
        z = np.log(c) - np.log(1 - c)
        if hasattr(self.clf, 'predict_proba'):
            p = self.clf.predict_proba(z.reshape(-1, 1))[:, 1]
        else:
            # regression 输出转 sigmoid
            logits = self.clf.predict(z.reshape(-1, 1))
            p = 1 / (1 + np.exp(-logits))
        return np.clip(p, 1e-4, 1 - 1e-4)


# =========================
# Grouped Isotonic
# =========================

class GroupedIsotonicCalibrator(BaseCalibrator):
    """12 组 Isotonic：token_type × attn_q × pos_bin，层级回退"""

    def __init__(self, min_samples_per_group: int = 200,
                 out_of_bounds: str = 'clip',
                 target: str = 'hard'):
        super().__init__()
        self.min_samples_per_group = min_samples_per_group
        self.out_of_bounds = out_of_bounds
        self.target = target  # 'hard' or 'soft'
        self.level1, self.level2, self.level3 = {}, {}, {}
        self.global_calibrator = None
        self.global_mean = None

    def _fit_iso_binned(self, x, y, w=None, n_bins: int = 20):
        s = np.argsort(x)
        xs, ys = x[s], y[s]
        ws = w[s] if w is not None else None
        # 等频分桶
        edges = np.linspace(0, len(xs), n_bins + 1).astype(int)
        xb, yb, wb = [], [], []
        for i in range(n_bins):
            a, b = edges[i], edges[i + 1]
            if b <= a:
                continue
            if ws is not None:
                w_i = ws[a:b]
                xb.append(float(xs[a:b].mean()))
                yb.append(float(np.average(ys[a:b], weights=w_i)))
                wb.append(float(w_i.sum()))
            else:
                xb.append(float(xs[a:b].mean()))
                yb.append(float(ys[a:b].mean()))
                wb.append(float(b - a))
        iso = IsotonicRegression(out_of_bounds=self.out_of_bounds, increasing=True)
        iso.fit(np.array(xb), np.array(yb), sample_weight=np.array(wb))
        return iso

    def fit(self, features, soft_labels, hard_labels, sample_weights=None):
        proc = self._preprocess_features(features, fit_mode=True)
        c = proc['draft_conf']
        token, attn, pos = proc['token_type'], proc['attn_q'], proc['pos_bin']

        y = hard_labels if self.target == 'hard' else soft_labels
        w = sample_weights

        # ========= 训练数据统计与诊断输出 =========
        try:
            def _safe_stats(arr, percentiles=(95, 99)):
                if arr.size == 0:
                    return {"count": 0}
                stats = {
                    "count": int(arr.size),
                    "max": float(np.max(arr)),
                    "mean": float(np.mean(arr)),
                }
                for pctl in percentiles:
                    stats[f"p{pctl}"] = float(np.percentile(arr, pctl))
                return stats

            # 全局统计
            soft_stats = _safe_stats(np.asarray(soft_labels), percentiles=(95, 99))
            hard_pos_rate = float(np.mean(hard_labels)) if len(hard_labels) > 0 else float('nan')
            conf_stats = _safe_stats(np.asarray(c), percentiles=(95,))
            print("[GroupedIsotonicCalibrator] 训练数据统计 - 全局")
            print(f"  soft_labels: count={soft_stats.get('count', 0)}, "
                  f"max={soft_stats.get('max', float('nan')):.4f}, "
                  f"p95={soft_stats.get('p95', float('nan')):.4f}, "
                  f"p99={soft_stats.get('p99', float('nan')):.4f}, "
                  f"mean={soft_stats.get('mean', float('nan')):.4f}")
            print(f"  hard_labels 正例率: {hard_pos_rate:.4f}")
            print(f"  draft_conf: count={conf_stats.get('count', 0)}, "
                  f"max={conf_stats.get('max', float('nan')):.4f}, "
                  f"p95={conf_stats.get('p95', float('nan')):.4f}")

            if soft_stats.get('max', 1.0) <= 0.47:
                print("  警告：soft_labels 的最大值较低（≤0.47），可能限制 Isotonic 回归的上限；请检查软标签的标定范围。")

            # L1: token_type
            print("[GroupedIsotonicCalibrator] 分组统计 - L1(token_type)")
            for t in range(3):
                idx = (token == t)
                cnt = int(idx.sum())
                if cnt == 0:
                    print(f"  t{t}: count=0")
                    continue
                soft_s = _safe_stats(np.asarray(soft_labels)[idx], percentiles=(95, 99))
                hard_rate = float(np.mean(np.asarray(hard_labels)[idx]))
                conf_s = _safe_stats(np.asarray(c)[idx], percentiles=(95,))
                print(f"  t{t}: count={cnt}, "
                      f"soft[max={soft_s.get('max'):.4f}, p95={soft_s.get('p95'):.4f}, p99={soft_s.get('p99'):.4f}, mean={soft_s.get('mean'):.4f}], "
                      f"hard_pos={hard_rate:.4f}, "
                      f"conf[max={conf_s.get('max'):.4f}, p95={conf_s.get('p95'):.4f}]")

            # L2: token_type × attn_q
            print("[GroupedIsotonicCalibrator] 分组统计 - L2(token_type × attn_q)")
            for t in range(3):
                for a in range(3):
                    idx = (token == t) & (attn == a)
                    cnt = int(idx.sum())
                    key = f"t{t}_a{a}"
                    if cnt == 0:
                        print(f"  {key}: count=0")
                        continue
                    soft_s = _safe_stats(np.asarray(soft_labels)[idx], percentiles=(95, 99))
                    hard_rate = float(np.mean(np.asarray(hard_labels)[idx]))
                    conf_s = _safe_stats(np.asarray(c)[idx], percentiles=(95,))
                    print(f"  {key}: count={cnt}, "
                          f"soft[max={soft_s.get('max'):.4f}, p95={soft_s.get('p95'):.4f}, p99={soft_s.get('p99'):.4f}, mean={soft_s.get('mean'):.4f}], "
                          f"hard_pos={hard_rate:.4f}, "
                          f"conf[max={conf_s.get('max'):.4f}, p95={conf_s.get('p95'):.4f}]")

            # L3: token_type × attn_q × pos_bin
            print("[GroupedIsotonicCalibrator] 分组统计 - L3(token_type × attn_q × pos_bin)")
            for t in range(3):
                for a in range(3):
                    for p in range(2):
                        key3 = self._create_group_key(t, a, p)
                        idx = (token == t) & (attn == a) & (pos == p)
                        cnt = int(idx.sum())
                        if cnt == 0:
                            print(f"  {key3}: count=0")
                            continue
                        soft_s = _safe_stats(np.asarray(soft_labels)[idx], percentiles=(95, 99))
                        hard_rate = float(np.mean(np.asarray(hard_labels)[idx]))
                        conf_s = _safe_stats(np.asarray(c)[idx], percentiles=(95,))
                        print(f"  {key3}: count={cnt}, "
                              f"soft[max={soft_s.get('max'):.4f}, p95={soft_s.get('p95'):.4f}, p99={soft_s.get('p99'):.4f}, mean={soft_s.get('mean'):.4f}], "
                              f"hard_pos={hard_rate:.4f}, "
                              f"conf[max={conf_s.get('max'):.4f}, p95={conf_s.get('p95'):.4f}]")
        except Exception as e:
            print(f"[GroupedIsotonicCalibrator] 统计打印时出现异常：{e}")

        # 全局回退
        self.global_calibrator = self._fit_iso_binned(c, y, w, n_bins=20)
        self.global_mean = float(np.average(y, weights=w) if w is not None else np.mean(y))

        # L1
        self.level1 = {}
        for t in range(3):
            idx = (token == t)
            if idx.sum() >= self.min_samples_per_group:
                self.level1[f"t{t}"] = self._fit_iso_binned(c[idx], y[idx], w[idx] if w is not None else None)
            else:
                self.level1[f"t{t}"] = None

        # L2
        self.level2 = {}
        for t in range(3):
            for a in range(3):
                idx = (token == t) & (attn == a)
                key = f"t{t}_a{a}"
                if idx.sum() >= self.min_samples_per_group:
                    self.level2[key] = self._fit_iso_binned(c[idx], y[idx], w[idx] if w is not None else None)
                else:
                    self.level2[key] = None

        # L3
        self.level3 = {}
        for t in range(3):
            for a in range(3):
                for p in range(2):
                    key = self._create_group_key(t, a, p)
                    idx = (token == t) & (attn == a) & (pos == p)
                    if idx.sum() >= self.min_samples_per_group:
                        self.level3[key] = self._fit_iso_binned(c[idx], y[idx], w[idx] if w is not None else None)
                    else:
                        self.level3[key] = None

        self.is_fitted = True

    def predict_proba(self, features):
        proc = self._preprocess_features(features, fit_mode=False)
        c = proc['draft_conf']
        token, attn, pos = proc['token_type'], proc['attn_q'], proc['pos_bin']

        out = np.zeros_like(c, dtype=np.float32)
        for t in range(3):
            for a in range(3):
                for p in range(2):
                    mask = (token == t) & (attn == a) & (pos == p)
                    if mask.sum() == 0:
                        continue
                    key3 = self._create_group_key(t, a, p)
                    key2 = f"t{t}_a{a}"
                    key1 = f"t{t}"
                    cal = self.level3.get(key3) or self.level2.get(key2) or self.level1.get(key1) or self.global_calibrator
                    if cal is not None:
                        out[mask] = cal.predict(c[mask])
                    else:
                        out[mask] = self.global_mean
        return np.clip(out, 1e-4, 1 - 1e-4)


# =========================
# Monotonic Network
# =========================

class MonotonicMLP(nn.Module):
    """logit(p̂) = β(φ) + Σ_k γ_k(φ)·s_k(logit(c)), 其中 γ_k(φ)≥0 通过 softplus 保证"""
    def __init__(self, phi_dim: int, n_basis: int, hidden_dim: int):
        super().__init__()
        self.beta_net = nn.Sequential(
            nn.Linear(phi_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.gamma_net = nn.Sequential(
            nn.Linear(phi_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_basis)
        )

    def forward(self, phi: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        beta  = self.beta_net(phi).squeeze(-1)
        gamma = F.softplus(self.gamma_net(phi))     # ≥ 0，保证对 c 的单调
        mono  = (gamma * basis).sum(dim=1)
        return beta + mono

# ----------------- calibrator -----------------
class MonotonicNetworkCalibrator(BaseCalibrator):
    """
    稳定版单调网络（修复版）
    - 主目标 BCE(hard) + 少量 soft MSE(soft)（alpha_soft=0.05）
    - 等频重加权（覆盖分布尾部）；pos_weight 默认关闭，可选开启（限幅 [1, 8]）
    - 非均匀 knots（左密右疏）在 logit(c) 上（训练/预测复用）
    - 新增：预测均值先验（lambda_mean=1.0），把 p̄ 拉回真实正例率
    """

    def __init__(self,
                 hidden_dim: int = 16,
                 n_basis: int = 4,
                 learning_rate: float = 3e-4,
                 max_epochs: int = 300,
                 patience: int = 20,
                 alpha_soft: float = 0.05,
                 n_reweight_bins: int = 20,
                 device: Optional[str] = None,
                 weight_decay: float = 5e-4,
                 grad_clip: float = 1.0,
                 use_pos_weight: bool = False,      # 新：默认不用 pos_weight
                 lambda_mean: float = 1.0):         # 新：均值先验强度
        super().__init__()  # 继承 BaseCalibrator，以复用 _preprocess_features 和公共属性
        set_seed(42)
        self.hidden_dim = hidden_dim
        self.n_basis = n_basis
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.patience = patience
        self.alpha_soft = alpha_soft
        self.n_reweight_bins = n_reweight_bins
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.use_pos_weight = use_pos_weight
        self.lambda_mean = lambda_mean

        # 保存的统计与 knot
        self.knots_z = None
        self.saved_stats = {}
        self.use_tree_depth = True
        self.model = None
        self.is_fitted = False
        self._train_conf_snapshot = None  # 仅用于日志诊断

    # ===== feature preprocessing（外部应复用你现有的 BaseCalibrator._preprocess_features）=====
    # 这里假定调用方已做了 _preprocess_features，传入 proc 字典字段：
    # ['token_type','attn_q','pos_bin','tree_depth'或'tree_position','avg_visual_attention_intensity','draft_margin'(可选),'draft_conf']

    # ---------- φ ----------
    def _build_phi(self, proc: Dict[str, np.ndarray], fit_mode: bool) -> np.ndarray:
        token = np.eye(3)[proc['token_type']]
        attn  = np.eye(3)[proc['attn_q']]
        pos   = np.eye(2)[proc['pos_bin']]

        # tree depth/position 归一化
        if 'tree_depth' in proc:
            td = proc['tree_depth']
            if fit_mode:
                self.saved_stats['tree_min'], self.saved_stats['tree_max'] = float(td.min()), float(td.max())
                self.use_tree_depth = True
            tmin, tmax = self.saved_stats['tree_min'], self.saved_stats['tree_max']
            tree = (td - tmin) / (tmax - tmin + 1e-8)
        else:
            tp = proc['tree_position']
            if fit_mode:
                self.saved_stats['tree_min'], self.saved_stats['tree_max'] = float(tp.min()), float(tp.max())
                self.use_tree_depth = False
            tmin, tmax = self.saved_stats['tree_min'], self.saved_stats['tree_max']
            tree = (tp - tmin) / (tmax - tmin + 1e-8)

        # 视觉注意力归一化
        ai = proc['avg_visual_attention_intensity']
        if fit_mode:
            self.saved_stats['attn_min'], self.saved_stats['attn_max'] = float(ai.min()), float(ai.max())
        amin, amax = self.saved_stats['attn_min'], self.saved_stats['attn_max']
        attn_norm = (ai - amin) / (amax - amin + 1e-8)

        parts = [token, attn, pos, tree.reshape(-1, 1), attn_norm.reshape(-1, 1)]

        # 可选 margin
        if 'draft_margin' in proc:
            m = proc['draft_margin']
            if fit_mode:
                self.saved_stats['margin_min'], self.saved_stats['margin_max'] = float(m.min()), float(m.max())
            mmin, mmax = self.saved_stats['margin_min'], self.saved_stats['margin_max']
            parts.append(((m - mmin) / (mmax - mmin + 1e-8)).reshape(-1, 1))

        phi = np.concatenate(parts, axis=1).astype('float32')
        if fit_mode:
            self.phi_dim = phi.shape[1]
        return phi

    # ---------- basis ----------
    def _init_knots(self, conf: np.ndarray):
        c = np.clip(conf, 1e-6, 1 - 1e-6)
        # 左密右疏（幂指数量化），再做 logit
        qs = np.linspace(0.05, 0.95, self.n_basis) ** 1.7
        c_knots = np.quantile(c, qs)
        z_knots = np.log(c_knots) - np.log(1 - c_knots)
        self.knots_z = z_knots.astype('float32')

    def _basis(self, conf: np.ndarray) -> np.ndarray:
        assert self.knots_z is not None
        c = np.clip(conf, 1e-6, 1 - 1e-6)
        z = np.log(c) - np.log(1 - c)
        return np.maximum(0.0, z[:, None] - self.knots_z[None, :]).astype('float32')

    # ---------- weighting ----------
    def _equal_freq_weights(self, conf: np.ndarray, n_bins: int = 20) -> np.ndarray:
        c = np.asarray(conf, dtype=np.float64)
        edges = np.quantile(c, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
        bins = np.digitize(c, edges[1:-1], right=True)
        counts = np.bincount(bins, minlength=len(edges) - 1).astype('float64')
        counts[counts == 0] = 1.0
        w = 1.0 / counts[bins]
        return (w * (len(w) / w.sum())).astype('float32')

    # ---------- fit/predict ----------
    def fit(self, features: Dict[str, np.ndarray], soft_labels: np.ndarray,
            hard_labels: np.ndarray, sample_weights: Optional[np.ndarray] = None):
        # 统一在此进行特征预处理，生成 token_type/attn_q/pos_bin/draft_conf 等键
        proc = self._preprocess_features(features, fit_mode=True)
        phi = self._build_phi(proc, fit_mode=True)
        conf = proc['draft_conf']
        self._init_knots(conf)
        basis = self._basis(conf)

        # 样本权重
        if sample_weights is None:
            weights = self._equal_freq_weights(conf, self.n_reweight_bins)
        else:
            weights = np.asarray(sample_weights, dtype='float32')

        # 不平衡处理：真实正例率
        pos_rate = float(np.mean(hard_labels))
        if self.use_pos_weight:
            pos_weight = (1.0 - pos_rate) / max(pos_rate, 1e-6)
            pos_weight = float(np.clip(pos_weight, 1.0, 8.0))  # 限幅
        else:
            pos_weight = None

        # tensors
        device = self.device
        phi_t   = torch.tensor(phi, dtype=torch.float32, device=device)
        basis_t = torch.tensor(basis, dtype=torch.float32, device=device)
        yh_t    = torch.tensor(hard_labels, dtype=torch.float32, device=device)
        ys_t    = torch.tensor(soft_labels, dtype=torch.float32, device=device)
        w_t     = torch.tensor(weights, dtype=torch.float32, device=device)

        # 模型 + 偏置初始化到 logit(pos_rate)
        self.model = MonotonicMLP(self.phi_dim, self.n_basis, self.hidden_dim).to(device)
        with torch.no_grad():
            b0 = float(np.log(pos_rate + 1e-8) - np.log(1 - pos_rate + 1e-8))
            nn.init.zeros_(self.model.beta_net[-1].weight)
            nn.init.constant_(self.model.beta_net[-1].bias, b0)

        print(f"[MonoCal] init: pos_rate={pos_rate:.4f}, "
              f"use_pos_weight={bool(pos_weight)}, "
              f"pos_weight={0 if pos_weight is None else pos_weight:.3f}, "
              f"lambda_mean={self.lambda_mean:.2f}, "
              f"alpha_soft={self.alpha_soft:.3f}")

        # 训练
        self._train(phi_t, basis_t, yh_t, ys_t, w_t, pos_rate, pos_weight)
        self.is_fitted = True
        self._train_conf_snapshot = conf.copy()  # 仅用于日志诊断

    def _train(self, phi, basis, y_hard, y_soft, weights, pos_rate: float, pos_weight: Optional[float]):
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        n = phi.size(0)
        idx = torch.randperm(n, device=phi.device)
        split = max(int(0.8 * n), 1)
        tr, va = idx[:split], idx[split:]

        best, patience, best_sd = float('inf'), self.patience, None

        for ep in range(self.max_epochs):
            self.model.train(); opt.zero_grad()

            logit_tr = self.model(phi[tr], basis[tr])
            # BCE：先算逐样本 loss，再叠加样本权重（等频权重）
            bce_element = F.binary_cross_entropy_with_logits(logit_tr, y_hard[tr], reduction='none')
            if pos_weight is not None:
                # 正例额外乘 pos_weight
                bce_element = torch.where(y_hard[tr] > 0.5, bce_element * pos_weight, bce_element)
            loss_bce = (bce_element * weights[tr]).mean()

            prob_tr = torch.sigmoid(logit_tr)
            # soft MSE（样本权重）
            mse_element = F.mse_loss(prob_tr, y_soft[tr], reduction='none')
            loss_mse = (mse_element * weights[tr]).mean()

            # 均值先验：把预测均值拉向 pos_rate
            mean_prior = (prob_tr.mean() - pos_rate) ** 2

            loss = (1 - self.alpha_soft) * loss_bce + self.alpha_soft * loss_mse + self.lambda_mean * mean_prior
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            opt.step()

            # 验证
            self.model.eval()
            with torch.no_grad():
                logit_va = self.model(phi[va], basis[va])
                prob_va = torch.sigmoid(logit_va)

                bce_v = F.binary_cross_entropy_with_logits(logit_va, y_hard[va], reduction='none')
                if pos_weight is not None:
                    bce_v = torch.where(y_hard[va] > 0.5, bce_v * pos_weight, bce_v)
                bce_v = (bce_v * weights[va]).mean()

                mse_v = (F.mse_loss(prob_va, y_soft[va], reduction='none') * weights[va]).mean()
                mean_prior_v = (prob_va.mean() - pos_rate) ** 2
                val = (1 - self.alpha_soft) * bce_v + self.alpha_soft * mse_v + self.lambda_mean * mean_prior_v

            if (ep % 20) == 0:
                p_tr_np = prob_tr.detach().cpu().numpy()
                # 与 draft_conf 的相关性（诊断）——训练集上
                # 为避免额外传参，这里只在 ep%20==0 时跳过相关性计算
                print(f"[MonoCal] Epoch {ep:03d}  train={float(loss):.4f}  val={float(val):.4f}  "
                      f"p_tr[min/mean/max]={p_tr_np.min():.3f}/{p_tr_np.mean():.3f}/{p_tr_np.max():.3f}  "
                      f"mean_prior={float(mean_prior):.5f}")

            # 早停
            if float(val) + 1e-6 < best:
                best = float(val); patience = self.patience
                best_sd = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience -= 1
                if patience <= 0:
                    break

        if best_sd is not None:
            self.model.load_state_dict(best_sd)

        # 最终训练集诊断
        self.model.eval()
        with torch.no_grad():
            p_all = torch.sigmoid(self.model(phi, basis)).detach().cpu().numpy()
        pred_mean = float(p_all.mean())
        print(f"[MonoCal] Final train pred mean={pred_mean:.4f} (target pos_rate={pos_rate:.4f}), "
              f"Δ={pred_mean - pos_rate:+.4f}")

    def predict_proba(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Call fit() first.")
        # 预测阶段也进行相同的预处理（fit_mode=False）
        proc = self._preprocess_features(features, fit_mode=False)
        phi = self._build_phi(proc, fit_mode=False)
        basis = self._basis(proc['draft_conf'])
        self.model.eval()
        with torch.no_grad():
            logit = self.model(torch.tensor(phi, dtype=torch.float32, device=self.device),
                               torch.tensor(basis, dtype=torch.float32, device=self.device))
            p = torch.sigmoid(logit).clamp(1e-4, 1 - 1e-4).cpu().numpy()
        return p.astype('float32')
# =========================
# Data loader + quick tests
# =========================

def load_calibration_data(path: str) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    if path.endswith('.json'):
        raw = json.load(open(path, 'r'))
        data = raw['candidate_calibration_data'] if isinstance(raw, dict) and 'candidate_calibration_data' in raw else raw
        feats = {}
        keys = ['draft_confidence', 'tree_depth', 'avg_visual_attention_intensity', 'draft_margin',
                'token_category']
        for k in keys:
            if k in data[0]:
                feats[k] = np.array([x[k] for x in data])
        soft = np.array([x['base_confidence'] for x in data])
        hard = np.array([x.get('base_top1_token', x.get('hard_label', 0)) for x in data]).astype(int)
        return feats, soft, hard
    elif path.endswith('.npz'):
        d = np.load(path)
        feats = {k: d[k] for k in ['draft_confidence', 'tree_depth', 'avg_visual_attention_intensity',
                                   'draft_margin', 'token_category'] if k in d}
        soft = d['base_confidence'] if 'base_confidence' in d else d['soft_labels']
        hard = d['base_top1_token'] if 'base_top1_token' in d else d['hard_labels']
        return feats, soft, hard
    else:
        raise ValueError("Unsupported file format")


# =========================
# Simple offline test
# =========================

def _print_metrics(name: str, metrics: Dict[str, float]):
    print(f"\n{name}:")
    for k in ['brier', 'ece_eqfreq20', 'ece_fixed10', 'soft_mse', 'auroc']:
        if k in metrics:
            print(f"  {k}: {metrics[k]:.6f}")


def calibrator_training(calibrator_dir: str, json_path: str):
    """
    训练 Grouped Isotonic 校准器并保存到指定目录。
    参数:
        calibrator_dir: 输出保存目录（若不存在会创建）
        json_path: 训练数据 JSON 文件路径（需包含 features/soft_labels/hard_labels）
    返回:
        训练好的 GroupedIsotonicCalibrator 实例
    """
    set_seed(42)
    print("=" * 60)
    print(f"[Calibrator] Training Grouped Isotonic from JSON: {json_path}")
    print("=" * 60)

    # 加载数据
    feats, soft, hard = load_calibration_data(json_path)
    n = len(soft)
    print(f"[Calibrator] Loaded {n} samples; pos_rate={hard.mean():.4f}")
    print(f"[Calibrator] Feature keys: {list(feats.keys())}")

    # 训练 Isotonic（硬标签）
    gi = GroupedIsotonicCalibrator(min_samples_per_group=200, target='hard')
    gi.fit(feats, soft, hard)

    # 评估并保存
    metrics = gi.evaluate(feats, soft, hard)
    os.makedirs(calibrator_dir, exist_ok=True)
    model_path = os.path.join(calibrator_dir, "grouped_isotonic_calibrator.pkl")
    metrics_path = os.path.join(calibrator_dir, "grouped_isotonic_metrics.json")
    gi.save(model_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[Calibrator] Saved Isotonic calibrator to: {model_path}")
    print(f"[Calibrator] Saved training metrics to: {metrics_path}")
    return gi

def test_calibrator_training():
    set_seed(42)
    data_path = "/root/Speculative_decoding/calibration_data/chartqa_train_40_val_0_test_60_total_100/non_calibrator/training_calibration_data.json"
    print("=" * 60)
    print("Testing Calibrator Training Pipeline (JSON)")
    print("=" * 60)
    feats, soft, hard = load_calibration_data(data_path)
    n = len(soft)
    print(f"Loaded {n} samples; pos_rate={hard.mean():.4f}")
    print(f"Feature keys: {list(feats.keys())}")

    # baseline
    base_pred = np.clip(feats['draft_confidence'], 1e-6, 1 - 1e-6)
    base_metrics = {
        'brier': brier_score_loss(hard, base_pred),
        'ece_eqfreq20': BaseCalibrator._compute_ece(base_pred, hard, None, 20, True),
        'soft_mse': float(np.mean((base_pred - soft) ** 2)),
        'auroc': roc_auc_score(hard, base_pred)
    }
    _print_metrics("BASELINE (draft_conf)", base_metrics)

    # Platt
    pl = PlattCalibrator(target='hard')
    pl.fit(feats, soft, hard)
    _print_metrics("PLATT (hard)", pl.evaluate(feats, soft, hard))

    # Grouped Isotonic（硬标签）
    gi = GroupedIsotonicCalibrator(min_samples_per_group=200, target='hard')
    gi.fit(feats, soft, hard)
    _print_metrics("Grouped Isotonic (hard)", gi.evaluate(feats, soft, hard))

    # Monotonic（硬标签主导）
    mono = MonotonicNetworkCalibrator(hidden_dim=32, n_basis=6, alpha_soft=0.0,
                                      max_epochs=200, patience=20, device='cpu')
    mono.fit(feats, soft, hard)
    _print_metrics("Monotonic MLP (hard-dominant)", mono.evaluate(feats, soft, hard))

    print("\n✓ Calibrator pipeline finished.")


def compare_before_after_calibration():
    set_seed(42)
    data_path = "/root/Speculative_decoding/calibration_data/chartqa_train_40_val_0_test_60_total_100/non_calibrator/training_calibration_data.json"
    feats, soft, hard = load_calibration_data(data_path)

    base_pred = np.clip(feats['draft_confidence'], 1e-6, 1 - 1e-6)
    print("\nBaseline Brier:", brier_score_loss(hard, base_pred))

    pl = PlattCalibrator(target='hard'); pl.fit(feats, soft, hard)
    gi = GroupedIsotonicCalibrator(min_samples_per_group=200, target='hard'); gi.fit(feats, soft, hard)
    mono = MonotonicNetworkCalibrator(hidden_dim=32, n_basis=6, alpha_soft=0.0,
                                      max_epochs=200, patience=20, device='cpu'); mono.fit(feats, soft, hard)

    for name, cal in [("Platt", pl), ("Isotonic", gi), ("Monotonic", mono)]:
        m = cal.evaluate(feats, soft, hard)
        _print_metrics(name, m)


if __name__ == "__main__":
    # test_calibrator_training()
    # compare_before_after_calibration()
    calibrator_path = "/root/Speculative_decoding/calibration_data/test_calibrators"
    json_path = "/root/Speculative_decoding/calibration_data/chartqa_calib_isotonic_train_1000_val_0_test_1500_total_2500/training_calibration_data.json"
    calibrator_training(calibrator_path, json_path)
