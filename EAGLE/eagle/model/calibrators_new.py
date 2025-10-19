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

        # avg_visual_attention_intensity -> 五分位分箱 attn_q (0..4)
        attn_intensity = np.asarray(features['avg_visual_attention_intensity'])
        if fit_mode:
            # 5 组分位点
            self.attn_quantiles = np.quantile(attn_intensity, [0.2, 0.4, 0.6, 0.8])
        attn_q = np.zeros_like(attn_intensity, dtype=int)
        # 根据四个分位点划分为五段
        attn_q[attn_intensity <= self.attn_quantiles[0]] = 0
        attn_q[(attn_intensity > self.attn_quantiles[0]) & (attn_intensity <= self.attn_quantiles[1])] = 1
        attn_q[(attn_intensity > self.attn_quantiles[1]) & (attn_intensity <= self.attn_quantiles[2])] = 2
        attn_q[(attn_intensity > self.attn_quantiles[2]) & (attn_intensity <= self.attn_quantiles[3])] = 3
        attn_q[attn_intensity > self.attn_quantiles[3]] = 4
        processed['attn_q'] = attn_q

        # tree_depth -> pos_bin
        depth = np.asarray(features['tree_depth'])
        processed['pos_bin'] = (depth > 2).astype(int)  # 0: depth<=2, 1: depth>2
        processed['tree_depth'] = depth

        # keep original
        processed['avg_visual_attention_intensity'] = attn_intensity

        # draft_margin -> 三分位分箱 margin_q (0..2)
        if 'draft_margin' in features:
            margin = np.asarray(features['draft_margin'])
            if fit_mode:
                # 3 组分位点
                self.margin_quantiles = np.quantile(margin, [0.33, 0.67])
            else:
                # 如果在预测阶段 self.margin_quantiles 未定义（极端情况），做鲁棒默认
                if not hasattr(self, 'margin_quantiles') or self.margin_quantiles is None:
                    # 使用当前数据的分位点以避免崩溃
                    self.margin_quantiles = np.quantile(margin, [0.33, 0.67])
            margin_q = np.zeros_like(margin, dtype=int)
            margin_q[margin <= self.margin_quantiles[0]] = 0
            margin_q[(margin > self.margin_quantiles[0]) & (margin <= self.margin_quantiles[1])] = 1
            margin_q[margin > self.margin_quantiles[1]] = 2
            processed['margin_q'] = margin_q
            processed['draft_margin'] = margin
        else:
            # 若无 draft_margin，则默认单一组（0），以便与现有流程兼容
            processed['margin_q'] = np.zeros_like(attn_intensity, dtype=int)

        # draft_confidence
        processed['draft_conf'] = np.asarray(features['draft_confidence'])

        return processed

    def _create_group_key(self, token_type: int, attn_q: int, pos_bin: int) -> str:
        return f"t{token_type}_a{attn_q}_p{pos_bin}"

    def _create_group_key4(self, token_type: int, attn_q: int, pos_bin: int, margin_q: int) -> str:
        # 新增的四维键：token_type × attn_q × pos_bin × margin_q
        return f"t{token_type}_a{attn_q}_p{pos_bin}_m{margin_q}"

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
        # 新增 margin_q 维度
        margin_q = proc['margin_q']

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

            # L2: token_type × attn_q (5 组)
            print("[GroupedIsotonicCalibrator] 分组统计 - L2(token_type × attn_q)")
            for t in range(3):
                for a in range(5):
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
                for a in range(5):
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

            # L4: token_type × attn_q × pos_bin × margin_q
            print("[GroupedIsotonicCalibrator] 分组统计 - L4(token_type × attn_q × pos_bin × margin_q)")
            for t in range(3):
                for a in range(5):
                    for p in range(2):
                        for m in range(3):
                            key4 = self._create_group_key4(t, a, p, m)
                            idx = (token == t) & (attn == a) & (pos == p) & (margin_q == m)
                            cnt = int(idx.sum())
                            if cnt == 0:
                                print(f"  {key4}: count=0")
                                continue
                            soft_s = _safe_stats(np.asarray(soft_labels)[idx], percentiles=(95, 99))
                            hard_rate = float(np.mean(np.asarray(hard_labels)[idx]))
                            conf_s = _safe_stats(np.asarray(c)[idx], percentiles=(95,))
                            print(f"  {key4}: count={cnt}, "
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
            for a in range(5):
                idx = (token == t) & (attn == a)
                key = f"t{t}_a{a}"
                if idx.sum() >= self.min_samples_per_group:
                    self.level2[key] = self._fit_iso_binned(c[idx], y[idx], w[idx] if w is not None else None)
                else:
                    self.level2[key] = None

        # L3
        self.level3 = {}
        for t in range(3):
            for a in range(5):
                for p in range(2):
                    key = self._create_group_key(t, a, p)
                    idx = (token == t) & (attn == a) & (pos == p)
                    if idx.sum() >= self.min_samples_per_group:
                        self.level3[key] = self._fit_iso_binned(c[idx], y[idx], w[idx] if w is not None else None)
                    else:
                        self.level3[key] = None

        # L4
        self.level4 = {}
        for t in range(3):
            for a in range(5):
                for p in range(2):
                    for m in range(3):
                        key = self._create_group_key4(t, a, p, m)
                        idx = (token == t) & (attn == a) & (pos == p) & (margin_q == m)
                        if idx.sum() >= self.min_samples_per_group:
                            self.level4[key] = self._fit_iso_binned(c[idx], y[idx], w[idx] if w is not None else None)
                        else:
                            self.level4[key] = None

        self.is_fitted = True

    def predict_proba(self, features):
        proc = self._preprocess_features(features, fit_mode=False)
        c = proc['draft_conf']
        token, attn, pos = proc['token_type'], proc['attn_q'], proc['pos_bin']
        margin_q = proc['margin_q']

        out = np.zeros_like(c, dtype=np.float32)
        for t in range(3):
            for a in range(5):
                for p in range(2):
                    for m in range(3):
                        mask = (token == t) & (attn == a) & (pos == p) & (margin_q == m)
                        if mask.sum() == 0:
                            continue
                        key4 = self._create_group_key4(t, a, p, m)
                        key3 = self._create_group_key(t, a, p)
                        key2 = f"t{t}_a{a}"
                        key1 = f"t{t}"
                        cal = (
                            (self.level4.get(key4) if hasattr(self, 'level4') else None)
                            or self.level3.get(key3)
                            or self.level2.get(key2)
                            or self.level1.get(key1)
                            or self.global_calibrator
                        )
                        if cal is not None:
                            out[mask] = cal.predict(c[mask])
                        else:
                            out[mask] = self.global_mean
        return np.clip(out, 1e-4, 1 - 1e-4)

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
