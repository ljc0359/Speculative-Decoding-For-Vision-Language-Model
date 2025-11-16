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
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss

import json
import time

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

    def _create_group_key2(self, token_type: int, attn_q: int) -> str:
        return f"t{token_type}_a{attn_q}"
        
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
        # 兼容旧 pickle 的命名空间映射
        class _CompatUnpickler(pickle.Unpickler):
            def find_class(self_inner, module, name):
                from eagle.model.calibrators import (
                    GroupedIsotonicCalibrator,
                    MonotonicNetworkCalibrator,
                    MonotonicMLP,
                )
                mapping = {
                    "GroupedIsotonicCalibrator": GroupedIsotonicCalibrator,
                    "MonotonicNetworkCalibrator": MonotonicNetworkCalibrator,
                    "MonotonicMLP": MonotonicMLP,
                    "_AffineMono": MonotonicNetworkCalibrator._AffineMono,
                    "MonotonicNetworkCalibrator._AffineMono": MonotonicNetworkCalibrator._AffineMono,
                }
                if name in mapping and module in (
                    "lmms_eval.__main__",
                    "__main__",
                    "eagle.model.calibrators",
                ):
                    return mapping[name]
                if name.endswith("._AffineMono"):
                    return MonotonicNetworkCalibrator._AffineMono
                if name in mapping:
                    return mapping[name]
                return super(_CompatUnpickler, self_inner).find_class(module, name)

        with open(path, 'rb') as f:
            return _CompatUnpickler(f).load()

# =========================
# Grouped Isotonic
# =========================

class GroupedIsotonicCalibrator(BaseCalibrator):
    """12 组 Isotonic：token_type × attn_q × pos_bin，层级回退"""

    def __init__(self, min_samples_per_group: int = 100,  # 从200降低到100，适应稀疏数据
                 out_of_bounds: str = 'clip',
                 target: str = 'hard',
                 use_adaptive_params: bool = False,
                 max_grouping_level: int = 2  # 控制最大分组层级：1=token_type, 2=token_type+attn, 3=+pos_bin, 4=+margin_q
                 ):
        super().__init__()
        self.min_samples_per_group = min_samples_per_group
        self.out_of_bounds = out_of_bounds
        self.target = target  # 'hard' or 'soft'
        self.use_adaptive_params = use_adaptive_params
        self.max_grouping_level = max_grouping_level
        self.level1, self.level2, self.level3, self.level4 = {}, {}, {}, {}
        self.verbose = True
        
        self.global_calibrator = None
        self.global_mean = None

    def _fit_iso_binned(self, x, y, w=None, n_bins: int = 20, group_key: str = "unknown"):
        # 简化版：直接在原始对 (confidence, label) 上拟合单调等概率校准器；不进行权重操控、尾部收缩或数据增强
        iso = IsotonicRegression(out_of_bounds=self.out_of_bounds, increasing=True)
        iso.fit(np.asarray(x), np.asarray(y), sample_weight=np.asarray(w) if w is not None else None)
        return iso

    # 新增：置信度带参与的最细分组 key（L5）
    def _create_group_key5(self, t: int, a: int, p: int, m: int, b: int) -> str:
        return f"t{t}_a{a}_p{p}_m{m}_b{b}"

    def fit(self, features, soft_labels, hard_labels, sample_weights=None):
        proc = self._preprocess_features(features, fit_mode=True)
        c = proc['draft_conf']
        token = proc['token_type']
        attn = proc['attn_q']
        pos = proc['pos_bin']
        margin_q = proc.get('margin_q', np.zeros_like(attn))
        
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
            hard_pos_rate = float(np.mean(y)) if len(y) > 0 else float('nan')
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
                hard_rate = float(np.mean(np.asarray(y)[idx]))
                conf_s = _safe_stats(np.asarray(c)[idx], percentiles=(95,))
                print(f"  t{t}: count={cnt}, "
                      f"hard_pos={hard_rate:.4f}, "
                      f"conf[max={conf_s.get('max'):.4f}, p95={conf_s.get('p95'):.4f}]")

            # L2: token_type × attn_q
            print("[GroupedIsotonicCalibrator] 分组统计 - L2(token_type × attn_q)")
            for t in range(3):
                for a in range(5):
                    idx = (token == t) & (attn == a)
                    cnt = int(idx.sum())
                    key = f"t{t}_a{a}"
                    if cnt == 0:
                        print(f"  {key}: count=0")
                        continue
                    hard_rate = float(np.mean(np.asarray(y)[idx]))
                    conf_s = _safe_stats(np.asarray(c)[idx], percentiles=(95,))
                    print(f"  {key}: count={cnt}, "
                          f"hard_pos={hard_rate:.4f}, "
                          f"conf[max={conf_s.get('max'):.4f}, p95={conf_s.get('p95'):.4f}]")

            # L3: token_type × attn_q × pos_bin
            print("[GroupedIsotonicCalibrator] 分组统计 - L3(token_type × attn_q × pos_bin)")
            for t in range(3):
                for a in range(5):
                    for p in range(2):
                        key = self._create_group_key(t, a, p)
                        idx = (token == t) & (attn == a) & (pos == p)
                        cnt = int(idx.sum())
                        if cnt == 0:
                            print(f"  {key}: count=0")
                            continue
                        hard_rate = float(np.mean(np.asarray(y)[idx]))
                        conf_s = _safe_stats(np.asarray(c)[idx], percentiles=(95,))
                        print(f"  {key}: count={cnt}, "
                              f"hard_pos={hard_rate:.4f}, "
                              f"conf[max={conf_s.get('max'):.4f}, p95={conf_s.get('p95'):.4f}]")

            # L4: token_type × attn_q × pos_bin × margin_q
            print("[GroupedIsotonicCalibrator] 分组统计 - L4(token_type × attn_q × pos_bin × margin_q)")
            for t in range(3):
                for a in range(5):
                    for p in range(2):
                        for m in range(3):
                            key = self._create_group_key4(t, a, p, m)
                            idx = (token == t) & (attn == a) & (pos == p) & (margin_q == m)
                            cnt = int(idx.sum())
                            if cnt == 0:
                                print(f"  {key}: count=0")
                                continue
                            hard_rate = float(np.mean(np.asarray(y)[idx]))
                            conf_s = _safe_stats(np.asarray(c)[idx], percentiles=(95,))
                            print(f"  {key}: count={cnt}, "
                                  f"hard_pos={hard_rate:.4f}, "
                                  f"conf[max={conf_s.get('max'):.4f}, p95={conf_s.get('p95'):.4f}]")
        except Exception as e:
            print(f"[GroupedIsotonicCalibrator] 统计打印时出现异常：{e}")

        # 全局回退
        self.global_calibrator = self._fit_iso_binned(c, y, w, n_bins=20)
        self.global_mean = float(np.average(y, weights=w) if w is not None else np.mean(y))

        # 统一使用四层分组策略
        print(f"[GroupedIsotonicCalibrator] 使用统一四层分组策略")
        
        # 初始化四层分组存储
        self.level1, self.level2, self.level3, self.level4 = {}, {}, {}, {}
        
        # L1: token_type only
        for t in range(3):
            idx = (token == t)
            key = f"t{t}"
            if idx.sum() >= self.min_samples_per_group:
                print(f"    L1 {key}: {int(idx.sum())} 样本")
                self.level1[key] = self._fit_iso_binned(c[idx], y[idx], w[idx] if w is not None else None)
            else:
                self.level1[key] = None

        # L2: token_type × attn_q
        for t in range(3):
            for a in range(5):
                idx = (token == t) & (attn == a)
                key = f"t{t}_a{a}"
                if idx.sum() >= self.min_samples_per_group:
                    print(f"    L2 {key}: {int(idx.sum())} 样本")
                    self.level2[key] = self._fit_iso_binned(c[idx], y[idx], w[idx] if w is not None else None)
                else:
                    self.level2[key] = None

        # L3: token_type × attn_q × pos_bin
        for t in range(3):
            for a in range(5):
                for p in range(2):
                    key = self._create_group_key(t, a, p)
                    idx = (token == t) & (attn == a) & (pos == p)
                    if idx.sum() >= self.min_samples_per_group:
                        print(f"    L3 {key}: {int(idx.sum())} 样本")
                        self.level3[key] = self._fit_iso_binned(c[idx], y[idx], w[idx] if w is not None else None)
                    else:
                        self.level3[key] = None

        # L4: token_type × attn_q × pos_bin × margin_q
        for t in range(3):
            for a in range(5):
                for p in range(2):
                    for m in range(3):
                        key = self._create_group_key4(t, a, p, m)
                        idx = (token == t) & (attn == a) & (pos == p) & (margin_q == m)
                        if idx.sum() >= self.min_samples_per_group:
                            print(f"    L4 {key}: {int(idx.sum())} 样本")
                            self.level4[key] = self._fit_iso_binned(c[idx], y[idx], w[idx] if w is not None else None)
                        else:
                            self.level4[key] = None
        
        self.is_fitted = True

    def predict_proba(self, features):
        proc = self._preprocess_features(features, fit_mode=False)
        c = proc['draft_conf']
        token, attn, pos = proc['token_type'], proc['attn_q'], proc['pos_bin']
        margin_q = proc['margin_q']

        # 有效置信度掩码：过滤 NaN/Inf 以及越界值
        mask_valid = np.isfinite(c)
        mask_valid &= (c >= 0.0) & (c <= 1.0)

        out = np.zeros_like(c, dtype=np.float32)
        
        # 根据max_grouping_level控制分组策略
        if self.max_grouping_level >= 4:
            # 使用四层分组策略
            for t in range(3):
                for a in range(5):
                    for p in range(2):
                        for m in range(3):
                            mask = (token == t) & (attn == a) & (pos == p) & (margin_q == m)
                            mask_group = mask & mask_valid
                            if mask_group.sum() == 0:
                                continue
                            key4 = self._create_group_key4(t, a, p, m)
                            key3 = self._create_group_key(t, a, p)
                            key2 = f"t{t}_a{a}"
                            key1 = f"t{t}"
                            cal = (
                                self.level4.get(key4)
                                or self.level3.get(key3)
                                or self.level2.get(key2)
                                or self.level1.get(key1)
                                or self.global_calibrator
                            )
                            if cal is not None:
                                try:
                                    out[mask_group] = cal.predict(c[mask_group])
                                except Exception:
                                    out[mask_group] = self.global_mean
                            else:
                                out[mask_group] = self.global_mean
        elif self.max_grouping_level == 3:
            # 使用三层分组策略（token_type + attn + pos_bin）
            for t in range(3):
                for a in range(5):
                    for p in range(2):
                        mask = (token == t) & (attn == a) & (pos == p)
                        mask_group = mask & mask_valid
                        if mask_group.sum() == 0:
                            continue
                        key3 = self._create_group_key(t, a, p)
                        key2 = f"t{t}_a{a}"
                        key1 = f"t{t}"
                        cal = (
                            self.level3.get(key3)
                            or self.level2.get(key2)
                            or self.level1.get(key1)
                            or self.global_calibrator
                        )
                        if cal is not None:
                            try:
                                out[mask_group] = cal.predict(c[mask_group])
                            except Exception:
                                out[mask_group] = self.global_mean
                        else:
                            out[mask_group] = self.global_mean
        elif self.max_grouping_level == 2:
            # 使用二层分组策略（token_type + attn）
            for t in range(3):
                for a in range(5):
                    mask = (token == t) & (attn == a)
                    mask_group = mask & mask_valid
                    if mask_group.sum() == 0:
                        continue
                    key2 = f"t{t}_a{a}"
                    key1 = f"t{t}"
                    cal = (
                        self.level2.get(key2)
                        or self.level1.get(key1)
                        or self.global_calibrator
                    )
                    if cal is not None:
                        try:
                            out[mask_group] = cal.predict(c[mask_group])
                        except Exception:
                            out[mask_group] = self.global_mean
                    else:
                        out[mask_group] = self.global_mean
        else:  # max_grouping_level == 1
            # 使用一层分组策略（仅token_type）
            for t in range(3):
                mask = (token == t)
                mask_group = mask & mask_valid
                if mask_group.sum() == 0:
                    continue
                key1 = f"t{t}"
                cal = (
                    self.level1.get(key1)
                    or self.global_calibrator
                )
                if cal is not None:
                    try:
                        out[mask_group] = cal.predict(c[mask_group])
                    except Exception:
                        out[mask_group] = self.global_mean
                else:
                    out[mask_group] = self.global_mean
        
        # 对无效置信度（NaN/Inf/越界）的样本使用全局均值回退
        out[~mask_valid] = self.global_mean
        # 最终数值清理与裁剪
        out = np.nan_to_num(out, nan=self.global_mean, posinf=1.0, neginf=0.0)
        return np.clip(out, 1e-4, 1 - 1e-4)

def load_calibration_data(path: str) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    加载校准数据，计算基于speculative decoding公式的真实接受率
    
    接受概率公式: acceptance_prob = min(1, p_base(token) / p_draft(token))
    其中 p_base 和 p_draft 是概率值（不是logits）
    """
    if path.endswith('.json'):
        raw = json.load(open(path, 'r'))
        data = raw['candidate_calibration_data'] if isinstance(raw, dict) and 'candidate_calibration_data' in raw else raw
        feats = {}
        keys = ['draft_confidence', 'tree_depth', 'avg_visual_attention_intensity', 'draft_margin',
                'token_category']
        for k in keys:
            if k in data[0]:
                feats[k] = np.array([x[k] for x in data])
        
        # 计算基于speculative decoding公式的真实接受率
        base_confidences = np.array([x['base_confidence'] for x in data])
        draft_confidences = np.array([x['draft_confidence'] for x in data])
        
        # 确保draft_confidence不为0，避免除零错误
        draft_confidences = np.maximum(draft_confidences, 1e-10)
        
        # 应用speculative decoding接受公式: min(1, p_base / p_draft)
        acceptance_probs = np.minimum(1.0, base_confidences / draft_confidences)
        
        soft = acceptance_probs
        hard = np.array([x.get('base_top1_token', x.get('hard_label', 0)) for x in data]).astype(int)
        # --- NaN过滤：移除任意相关字段为NaN的样本 ---
        valid_mask = (~np.isnan(soft)) & (~np.isnan(base_confidences)) & (~np.isnan(draft_confidences))
        for k, arr in list(feats.items()):
            if np.issubdtype(arr.dtype, np.floating):
                valid_mask &= ~np.isnan(arr)
        # 应用过滤掩码
        soft = soft[valid_mask]
        hard = hard[valid_mask]
        for k in feats:
            feats[k] = feats[k][valid_mask]
        return feats, soft, hard
        
    elif path.endswith('.npz'):
        d = np.load(path)
        feats = {k: d[k] for k in ['draft_confidence', 'tree_depth', 'avg_visual_attention_intensity',
                                   'draft_margin', 'token_category'] if k in d}
        
        # 检查是否已经有预计算的接受率，否则计算
        if 'acceptance_probability' in d:
            soft = d['acceptance_probability']
        elif 'base_confidence' in d and 'draft_confidence' in d:
            # 计算基于speculative decoding公式的真实接受率
            base_confidences = d['base_confidence']
            draft_confidences = d['draft_confidence']
            
            # 确保draft_confidence不为0，避免除零错误
            draft_confidences = np.maximum(draft_confidences, 1e-10)
            
            # 应用speculative decoding接受公式: min(1, p_base / p_draft)
            acceptance_probs = np.minimum(1.0, base_confidences / draft_confidences)
            soft = acceptance_probs
        else:
            # 回退到原始逻辑
            soft = d['base_confidence'] if 'base_confidence' in d else d['soft_labels']
            
        hard = d['base_top1_token'] if 'base_top1_token' in d else d['hard_labels']
        # --- NaN过滤：移除任意相关字段为NaN的样本 ---
        valid_mask = ~np.isnan(soft)
        if 'draft_confidence' in d:
            valid_mask &= ~np.isnan(d['draft_confidence'])
        if 'base_confidence' in d:
            valid_mask &= ~np.isnan(d['base_confidence'])
        for k, arr in list(feats.items()):
            if np.issubdtype(arr.dtype, np.floating):
                valid_mask &= ~np.isnan(arr)
        # 应用过滤掩码
        soft = soft[valid_mask]
        hard = hard[valid_mask]
        for k in feats:
            feats[k] = feats[k][valid_mask]
        return feats, soft, hard
    else:
        raise ValueError("Unsupported file format")


def benchmark_calibrator_timing(
    calibrator: BaseCalibrator,
    features: Dict[str, np.ndarray],
    soft_labels: np.ndarray,
    hard_labels: np.ndarray,
    sample_weights: Optional[np.ndarray] = None,
    n_predict_rounds: int = 5,
    split_batches: Optional[int] = None,
    seed: int = 42,
    refit: bool = True,
):
    """
    评估校准器的训练与推理耗时，并打印结构化表格。

    参数:
    - calibrator: 已实例化的校准器 (如 GroupedIsotonicCalibrator)。
    - features: 特征字典，键必须包含 'draft_confidence'、'token_category'、'avg_visual_attention_intensity'、'tree_depth' 等训练所需字段。
    - soft_labels: 软标签 (概率)。
    - hard_labels: 硬标签 (0/1)。
    - sample_weights: 可选样本权重。
    - n_predict_rounds: 重复预测轮数，用于统计均值与方差。
    - split_batches: 若设置为正整数，则将数据按该批次数分块进行推理计时，模拟批处理。
    - seed: 随机种子。
    - refit: 是否在计时前对传入校准器进行重新训练 (fit)。

    输出: 在标准输出打印结构化表格，包括训练耗时、预测均值/方差、吞吐量等。
    """
    set_seed(seed)

    # 先做一次预处理以拿到 draft_conf 并构造有效掩码，避免 NaN/Inf 干扰
    try:
        proc = calibrator._preprocess_features(features, fit_mode=True)
        c = np.asarray(proc['draft_conf'])
    except Exception:
        # 在极端情况下（预处理失败）直接尝试使用原始 features
        c = np.asarray(features.get('draft_confidence', []))

    # 构建有效样本掩码
    mask_valid = np.isfinite(c) & (c >= 0.0) & (c <= 1.0)
    if soft_labels is not None:
        mask_valid &= np.isfinite(np.asarray(soft_labels))
    if hard_labels is not None:
        mask_valid &= np.isfinite(np.asarray(hard_labels))
    if sample_weights is not None:
        mask_valid &= np.isfinite(np.asarray(sample_weights))

    def _apply_mask_to_features(feats: Dict[str, np.ndarray], mask: np.ndarray) -> Dict[str, np.ndarray]:
        out = {}
        for k, v in feats.items():
            arr = np.asarray(v)
            # 仅当长度匹配时才使用掩码过滤
            if arr.ndim > 0 and arr.shape[0] == mask.shape[0]:
                out[k] = arr[mask]
            else:
                out[k] = arr
        return out

    feats_valid = _apply_mask_to_features(features, mask_valid)
    soft_valid = np.asarray(soft_labels)[mask_valid]
    hard_valid = np.asarray(hard_labels)[mask_valid]
    weights_valid = np.asarray(sample_weights)[mask_valid] if sample_weights is not None else None

    n_total = int(len(next(iter(features.values())))) if len(features) > 0 else 0
    n_used = int(mask_valid.sum())

    # 训练计时
    train_time = None
    if refit:
        t0 = time.perf_counter()
        calibrator.fit(feats_valid, soft_valid, hard_valid, sample_weights=weights_valid)
        train_time = time.perf_counter() - t0

    # 预测计时 (重复 n_predict_rounds 次，包含一次预热)
    # 预热
    try:
        _ = calibrator.predict_proba(feats_valid)
    except Exception:
        # 若预热失败，仍继续计时但标记异常
        pass

    pred_times = []
    if split_batches and isinstance(split_batches, int) and split_batches > 0:
        # 分批计时：将有效数据均匀划分为 split_batches 份
        M = n_used
        idx_all = np.arange(M)
        batches = np.array_split(idx_all, split_batches)

        def _slice_features(feats: Dict[str, np.ndarray], idx: np.ndarray) -> Dict[str, np.ndarray]:
            sliced = {}
            for k, v in feats.items():
                arr = np.asarray(v)
                if arr.ndim > 0 and arr.shape[0] == M:
                    sliced[k] = arr[idx]
                else:
                    sliced[k] = arr
            return sliced

        for _ in range(max(n_predict_rounds, 1)):
            t0 = time.perf_counter()
            for b in batches:
                f_b = _slice_features(feats_valid, b)
                _ = calibrator.predict_proba(f_b)
            dt = time.perf_counter() - t0
            pred_times.append(dt)
    else:
        # 整体一次性预测计时
        for _ in range(max(n_predict_rounds, 1)):
            t0 = time.perf_counter()
            _ = calibrator.predict_proba(feats_valid)
            dt = time.perf_counter() - t0
            pred_times.append(dt)

    pred_mean = float(np.mean(pred_times)) if len(pred_times) > 0 else float('nan')
    pred_std = float(np.std(pred_times)) if len(pred_times) > 0 else float('nan')
    throughput_samples_per_s = (n_used / pred_mean) if pred_mean and pred_mean > 0 else float('nan')

    # 尝试收集部分校准器配置，便于表格展示
    cal_name = calibrator.__class__.__name__
    cfg_items = []
    for attr in [
        'min_samples_per_group', 'out_of_bounds', 'target', 'use_adaptive_params', 'max_grouping_level'
    ]:
        if hasattr(calibrator, attr):
            cfg_items.append(f"{attr}={getattr(calibrator, attr)}")
    cfg_str = (", ".join(cfg_items)) if cfg_items else "(no extra params)"

    # 打印结构化表格
    def _row(label: str, value: str, w_label: int = 28, w_value: int = 24) -> str:
        return f"| {label:<{w_label}} | {value:<{w_value}} |"

    w_label = 28
    w_value = 24
    sep = "+" + "-" * (w_label + 2) + "+" + "-" * (w_value + 2) + "+"
    print(sep)
    print(_row("Calibrator", f"{cal_name}"))
    print(sep)
    print(_row("Config", cfg_str[:w_value]))
    print(sep)
    print(_row("Total samples", f"{n_total}"))
    print(_row("Used (valid) samples", f"{n_used}"))
    print(sep)
    if train_time is not None:
        print(_row("Train time (s)", f"{train_time:.6f}"))
    else:
        print(_row("Train time (s)", "(skipped)"))
    print(_row("Predict rounds", f"{n_predict_rounds}"))
    print(_row("Predict mean (s)", f"{pred_mean:.6f}"))
    print(_row("Predict std (s)", f"{pred_std:.6f}"))
    print(_row("Throughput (samples/s)", f"{throughput_samples_per_s:.2f}"))
    if split_batches and isinstance(split_batches, int) and split_batches > 0:
        print(_row("Batches", f"{split_batches}"))
    print(sep)

# =========================
# Simple offline test
# =========================

def _print_metrics(name: str, metrics: Dict[str, float]):
    print(f"\n{name}:")
    for k in ['brier', 'ece_eqfreq20', 'ece_fixed10', 'soft_mse', 'auroc']:
        if k in metrics:
            print(f"  {k}: {metrics[k]:.6f}")

def calibrator_training(calibrator_dir: str, json_path: str, target="soft"):
    """
    训练 Grouped Isotonic 与 Monotonic Network 两种校准器并保存到指定目录。
    参数:
        calibrator_dir: 输出保存目录（若不存在会创建）
        json_path: 训练数据 JSON 文件路径（需包含 features/soft_labels/hard_labels）
    返回:
        (GroupedIsotonicCalibrator, MonotonicNetworkCalibrator)
    """
    set_seed(42)
    print("=" * 60)
    print(f"[Calibrator] Training Grouped Isotonic & Monotonic Network from JSON: {json_path}")
    print("=" * 60)

    # 加载数据
    feats, soft, hard = load_calibration_data(json_path)
    n = len(soft)
    print(f"[Calibrator] Loaded {n} samples; pos_rate={hard.mean():.4f}")
    print(f"[Calibrator] Feature keys: {list(feats.keys())}")

    os.makedirs(calibrator_dir, exist_ok=True)

    # 训练 Isotonic（硬标签）
    gi = GroupedIsotonicCalibrator(
        min_samples_per_group=200,
        use_adaptive_params=True,
        max_grouping_level=2,
        target=target
    )
    
    gi.fit(feats, soft, hard)
    metrics_gi = gi.evaluate(feats, soft, hard)
    gi_model_path = os.path.join(calibrator_dir, f"grouped_isotonic_calibrator.pkl")
    gi_metrics_path = os.path.join(calibrator_dir, "grouped_isotonic_metrics.json")
    gi.save(gi_model_path)
    with open(gi_metrics_path, "w") as f:
        json.dump(metrics_gi, f, indent=2)
    print(f"[Calibrator] Saved Isotonic calibrator to: {gi_model_path}")
    print(f"[Calibrator] Saved training metrics to: {gi_metrics_path}")

    # 设备检查：若有 GPU 则用 CUDA，否则用 CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Calibrator] Using device for Monotonic Network: {device}")

    return gi, None


def compare_ece_train_val(json_path: str, val_ratio: float = 0.2, target: str = "soft", seed: int = 42,
                           n_bins: int = 20, equal_freq: bool = True, raw_conf_is_prob: bool = True,
                           save_svg_dir: str = None, pre_use_acceptance_prob: bool = True):
    """
    将数据划分为 80% 训练 + 20% 验证，训练校准器后在验证集上对比训练前后的校准质量（ECE/Brier/NLL）。

    参数:
        json_path: 校准数据 JSON/NPZ 路径
        val_ratio: 验证集占比（默认 0.2）
        target: 校准器学习目标（"soft" 使用真实接受率，或 "hard" 使用硬标签）
        seed: 随机种子保证可复现划分
        n_bins: ECE 分箱数量（默认 20）
        equal_freq: True 使用等频分箱，False 使用等宽分箱
        raw_conf_is_prob: 如果为 False，表示 draft_confidence 不是概率，将通过 Sigmoid 映射到 [0,1]
        save_svg_dir: 若提供路径，则在该目录生成 pre/post 可靠性图（SVG）
        pre_use_acceptance_prob: 若为 True（默认），pre 预测直接使用接受概率 soft_val，确保与后验预测同一事件语义；
                                 若为 False，则以 draft_confidence 为未校准预测（按 raw_conf_is_prob 处理）。

    打印:
        - 训练/验证样本数及正例率
        - 验证集上 pre/post 的 ECE、Brier 分数、LogLoss（NLL）
        - 指标改善量（post - pre，负值表示改善）

    返回:
        dict，包含 ece_pre/post、brier_pre/post、nll_pre/post、delta_ece/brier/nll、sizes
    """
    set_seed(seed)
    feats, soft, hard = load_calibration_data(json_path)
    n = len(hard)
    assert n > 0, "数据为空"
    assert 'draft_confidence' in feats, "features 必须包含 'draft_confidence'"

    # 划分索引
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    split = int(n * (1.0 - val_ratio))
    train_idx, val_idx = idx[:split], idx[split:]

    # 子集构造
    feats_train = {k: v[train_idx] for k, v in feats.items()}
    feats_val = {k: v[val_idx] for k, v in feats.items()}
    soft_train, soft_val = soft[train_idx], soft[val_idx]
    hard_train, hard_val = hard[train_idx], hard[val_idx]

    print("=" * 60)
    print(f"[Calibrator] Split sizes: train={len(train_idx)} ({len(train_idx)/n:.1%}), val={len(val_idx)} ({len(val_idx)/n:.1%})")
    print(f"[Calibrator] Train pos_rate={hard_train.mean():.4f}; Val pos_rate={hard_val.mean():.4f}")
    print("=" * 60)

    # 训练校准器
    gi = GroupedIsotonicCalibrator(
        min_samples_per_group=200,
        use_adaptive_params=True,
        max_grouping_level=2,
        target=target
    )
    gi.fit(feats_train, soft_train, hard_train)

    # 验证集：数值过滤（不改训练流程，只在验证上做过滤）
    val_mask = np.isfinite(soft_val) & (soft_val >= 0.0) & (soft_val <= 1.0)
    filter_reason = "soft in [0,1] & finite"
    if not pre_use_acceptance_prob:
        rc = feats_val['draft_confidence']
        rc_is_finite = np.isfinite(rc)
        if raw_conf_is_prob:
            rc_in_range = (rc >= 0.0) & (rc <= 1.0)
            val_mask &= rc_is_finite & rc_in_range
            filter_reason += "; draft_confidence in [0,1] & finite (raw_conf_is_prob=True)"
        else:
            val_mask &= rc_is_finite
            filter_reason += "; draft_confidence finite (raw_conf_is_prob=False)"
    kept = int(np.sum(val_mask))
    dropped = int(len(val_mask) - kept)
    
    if dropped > 0:
        print(f"[Validation] Filtering invalid values on val set: kept={kept}, dropped={dropped} (rules: {filter_reason})")
    else:
        print(f"[Validation] No invalid values found on val set (rules: {filter_reason})")
    
    # 若过滤结果为空，回退到未过滤的验证集
    if kept == 0:
        print("[Validation] Warning: No valid samples after filtering; falling back to unfiltered val set.")
        feats_val_filt = feats_val
        soft_val_filt = soft_val
        hard_val_filt = hard_val
    else:
        feats_val_filt = {k: v[val_mask] for k, v in feats_val.items()}
        soft_val_filt = soft_val[val_mask]
        hard_val_filt = hard_val[val_mask]

    # 验证集：训练前（未校准）预测（在过滤后的验证集上）
    if pre_use_acceptance_prob:
        # 使用接受概率（同一事件语义）作为 pre baseline，确保公平比较
        pred_pre = np.clip(soft_val_filt, 1e-4, 1 - 1e-4)
        print("[Validation] Pre predictor: acceptance_prob (soft)")
    else:
        raw_pre = feats_val_filt['draft_confidence']
        if raw_conf_is_prob:
            pred_pre = np.clip(raw_pre, 1e-4, 1 - 1e-4)
            print("[Validation] Pre predictor: raw draft_confidence (treated as probability)")
        else:
            pred_pre = 1.0 / (1.0 + np.exp(-raw_pre))  # Sigmoid 映射到概率域
            pred_pre = np.clip(pred_pre, 1e-4, 1 - 1e-4)
            print("[Validation] Pre predictor: sigmoid(draft_confidence)")

    # 验证集：训练后（经校准器输出）预测（在过滤后的验证集上）
    pred_post = gi.predict_proba(feats_val_filt)

    # 指标计算（在过滤后的验证集上）
    ece_pre = BaseCalibrator._compute_ece(pred_pre, hard_val_filt, sample_weights=None, n_bins=n_bins, equal_freq=equal_freq)
    ece_post = BaseCalibrator._compute_ece(pred_post, hard_val_filt, sample_weights=None, n_bins=n_bins, equal_freq=equal_freq)
    brier_pre = brier_score_loss(hard_val_filt, pred_pre)
    brier_post = brier_score_loss(hard_val_filt, pred_post)
    nll_pre = log_loss(hard_val_filt, np.vstack([1-pred_pre, pred_pre]).T, labels=[0,1])
    nll_post = log_loss(hard_val_filt, np.vstack([1-pred_post, pred_post]).T, labels=[0,1])

    delta_ece = ece_post - ece_pre
    delta_brier = brier_post - brier_pre
    delta_nll = nll_post - nll_pre

    print(f"[Validation] ECE(pre, bins={n_bins}, equal_freq={equal_freq}) = {ece_pre:.6f}")
    print(f"[Validation] ECE(post, bins={n_bins}, equal_freq={equal_freq}) = {ece_post:.6f}")
    print(f"[Validation] ΔECE = {delta_ece:.6f} (负值表示改善)")
    print(f"[Validation] Brier(pre) = {brier_pre:.6f}")
    print(f"[Validation] Brier(post) = {brier_post:.6f}")
    print(f"[Validation] ΔBrier = {delta_brier:.6f} (负值表示改善)")
    print(f"[Validation] NLL(pre) = {nll_pre:.6f}")
    print(f"[Validation] NLL(post) = {nll_post:.6f}")
    print(f"[Validation] ΔNLL = {delta_nll:.6f} (负值表示改善)")

    # 可靠性图（SVG）
    if save_svg_dir is not None:
        import os
        import matplotlib.pyplot as plt
        os.makedirs(save_svg_dir, exist_ok=True)

        def _bin_stats(pred: np.ndarray, labels: np.ndarray):
            if equal_freq:
                quantiles = np.linspace(0, 1, n_bins + 1)
                boundaries = np.quantile(pred, quantiles)
                boundaries = np.unique(boundaries)
                if len(boundaries) < n_bins + 1:
                    boundaries = np.linspace(0, 1, n_bins + 1)
                bin_idx = np.digitize(pred, boundaries) - 1
                bin_idx = np.clip(bin_idx, 0, len(boundaries) - 2)
                actual_bins = len(boundaries) - 1
            else:
                boundaries = np.linspace(0, 1, n_bins + 1)
                bin_idx = np.digitize(pred, boundaries) - 1
                bin_idx = np.clip(bin_idx, 0, n_bins - 1)
                actual_bins = n_bins
            bin_conf, bin_acc = [], []
            for i in range(actual_bins):
                m = bin_idx == i
                if np.sum(m) > 0:
                    bin_conf.append(float(np.mean(pred[m])))
                    bin_acc.append(float(np.mean(labels[m])))
            return np.array(bin_conf), np.array(bin_acc)

        def _plot_reliability(bin_conf: np.ndarray, bin_acc: np.ndarray, title: str, save_name: str):
            plt.figure(figsize=(8, 6))
            plt.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.8, label='Perfect Calibration')
            plt.plot(bin_conf, bin_acc, 'o-', color='steelblue', linewidth=2, markersize=6, label='Binned')
            plt.xlabel('Confidence', fontsize=12)
            plt.ylabel('Empirical Acceptance Rate', fontsize=12)
            plt.title(title, fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.legend(loc='upper left')
            save_path = os.path.join(save_svg_dir, save_name)
            plt.savefig(save_path, format='svg', bbox_inches='tight')
            plt.close()
            print(f"[Validation] Reliability diagram saved: {save_path}")

        conf_pre, acc_pre = _bin_stats(pred_pre, hard_val)
        conf_post, acc_post = _bin_stats(pred_post, hard_val)
        _plot_reliability(conf_pre, acc_pre, f'Reliability (pre, bins={n_bins}, equal_freq={equal_freq})', 'reliability_pre.svg')
        _plot_reliability(conf_post, acc_post, f'Reliability (post, bins={n_bins}, equal_freq={equal_freq})', 'reliability_post.svg')

    return {
        'train_size': int(len(train_idx)),
        'val_size': int(len(val_idx)),
        'ece_pre': float(ece_pre),
        'ece_post': float(ece_post),
        'delta_ece': float(delta_ece),
        'brier_pre': float(brier_pre),
        'brier_post': float(brier_post),
        'delta_brier': float(delta_brier),
        'nll_pre': float(nll_pre),
        'nll_post': float(nll_post),
        'delta_nll': float(delta_nll)
    }

if __name__ == "__main__":
    # test_calibrator_training()
    # compare_before_after_calibration()
    calibrator_path = "/root/Speculative_decoding/calibration_data_13b/test_calibrators"
    json_path = "/root/Speculative_decoding/calibration_data/chartqa_calib_isotonic_train_600_val_0_test_400_total_1000_temperature_0/training_calibration_data.json"
    compare_ece_train_val(json_path)
    # calibrator_training(calibrator_path, json_path, "soft")

    # ======= Benchmark: 训练与推理耗时 =======
    try:
        res = load_calibration_data(json_path)
        if isinstance(res, tuple):
            if len(res) == 4:
                features, soft_labels, hard_labels, sample_weights = res
            elif len(res) == 3:
                features, soft_labels, hard_labels = res
                sample_weights = None
            else:
                raise ValueError(f"Unexpected return length {len(res)} from load_calibration_data")
        else:
            raise ValueError("load_calibration_data did not return a tuple")
    except Exception as e:
        print(f"[main] 加载校准数据失败：{e}")
        features, soft_labels, hard_labels, sample_weights = {}, np.array([]), np.array([]), None

    # 尝试补全缺失的关键特征键，避免计时过程因 KeyError 中断
    required_keys = ['draft_confidence', 'token_category', 'avg_visual_attention_intensity', 'tree_depth']
    if isinstance(features, dict):
        # 推断样本数 n
        if 'draft_confidence' in features:
            n = int(len(np.asarray(features['draft_confidence'])))
        else:
            n = 0
            for v in features.values():
                try:
                    n = int(len(np.asarray(v)))
                    if n > 0:
                        break
                except Exception:
                    continue
        for k in required_keys:
            if k not in features:
                if k == 'token_category':
                    features[k] = np.array(['content'] * n)
                elif k == 'avg_visual_attention_intensity':
                    features[k] = np.zeros(n, dtype=float)
                elif k == 'tree_depth':
                    features[k] = np.zeros(n, dtype=int)
                elif k == 'draft_confidence':
                    features[k] = np.clip(np.zeros(n, dtype=float), 0.0, 1.0)

    calib = GroupedIsotonicCalibrator(min_samples_per_group=200, target='hard')
    try:
        benchmark_calibrator_timing(
            calibrator=calib,
            features=features,
            soft_labels=soft_labels,
            hard_labels=hard_labels,
            sample_weights=sample_weights,
            n_predict_rounds=5,
            split_batches=4,
            seed=42,
            refit=True,
        )
    except Exception as e:
        print(f"[main] 计时基准运行失败：{e}")
