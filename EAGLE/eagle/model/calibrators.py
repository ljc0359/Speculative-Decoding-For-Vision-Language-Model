# -*- coding: utf-8 -*-
"""
Calibrators for EAGLE draft confidence
- Grouped Isotonic (with hierarchical fallback)
- Monotonic MLP (stabilized, hard-label first)
- Platt scaling baseline

=== 校准优化改进 (基于文献研究) ===
本版本针对高置信度区域的校准问题进行了以下优化：

1. **参数温和化**：
   - tail_wrong_boost: 5.0 → 1.5-3.0 (避免过度提升错误样本权重)
   - tail_gate_k: 10.0 → 3.0 (使sigmoid门控更平滑)
   - tail_shrink_alpha: 0.5-0.8 → 0.3-0.5 (减少先验收缩强度)

2. **自适应分桶策略**：
   - 高置信度样本<10%时使用2倍最小桶大小
   - 高置信度样本<20%时使用1.5倍最小桶大小
   - 实现等频分桶替代等宽分桶

3. **数据增强策略**：
   - 对高置信度稀疏桶增加样本权重(最大3倍)
   - 根据稀疏程度动态调整权重提升倍数

这些改进遵循了概率校准的核心原则：平滑性、泛化能力和统计稳定性。
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

    def __init__(self, min_samples_per_group: int = 200,
                 out_of_bounds: str = 'clip',
                 target: str = 'hard',
                 use_adaptive_params: bool = False,
                 use_hybrid_grouping: bool = True,
                 confidence_threshold: float = 0.6,
                 enable_temperature_scaling: bool = True,
                 temperature_search_range: tuple = (0.5, 3.0),
                 temperature_search_steps: int = 20):
        super().__init__()
        self.min_samples_per_group = min_samples_per_group
        self.out_of_bounds = out_of_bounds
        self.target = target  # 'hard' or 'soft'
        self.use_adaptive_params = use_adaptive_params  # 动态校准参数开关
        self.use_hybrid_grouping = use_hybrid_grouping  # 混合分组策略开关
        self.confidence_threshold = confidence_threshold  # 高低置信度分组阈值
        
        # Temperature Scaling 相关参数
        self.enable_temperature_scaling = enable_temperature_scaling
        self.temperature_search_range = temperature_search_range
        self.temperature_search_steps = temperature_search_steps
        
        # 四层分组存储（低置信度使用）
        self.level1, self.level2, self.level3, self.level4 = {}, {}, {}, {}
        # 两层分组存储（高置信度使用）
        self.high_conf_level1, self.high_conf_level2 = {}, {}
        self.verbose = True

        # 分组特定的temperature参数存储
        self.group_temperatures = {}  # 存储每个组的最优temperature
        self.global_temperature = 1.0  # 全局默认temperature
        
        self.global_calibrator = None
        self.global_mean = None
        
        # 静态参数（当use_adaptive_params=False时使用）- 优化为更温和的值
        self.static_tail_wrong_boost = 1.5  # 从2.0降低到1.5，避免过度提升错误样本权重
        self.static_tail_shrink_alpha = 0.3  # 从0.5降低到0.3，减少先验收缩强度
        self.static_tail_conf_threshold = 0.75

    def _compute_overconfidence_metrics(self, x, y, w=None):
        """
        计算组内过度自信程度的量化指标
        
        Returns:
            dict: 包含各种过度自信指标的字典
        """
        if len(x) == 0:
            return {"overconf_ratio": 0.0, "high_conf_error_rate": 0.0, "calibration_gap": 0.0}
        
        # 1. 过度自信比例：高置信度但错误的样本占比
        high_conf_mask = x >= 0.7  # 高置信度阈值
        if high_conf_mask.sum() == 0:
            high_conf_error_rate = 0.0
            overconf_ratio = 0.0
        else:
            high_conf_errors = (high_conf_mask & (y < 0.5)).sum()
            high_conf_total = high_conf_mask.sum()
            high_conf_error_rate = float(high_conf_errors / high_conf_total)
            overconf_ratio = float(high_conf_errors / len(x))
        
        # 2. 校准差距：置信度与准确率的最大偏差
        # 将置信度分为5个区间，计算每个区间的校准差距
        conf_bins = np.linspace(0, 1, 6)  # [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        max_gap = 0.0
        
        for i in range(len(conf_bins) - 1):
            bin_mask = (x >= conf_bins[i]) & (x < conf_bins[i + 1])
            if i == len(conf_bins) - 2:  # 最后一个区间包含右端点
                bin_mask = (x >= conf_bins[i]) & (x <= conf_bins[i + 1])
            
            if bin_mask.sum() > 0:
                if w is not None:
                    bin_conf = float(np.average(x[bin_mask], weights=w[bin_mask]))
                    bin_acc = float(np.average(y[bin_mask], weights=w[bin_mask]))
                else:
                    bin_conf = float(np.mean(x[bin_mask]))
                    bin_acc = float(np.mean(y[bin_mask]))
                
                gap = abs(bin_conf - bin_acc)
                max_gap = max(max_gap, gap)
        
        # 3. 尾段过度自信强度：0.8+区间的平均校准偏差
        tail_mask = x >= 0.8
        if tail_mask.sum() > 0:
            if w is not None:
                tail_conf = float(np.average(x[tail_mask], weights=w[tail_mask]))
                tail_acc = float(np.average(y[tail_mask], weights=w[tail_mask]))
            else:
                tail_conf = float(np.mean(x[tail_mask]))
                tail_acc = float(np.mean(y[tail_mask]))
            tail_overconf = max(0.0, tail_conf - tail_acc)
        else:
            tail_overconf = 0.0
        
        return {
            "overconf_ratio": overconf_ratio,
            "high_conf_error_rate": high_conf_error_rate, 
            "calibration_gap": max_gap,
            "tail_overconf": tail_overconf
        }

    def _compute_adaptive_params(self, overconf_metrics):
        """
        基于过度自信指标动态计算校准参数
        
        Args:
            overconf_metrics: _compute_overconfidence_metrics的输出
            
        Returns:
            dict: 自适应的校准参数
        """
        # 基础参数 - 优化为更温和的值
        base_wrong_boost = 1.5  # 从2.0降低到1.5
        base_shrink_alpha = 0.3  # 从0.5降低到0.3
        base_threshold = 0.75
        
        # 根据过度自信程度动态调整
        overconf_ratio = overconf_metrics["overconf_ratio"]
        high_conf_error_rate = overconf_metrics["high_conf_error_rate"]
        calibration_gap = overconf_metrics["calibration_gap"]
        tail_overconf = overconf_metrics["tail_overconf"]
        
        # 1. 动态错误权重提升系数
        # 过度自信越严重，错误样本权重提升越大，但使用更温和的系数
        boost_multiplier = 1.0 + 2.0 * overconf_ratio + 1.5 * high_conf_error_rate  # 降低系数
        adaptive_wrong_boost = base_wrong_boost * boost_multiplier
        adaptive_wrong_boost = min(adaptive_wrong_boost, 3.0)  # 从5.0降低到3.0，避免过度提升
        
        # 2. 动态先验收缩强度
        # 校准差距越大，收缩越强，但使用更温和的系数
        shrink_multiplier = 1.0 + 1.0 * calibration_gap + 1.5 * tail_overconf  # 降低系数
        adaptive_shrink_alpha = base_shrink_alpha * shrink_multiplier
        adaptive_shrink_alpha = min(adaptive_shrink_alpha, 0.5)  # 从0.8降低到0.5，避免过度收缩
        
        # 3. 动态置信度阈值
        # 如果尾段过度自信严重，降低阈值以扩大校准范围
        threshold_adjustment = -0.1 * tail_overconf
        adaptive_threshold = base_threshold + threshold_adjustment
        adaptive_threshold = max(adaptive_threshold, 0.6)  # 下限保护
        
        return {
            "tail_wrong_boost": adaptive_wrong_boost,
            "tail_shrink_alpha": adaptive_shrink_alpha,
            "tail_conf_threshold": adaptive_threshold,
            "boost_multiplier": boost_multiplier,
            "shrink_multiplier": shrink_multiplier
        }
    


    def _fit_iso_binned(self, x, y, w=None, n_bins: int = 20):
        s = np.argsort(x)
        xs, ys = x[s], y[s]
        ws = w[s] if w is not None else None

        # ========= 校准参数设置 =========
        if self.use_adaptive_params:
            # 动态参数计算
            # 1. 计算组内过度自信指标
            overconf_metrics = self._compute_overconfidence_metrics(xs, ys, ws)
            
            # 2. 基于指标动态调整校准参数
            adaptive_params = self._compute_adaptive_params(overconf_metrics)
            
            # 3. 应用自适应参数
            self.adaptive_bins = True
            self.max_bins = 20
            self.min_bin_size = 50
            self.tail_conf_threshold = adaptive_params["tail_conf_threshold"]
            self.tail_gate_k = 3.0  # 从10.0降低到3.0，使门控更平滑
            self.tail_wrong_boost = adaptive_params["tail_wrong_boost"]
            self.tail_shrink_alpha = adaptive_params["tail_shrink_alpha"]
            
            # 可选：打印调试信息（仅在样本数足够时）
            if len(xs) >= 100:  # 避免小样本组的噪声输出
                print(f"    [自适应校准] 样本数={len(xs)}, "
                      f"过度自信比例={overconf_metrics['overconf_ratio']:.3f}, "
                      f"高置信错误率={overconf_metrics['high_conf_error_rate']:.3f}, "
                      f"校准差距={overconf_metrics['calibration_gap']:.3f}")
                print(f"    [参数调整] wrong_boost={self.tail_wrong_boost:.2f}(×{adaptive_params['boost_multiplier']:.2f}), "
                      f"shrink_alpha={self.tail_shrink_alpha:.2f}(×{adaptive_params['shrink_multiplier']:.2f}), "
                      f"threshold={self.tail_conf_threshold:.2f}")
        else:
            # 使用静态参数（原始硬编码方式）
            self.adaptive_bins = True
            self.max_bins = 20
            self.min_bin_size = 50
            self.tail_conf_threshold = self.static_tail_conf_threshold
            self.tail_gate_k = 3.0  # 从10.0降低到3.0，使门控更平滑
            self.tail_wrong_boost = self.static_tail_wrong_boost
            self.tail_shrink_alpha = self.static_tail_shrink_alpha

        # ========= 改进的自适应分桶策略 =========
        # 自适应分桶：保证每桶有足够样本，降低高段统计方差；
        if self.adaptive_bins:
            # 检查高置信度样本稀疏性
            high_conf_mask = xs >= self.tail_conf_threshold
            high_conf_ratio = np.sum(high_conf_mask) / len(xs) if len(xs) > 0 else 0
            
            # 如果高置信度样本不足，使用更粗粒度的分桶
            if high_conf_ratio < 0.1:  # 高置信度样本少于10%
                # 使用更大的最小桶大小，减少桶数
                adaptive_min_bin_size = self.min_bin_size * 2
                print(f"    [自适应分桶] 高置信度样本稀疏({high_conf_ratio:.3f})，使用粗粒度分桶(min_size={adaptive_min_bin_size})")
            elif high_conf_ratio < 0.2:  # 高置信度样本少于20%
                adaptive_min_bin_size = int(self.min_bin_size * 1.5)
                print(f"    [自适应分桶] 高置信度样本较少({high_conf_ratio:.3f})，使用中等粒度分桶(min_size={adaptive_min_bin_size})")
            else:
                adaptive_min_bin_size = self.min_bin_size
            
            # 根据调整后的最小桶大小决定桶数
            bins_by_size = int(len(xs) // max(adaptive_min_bin_size, 1))
            bins = max(1, min(self.max_bins, bins_by_size))
        else:
            bins = n_bins
        
        # 实现等频分桶替代等宽分桶（在样本稀疏区域提供更好的校准效果）
        if self.adaptive_bins and len(xs) > bins:
            # 使用等频分桶：每个桶包含相似数量的样本
            quantiles = np.linspace(0, 1, bins + 1)
            edges = np.quantile(range(len(xs)), quantiles).astype(int)
            # 确保边界不重复
            edges = np.unique(edges)
            if len(edges) < bins + 1:
                # 如果去重后桶数不够，回退到等宽分桶
                edges = np.linspace(0, len(xs), bins + 1).astype(int)
        else:
            edges = np.linspace(0, len(xs), bins + 1).astype(int)

        xb, yb, wb = [], [], []
        # 组内先验：用于尾段收缩（有外部样本权重则用加权平均）
        prior_rate = float(np.average(ys, weights=ws)) if ws is not None else float(np.mean(ys))

        for i in range(bins):
            a, b = edges[i], edges[i + 1]
            if b <= a:
                continue

            xmean = float(xs[a:b].mean())
            # 尾段门控：置信度越高 gate 越接近 1（使用动态或静态阈值）
            gate = 1.0 / (1.0 + np.exp(-self.tail_gate_k * (xmean - self.tail_conf_threshold)))

            # 构造样本权重：在尾段提升负例（错误）的权重以抑制过置信（使用动态或静态权重）
            if ws is not None:
                w_i = ws[a:b].astype(np.float64)
            else:
                w_i = np.ones(b - a, dtype=np.float64)
            wrong = (ys[a:b] < 0.5).astype(np.float64)  # 负例视为"错误"
            
            # 数据增强策略：为高置信度稀疏区域增加有效样本权重
            bin_size = b - a
            if xmean >= self.tail_conf_threshold and bin_size < self.min_bin_size:
                # 高置信度且样本稀疏的桶，增加样本权重以提高统计稳定性
                sparsity_boost = max(1.0, self.min_bin_size / bin_size)  # 根据稀疏程度调整权重
                sparsity_boost = min(sparsity_boost, 3.0)  # 限制最大提升倍数
                w_i = w_i * sparsity_boost
                if self.verbose:
                    print(f"    [数据增强] 桶{i}(conf={xmean:.3f}, size={bin_size})权重提升{sparsity_boost:.2f}倍")
            
            w_boost = w_i * (1.0 + self.tail_wrong_boost * gate * wrong)

            # 桶内加权平均与总权重
            y_bin_raw = float(np.average(ys[a:b], weights=w_boost))
            w_sum = float(w_boost.sum())

            # 尾段先验收缩：向组内整体正例率收缩，稳定尾部输出（使用动态或静态收缩强度）
            shrink = float(self.tail_shrink_alpha * gate)
            y_bin = (1.0 - shrink) * y_bin_raw + shrink * prior_rate

            xb.append(xmean)
            yb.append(y_bin)
            wb.append(w_sum)

        iso = IsotonicRegression(out_of_bounds=self.out_of_bounds, increasing=True)
        iso.fit(np.array(xb), np.array(yb), sample_weight=np.array(wb))
        return iso

    # 新增：置信度带参与的最细分组 key（L5）
    def _create_group_key5(self, t: int, a: int, p: int, m: int, b: int) -> str:
        return f"t{t}_a{a}_p{p}_m{m}_b{b}"
    
    # 高置信度两层分组键生成方法
    def _create_high_conf_key1(self, t: int) -> str:
        """L1: token_type only (for high confidence)"""
        return f"hc_t{t}"
    
    def _create_high_conf_key2(self, t: int, a: int) -> str:
        """L2: token_type × attn_q (for high confidence)"""
        return f"hc_t{t}_a{a}"

    def _optimize_temperature_for_group(self, logits: np.ndarray, true_labels: np.ndarray, 
                                       sample_weights: Optional[np.ndarray] = None) -> float:
        """
        为特定组优化temperature参数，基于ECE最小化
        
        Args:
            logits: 原始logits (未经过softmax)
            true_labels: 真实标签 (0/1)
            sample_weights: 样本权重
            
        Returns:
            float: 最优temperature值
        """
        if len(logits) < 50:  # 样本数太少，使用全局默认值
            return 1.0
            
        # 将logits转换为概率（假设是二分类的logits）
        def logits_to_probs(logits_arr, temp):
            # 如果logits是单维的，假设是正类的logit
            if logits_arr.ndim == 1:
                scaled_logits = logits_arr / temp
                return 1.0 / (1.0 + np.exp(-scaled_logits))  # sigmoid
            else:
                # 如果是多类logits，使用softmax
                scaled_logits = logits_arr / temp
                exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
                return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # 搜索最优temperature
        temp_candidates = np.linspace(
            self.temperature_search_range[0], 
            self.temperature_search_range[1], 
            self.temperature_search_steps
        )
        
        best_temp = 1.0
        best_ece = float('inf')
        
        for temp in temp_candidates:
            try:
                probs = logits_to_probs(logits, temp)
                if probs.ndim > 1:
                    probs = probs[:, 1]  # 取正类概率
                
                # 计算ECE
                ece = self._compute_ece(probs, true_labels, sample_weights, n_bins=10, equal_freq=True)
                
                if ece < best_ece:
                    best_ece = ece
                    best_temp = temp
                    
            except Exception as e:
                continue  # 跳过有问题的temperature值
        
        return best_temp

    def _apply_temperature_scaling(self, logits: np.ndarray, temperature: float) -> np.ndarray:
        """
        应用temperature scaling到logits
        
        Args:
            logits: 原始logits
            temperature: temperature参数
            
        Returns:
            np.ndarray: 经过temperature scaling的概率
        """
        if temperature <= 0:
            temperature = 1.0
            
        if logits.ndim == 1:
            scaled_logits = logits / temperature
            return 1.0 / (1.0 + np.exp(-scaled_logits))  # sigmoid
        else:
            scaled_logits = logits / temperature
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
            return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def get_group_temperature(self, features) -> float:
        """
        根据token特征返回对应组的最优temperature
        
        Args:
            features: 可以是单个数据点字典或包含数组的特征字典
            
        Returns:
            float: 对应的temperature值（单个数据点）或 np.ndarray（多个数据点）
        """
        if not self.enable_temperature_scaling or not self.is_fitted:
            return 1.0
            
        # 检查是否为单个数据点
        if isinstance(features, dict) and 'draft_confidence' in features:
            # 单个数据点的情况
            if isinstance(features['draft_confidence'], (int, float)):
                return self._get_single_temperature(features)
            
        # 多个数据点的情况（原有逻辑）
        proc = self._preprocess_features(features, fit_mode=False)
        token = proc['token_type']
        attn = proc['attn_q']
        pos = proc['pos_bin']
        margin_q = proc.get('margin_q', np.zeros_like(attn))
        conf = proc['draft_conf']
        
        temperatures = np.ones(len(conf))
        
        # 根据置信度选择分组策略
        high_conf_mask = conf >= self.confidence_threshold
        
        for i in range(len(conf)):
            if high_conf_mask[i]:
                # 高置信度：使用简化分组
                key1 = self._create_high_conf_key1(token[i])
                key2 = self._create_high_conf_key2(token[i], attn[i])
                
                if key2 in self.group_temperatures:
                    temperatures[i] = self.group_temperatures[key2]
                elif key1 in self.group_temperatures:
                    temperatures[i] = self.group_temperatures[key1]
                else:
                    temperatures[i] = self.global_temperature
            else:
                # 低置信度：使用完整分组
                key4 = self._create_group_key4(token[i], attn[i], pos[i], margin_q[i])
                key3 = self._create_group_key(token[i], attn[i], pos[i])
                key2 = self._create_group_key2(token[i], attn[i])
                key1 = f"t{token[i]}"
                
                if key4 in self.group_temperatures:
                    temperatures[i] = self.group_temperatures[key4]
                elif key3 in self.group_temperatures:
                    temperatures[i] = self.group_temperatures[key3]
                elif key2 in self.group_temperatures:
                    temperatures[i] = self.group_temperatures[key2]
                elif key1 in self.group_temperatures:
                    temperatures[i] = self.group_temperatures[key1]
                else:
                    temperatures[i] = self.global_temperature
                    
        return temperatures
    
    def _get_single_temperature(self, data_point: dict) -> float:
        """
        为单个数据点获取temperature值
        
        Args:
            data_point: 单个数据点的特征字典
            
        Returns:
            float: 对应的temperature值
        """
        # 转换为数组格式以使用现有的预处理逻辑
        features_array = {}
        for key, value in data_point.items():
            if isinstance(value, (int, float)):
                features_array[key] = np.array([value])
            elif isinstance(value, str):
                features_array[key] = np.array([value])
            else:
                features_array[key] = np.array([value])
        
        proc = self._preprocess_features(features_array, fit_mode=False)
        token = proc['token_type'][0]
        attn = proc['attn_q'][0]
        pos = proc['pos_bin'][0]
        margin_q = proc.get('margin_q', np.array([0]))[0]
        conf = proc['draft_conf'][0]
        
        # 根据置信度选择分组策略
        if conf >= self.confidence_threshold:
            # 高置信度：使用简化分组
            key1 = self._create_high_conf_key1(token)
            key2 = self._create_high_conf_key2(token, attn)
            
            if key2 in self.group_temperatures:
                return self.group_temperatures[key2]
            elif key1 in self.group_temperatures:
                return self.group_temperatures[key1]
            else:
                return self.global_temperature
        else:
            # 低置信度：使用完整分组
            key4 = self._create_group_key4(token, attn, pos, margin_q)
            key3 = self._create_group_key(token, attn, pos)
            key2 = self._create_group_key2(token, attn)
            key1 = f"t{token}"
            
            if key4 in self.group_temperatures:
                return self.group_temperatures[key4]
            elif key3 in self.group_temperatures:
                return self.group_temperatures[key3]
            elif key2 in self.group_temperatures:
                return self.group_temperatures[key2]
            elif key1 in self.group_temperatures:
                return self.group_temperatures[key1]
            else:
                return self.global_temperature

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

        if self.use_hybrid_grouping:
            # 混合分组策略：根据置信度分别训练
            print(f"[GroupedIsotonicCalibrator] 使用混合分组策略，置信度阈值: {self.confidence_threshold}")
            
            # 分离高低置信度数据
            high_conf_mask = c >= self.confidence_threshold
            low_conf_mask = c < self.confidence_threshold
            
            high_conf_count = int(high_conf_mask.sum())
            low_conf_count = int(low_conf_mask.sum())
            print(f"  高置信度样本数: {high_conf_count} ({high_conf_count/len(c)*100:.1f}%)")
            print(f"  低置信度样本数: {low_conf_count} ({low_conf_count/len(c)*100:.1f}%)")
            
            # 高置信度两层分组训练
            self.high_conf_level1, self.high_conf_level2 = {}, {}
            

            
            # 高置信度 L1: token_type only
            for t in range(3):
                idx = high_conf_mask & (token == t)
                key = self._create_high_conf_key1(t)
                if idx.sum() >= self.min_samples_per_group:
                    self.high_conf_level1[key] = self._fit_iso_binned(c[idx], y[idx], w[idx] if w is not None else None)
                    print(f"    高置信度L1 {key}: {int(idx.sum())} 样本")
                else:
                    self.high_conf_level1[key] = None
            
            # 高置信度 L2: token_type × attn_q
            for t in range(3):
                for a in range(5):
                    idx = high_conf_mask & (token == t) & (attn == a)
                    key = self._create_high_conf_key2(t, a)
                    if idx.sum() >= self.min_samples_per_group:
                        self.high_conf_level2[key] = self._fit_iso_binned(c[idx], y[idx], w[idx] if w is not None else None)
                        print(f"    高置信度L2 {key}: {int(idx.sum())} 样本")
                    else:
                        self.high_conf_level2[key] = None
            
            # 低置信度四层分组训练
            self.level1, self.level2, self.level3, self.level4 = {}, {}, {}, {}
            
            # 低置信度 L1
            for t in range(3):
                idx = low_conf_mask & (token == t)
                key = f"t{t}"
                if idx.sum() >= self.min_samples_per_group:
                    self.level1[key] = self._fit_iso_binned(c[idx], y[idx], w[idx] if w is not None else None)
                else:
                    self.level1[key] = None

            # 低置信度 L2
            for t in range(3):
                for a in range(5):
                    idx = low_conf_mask & (token == t) & (attn == a)
                    key = f"t{t}_a{a}"
                    if idx.sum() >= self.min_samples_per_group:
                        self.level2[key] = self._fit_iso_binned(c[idx], y[idx], w[idx] if w is not None else None)
                    else:
                        self.level2[key] = None

            # 低置信度 L3
            for t in range(3):
                for a in range(5):
                    for p in range(2):
                        key = self._create_group_key(t, a, p)
                        idx = low_conf_mask & (token == t) & (attn == a) & (pos == p)
                        if idx.sum() >= self.min_samples_per_group:
                            self.level3[key] = self._fit_iso_binned(c[idx], y[idx], w[idx] if w is not None else None)
                        else:
                            self.level3[key] = None

            # 低置信度 L4
            for t in range(3):
                for a in range(5):
                    for p in range(2):
                        for m in range(3):
                            key = self._create_group_key4(t, a, p, m)
                            idx = low_conf_mask & (token == t) & (attn == a) & (pos == p) & (margin_q == m)
                            if idx.sum() >= self.min_samples_per_group:
                                self.level4[key] = self._fit_iso_binned(c[idx], y[idx], w[idx] if w is not None else None)
                            else:
                                self.level4[key] = None
        else:
            # 原始四层分组策略
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

        # ========= Temperature Scaling 优化 =========
        if self.enable_temperature_scaling:
            print("[GroupedIsotonicCalibrator] 开始为每个组优化temperature参数...")
            
            # 注意：这里我们需要原始logits，但当前只有draft_confidence
            # 作为临时方案，我们使用draft_confidence的logit形式
            # 在实际应用中，应该传入原始logits
            draft_logits = np.log(c / (1 - c + 1e-8))  # 将概率转换回logit（近似）
            
            # 优化全局temperature
            self.global_temperature = self._optimize_temperature_for_group(draft_logits, y, w)
            print(f"  全局最优temperature: {self.global_temperature:.3f}")
            
            if self.use_hybrid_grouping:
                # 高置信度组的temperature优化
                for t in range(3):
                    idx = high_conf_mask & (token == t)
                    if idx.sum() >= 50:  # 最小样本数要求
                        key = self._create_high_conf_key1(t)
                        temp = self._optimize_temperature_for_group(draft_logits[idx], y[idx], w[idx] if w is not None else None)
                        self.group_temperatures[key] = temp
                        print(f"  高置信度L1 {key}: temperature={temp:.3f}")
                
                for t in range(3):
                    for a in range(5):
                        idx = high_conf_mask & (token == t) & (attn == a)
                        if idx.sum() >= 50:
                            key = self._create_high_conf_key2(t, a)
                            temp = self._optimize_temperature_for_group(draft_logits[idx], y[idx], w[idx] if w is not None else None)
                            self.group_temperatures[key] = temp
                            print(f"  高置信度L2 {key}: temperature={temp:.3f}")
                
                # 低置信度组的temperature优化
                for t in range(3):
                    idx = low_conf_mask & (token == t)
                    if idx.sum() >= 50:
                        key = f"t{t}"
                        temp = self._optimize_temperature_for_group(draft_logits[idx], y[idx], w[idx] if w is not None else None)
                        self.group_temperatures[key] = temp
                        print(f"  低置信度L1 {key}: temperature={temp:.3f}")
                
                for t in range(3):
                    for a in range(5):
                        idx = low_conf_mask & (token == t) & (attn == a)
                        if idx.sum() >= 50:
                            key = self._create_group_key2(t, a)
                            temp = self._optimize_temperature_for_group(draft_logits[idx], y[idx], w[idx] if w is not None else None)
                            self.group_temperatures[key] = temp
                            print(f"  低置信度L2 {key}: temperature={temp:.3f}")
                
                for t in range(3):
                    for a in range(5):
                        for p in range(2):
                            idx = low_conf_mask & (token == t) & (attn == a) & (pos == p)
                            if idx.sum() >= 50:
                                key = self._create_group_key(t, a, p)
                                temp = self._optimize_temperature_for_group(draft_logits[idx], y[idx], w[idx] if w is not None else None)
                                self.group_temperatures[key] = temp
                                print(f"  低置信度L3 {key}: temperature={temp:.3f}")
                
                for t in range(3):
                    for a in range(5):
                        for p in range(2):
                            for m in range(3):
                                idx = low_conf_mask & (token == t) & (attn == a) & (pos == p) & (margin_q == m)
                                if idx.sum() >= 50:
                                    key = self._create_group_key4(t, a, p, m)
                                    temp = self._optimize_temperature_for_group(draft_logits[idx], y[idx], w[idx] if w is not None else None)
                                    self.group_temperatures[key] = temp
                                    print(f"  低置信度L4 {key}: temperature={temp:.3f}")
            else:
                # 传统分组的temperature优化
                for t in range(3):
                    idx = (token == t)
                    if idx.sum() >= 50:
                        key = f"t{t}"
                        temp = self._optimize_temperature_for_group(draft_logits[idx], y[idx], w[idx] if w is not None else None)
                        self.group_temperatures[key] = temp
                        print(f"  L1 {key}: temperature={temp:.3f}")
                
                # 其他层级的优化...（类似上面的逻辑）
            
            print(f"[GroupedIsotonicCalibrator] Temperature优化完成，共优化了{len(self.group_temperatures)}个组")

        self.is_fitted = True

    def predict_proba(self, features):
        proc = self._preprocess_features(features, fit_mode=False)
        c = proc['draft_conf']
        token, attn, pos = proc['token_type'], proc['attn_q'], proc['pos_bin']
        margin_q = proc['margin_q']

        out = np.zeros_like(c, dtype=np.float32)
        
        if self.use_hybrid_grouping:
            # 混合分组策略预测
            high_conf_mask = c >= self.confidence_threshold
            low_conf_mask = c < self.confidence_threshold
            
            # 高置信度：使用两层分组
            for t in range(3):
                for a in range(5):
                    mask = high_conf_mask & (token == t) & (attn == a)
                    if mask.sum() == 0:
                        continue
                    
                    # 高置信度回退：L2 -> L1 -> global
                    key2 = self._create_high_conf_key2(t, a)
                    key1 = self._create_high_conf_key1(t)
                    cal = (
                        self.high_conf_level2.get(key2)
                        or self.high_conf_level1.get(key1)
                        or self.global_calibrator
                    )
                    if cal is not None:
                        # 进行等渗校准
                        calibrated_probs = cal.predict(c[mask])
                        out[mask] = calibrated_probs
                    else:
                        out[mask] = self.global_mean
            
            # 低置信度：使用四层分组
            for t in range(3):
                for a in range(5):
                    for p in range(2):
                        for m in range(3):
                            mask = low_conf_mask & (token == t) & (attn == a) & (pos == p) & (margin_q == m)
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
        else:
            # 原始四层分组策略
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
        use_hybrid_grouping=True,
        confidence_threshold=0.7,
        min_samples_per_group=50
    )
    gi.fit(feats, soft, hard)
    metrics_gi = gi.evaluate(feats, soft, hard)
    gi_model_path = os.path.join(calibrator_dir, "grouped_isotonic_calibrator.pkl")
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


if __name__ == "__main__":
    # test_calibrator_training()
    # compare_before_after_calibration()
    calibrator_path = "/root/Speculative_decoding/calibration_data/test_calibrators"
    json_path = "/root/Speculative_decoding/calibration_data/chartqa_calib_isotonic_train_600_val_0_test_900_total_1500_temperature_0/training_calibration_data.json"
    calibrator_training(calibrator_path, json_path)
