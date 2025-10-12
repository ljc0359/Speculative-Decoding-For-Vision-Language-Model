"""
校准器实现：分组Isotonic和单调网络两种方案
用于校准draft model的confidence，使其更接近base model
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from typing import Dict, List, Tuple, Optional, Union
import pickle
import json
import os
from abc import ABC, abstractmethod

class BaseCalibrator(ABC):
    """校准器基类，定义通用接口"""
    
    def __init__(self):
        self.is_fitted = False
        self.feature_stats = {}
        
    @abstractmethod
    def fit(self, features: Dict[str, np.ndarray], 
            soft_labels: np.ndarray, 
            hard_labels: np.ndarray,
            sample_weights: Optional[np.ndarray] = None):
        """训练校准器"""
        pass
    
    @abstractmethod
    def predict_proba(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """预测校准后的概率"""
        pass
    
    def _preprocess_features(self, features: Dict[str, np.ndarray], fit_mode: bool = False) -> Dict[str, np.ndarray]:
        """特征预处理（仅使用 5 个指定特征）"""
        processed = {}
    
        # 1) token_category -> token_type 编码
        token_categories = features['token_category']
        if fit_mode:
            self.token_category_map = {cat: i for i, cat in enumerate(['content', 'func_punct', 'number'])}
        processed['token_type'] = np.array([self.token_category_map.get(cat, 0) for cat in token_categories])
    
        # 2) avg_visual_attention_intensity -> 三分位分箱 attn_q
        attn_intensity = features['avg_visual_attention_intensity']
        if fit_mode:
            self.attn_quantiles = np.quantile(attn_intensity, [0.33, 0.67])
        attn_q = np.zeros_like(attn_intensity, dtype=int)
        attn_q[attn_intensity <= self.attn_quantiles[0]] = 0
        attn_q[(attn_intensity > self.attn_quantiles[0]) & (attn_intensity <= self.attn_quantiles[1])] = 1
        attn_q[attn_intensity > self.attn_quantiles[1]] = 2
        processed['attn_q'] = attn_q
    
        # 3) tree_depth 生成 pos_bin（严格依赖 tree_depth，不再回退 tree_position）
        depth = features['tree_depth']
        processed['pos_bin'] = (depth > 2).astype(int)
        processed['tree_depth'] = depth
    
        # 4) 保留用于归一化的强度与可选 margin
        processed['avg_visual_attention_intensity'] = attn_intensity
        if 'draft_margin' in features:
            processed['draft_margin'] = features['draft_margin']
    
        # 5) draft_confidence -> draft_conf（只用该键，不再支持别名）
        processed['draft_conf'] = features['draft_confidence']
    
        return processed
    
    def _create_group_key(self, token_type: int, attn_q: int, pos_bin: int) -> str:
        """创建分组键"""
        return f"t{token_type}_a{attn_q}_p{pos_bin}"
    
    def evaluate(self, features: Dict[str, np.ndarray], 
                 soft_labels: np.ndarray, 
                 hard_labels: np.ndarray,
                 sample_weights: Optional[np.ndarray] = None) -> Dict[str, float]:
        """评估校准器性能"""
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before evaluation")
        
        pred_probs = self.predict_proba(features)
        
        # 计算各种指标
        metrics = {}
        
        # Brier Score (越小越好) - 使用hard_labels而不是soft_labels
        if sample_weights is not None:
            metrics['brier_score'] = np.average((pred_probs - hard_labels) ** 2, weights=sample_weights)
        else:
            metrics['brier_score'] = brier_score_loss(hard_labels, pred_probs)  # 修复：使用hard_labels
        
        # 额外计算与soft_labels的MSE（用于校准质量评估）
        if sample_weights is not None:
            metrics['soft_mse'] = np.average((pred_probs - soft_labels) ** 2, weights=sample_weights)
        else:
            metrics['soft_mse'] = np.mean((pred_probs - soft_labels) ** 2)
        
        # AUROC (排序能力)
        try:
            metrics['auroc'] = roc_auc_score(hard_labels, pred_probs, sample_weight=sample_weights)
        except:
            metrics['auroc'] = 0.5
        
        # ECE (Expected Calibration Error) - 使用hard_labels
        metrics['ece'] = self._compute_ece(pred_probs, hard_labels, sample_weights)
        
        # 决策加权ECE (wECE)
        if sample_weights is not None:
            metrics['wece'] = self._compute_ece(pred_probs, hard_labels, sample_weights)
        else:
            metrics['wece'] = metrics['ece']
        
        return metrics
    
    def _compute_ece(self, pred_probs: np.ndarray, true_labels: np.ndarray, 
                     sample_weights: Optional[np.ndarray] = None, n_bins: int = 10, equal_freq: bool = True) -> float:
        """计算 Expected Calibration Error（默认使用等频分桶，并去重边界）"""
        if equal_freq:
            qs = np.linspace(0, 1, n_bins + 1)
            boundaries = np.quantile(pred_probs, qs)
            boundaries = np.unique(boundaries)  # 去重，避免空桶
            if len(boundaries) < 2:
                return 0.0
            bin_lowers = boundaries[:-1]
            bin_uppers = boundaries[1:]
        else:
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        total_weight = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (pred_probs > bin_lower) & (pred_probs <= bin_upper)
            if in_bin.sum() == 0:
                continue
            if sample_weights is not None:
                bin_weights = sample_weights[in_bin]
                bin_weight = float(bin_weights.sum())
                accuracy_in_bin = float(np.average(true_labels[in_bin], weights=bin_weights))
                avg_confidence_in_bin = float(np.average(pred_probs[in_bin], weights=bin_weights))
            else:
                bin_weight = float(in_bin.sum())
                accuracy_in_bin = float(true_labels[in_bin].mean())
                avg_confidence_in_bin = float(pred_probs[in_bin].mean())
            ece += bin_weight * abs(avg_confidence_in_bin - accuracy_in_bin)
            total_weight += bin_weight
        return ece / total_weight if total_weight > 0 else 0.0
    
    def save(self, filepath: str):
        """保存校准器"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: str):
        """加载校准器"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class GroupedIsotonicCalibrator(BaseCalibrator):
    """分组Isotonic校准器
    
    按 token_type × attn_q × pos_bin 分成12组，每组内用PAV算法拟合单调函数
    """
    
    def __init__(self, min_samples_per_group: int = 50, out_of_bounds: str = 'clip'):
        super().__init__()
        self.min_samples_per_group = min_samples_per_group
        self.out_of_bounds = out_of_bounds
        self.group_calibrators = {}
        self.group_stats = {}
        
    def _fit_isotonic_binned(self, x: np.ndarray, y: np.ndarray, w: Optional[np.ndarray], n_bins: int = 15) -> IsotonicRegression:
        """在等频桶聚合后拟合 Isotonic，提升稳定性"""
        if len(x) == 0:
            raise ValueError("Empty data for isotonic fitting")
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]
        w_sorted = w[sort_idx] if w is not None else None
        
        # 等频分桶
        bin_edges = np.linspace(0, len(x_sorted), n_bins + 1).astype(int)
        x_binned, y_binned, w_binned = [], [], []
        for b in range(n_bins):
            start, end = bin_edges[b], bin_edges[b + 1]
            if end <= start:
                continue
            xb = x_sorted[start:end]
            yb = y_sorted[start:end]
            if w_sorted is not None:
                wb = w_sorted[start:end]
                w_binned.append(float(wb.sum()))
                y_binned.append(float(np.average(yb, weights=wb)))
            else:
                w_binned.append(float(len(yb)))
                y_binned.append(float(yb.mean()))
            x_binned.append(float(xb.mean()))
        
        iso_reg = IsotonicRegression(out_of_bounds=self.out_of_bounds, increasing=True)
        iso_reg.fit(np.array(x_binned), np.array(y_binned), sample_weight=np.array(w_binned))
        return iso_reg
    
    def fit(self, features: Dict[str, np.ndarray], 
            soft_labels: np.ndarray, 
            hard_labels: np.ndarray,
            sample_weights: Optional[np.ndarray] = None):
        """训练分组Isotonic校准器（x = draft_conf），含层级回退与等频聚合"""
        processed = self._preprocess_features(features, fit_mode=True)
        c = processed['draft_conf']  # 单调自变量：top-1 置信度
        token_type = processed['token_type']
        attn_q = processed['attn_q']
        pos_bin = processed['pos_bin']
        
        n_samples = len(soft_labels)
        if n_samples != len(c):
            raise ValueError(f"Length mismatch: soft_labels({n_samples}) vs features({len(c)})")
        
        weights = sample_weights if sample_weights is not None else None
        
        # 先训练全局 Isotonic 作为最终回退
        try:
            self.global_calibrator = self._fit_isotonic_binned(c, soft_labels, weights, n_bins=15)
            self.global_mean = float(soft_labels.mean())
        except Exception as e:
            print(f"Warning: global isotonic failed ({e}), using global mean")
            self.global_calibrator = None
            self.global_mean = float(soft_labels.mean())
        
        # 层级容器
        self.level1_calibrators = {}  # t
        self.level2_calibrators = {}  # t×a
        self.group_calibrators = {}   # t×a×p
        self.group_stats = {}
        
        # Level 1: 按 token_type
        for t in range(3):
            idx = np.where(token_type == t)[0]
            key = f"t{t}"
            if len(idx) >= self.min_samples_per_group:
                iso = self._fit_isotonic_binned(c[idx], soft_labels[idx], weights[idx] if weights is not None else None, n_bins=15)
                self.level1_calibrators[key] = iso
            else:
                self.level1_calibrators[key] = None
        
        # Level 2: 按 token_type × attn_q
        for t in range(3):
            for a in range(3):
                idx = np.where((token_type == t) & (attn_q == a))[0]
                key = f"t{t}_a{a}"
                if len(idx) >= self.min_samples_per_group:
                    iso = self._fit_isotonic_binned(c[idx], soft_labels[idx], weights[idx] if weights is not None else None, n_bins=15)
                    self.level2_calibrators[key] = iso
                else:
                    self.level2_calibrators[key] = None
        
        # Level 3: 按 token_type × attn_q × pos_bin
        for t in range(3):
            for a in range(3):
                for p in range(2):
                    group_key = self._create_group_key(t, a, p)
                    idx = np.where((token_type == t) & (attn_q == a) & (pos_bin == p))[0]
                    self.group_stats[group_key] = {'n_samples': int(len(idx))}
                    if len(idx) < self.min_samples_per_group:
                        # 稀疏组：不训练，留待层级回退
                        self.group_calibrators[group_key] = None
                        continue
                    # 组内等频桶聚合拟合
                    iso = self._fit_isotonic_binned(c[idx], soft_labels[idx], weights[idx] if weights is not None else None, n_bins=15)
                    self.group_calibrators[group_key] = iso
        
        self.is_fitted = True
        fitted_groups = sum(v is not None for v in self.group_calibrators.values())
        print(f"Fitted {fitted_groups} fine-grained groups; hierarchical fallback enabled")
    
    def predict_proba(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """预测校准后的概率（层级回退 + 向量化组内预测）"""
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before prediction")
        
        processed = self._preprocess_features(features, fit_mode=False)
        c = processed['draft_conf']
        token_type = processed['token_type']
        attn_q = processed['attn_q']
        pos_bin = processed['pos_bin']
        
        n = len(c)
        preds = np.zeros(n, dtype=np.float32)
        
        # 逐组向量化预测
        for t in range(3):
            for a in range(3):
                for p in range(2):
                    group_key = self._create_group_key(t, a, p)
                    mask = (token_type == t) & (attn_q == a) & (pos_bin == p)
                    if mask.sum() == 0:
                        continue
                    indices = np.where(mask)[0]
                    
                    # 层级选择最优可用校准器
                    calibrator = None
                    if self.group_calibrators.get(group_key) is not None:
                        calibrator = self.group_calibrators[group_key]
                    elif self.level2_calibrators.get(f"t{t}_a{a}") is not None:
                        calibrator = self.level2_calibrators[f"t{t}_a{a}"]
                    elif self.level1_calibrators.get(f"t{t}") is not None:
                        calibrator = self.level1_calibrators[f"t{t}"]
                    else:
                        calibrator = self.global_calibrator
                    
                    if calibrator is not None:
                        preds[indices] = calibrator.predict(c[indices])
                    else:
                        # 最终回退：全局均值
                        preds[indices] = self.global_mean
        
        preds = np.clip(preds, 1e-4, 1 - 1e-4)
        return preds


class MonotonicNetworkCalibrator(BaseCalibrator):
    """单调网络校准器
    
    使用单调MLP架构确保对confidence的单调性
    logit(p_tilde) = β(φ) + Σ_k γ_k(φ) * s_k(c)
    其中 γ_k(φ) ≥ 0 确保对 c 的单调性
    """
    
    def __init__(self, hidden_dim: int = 64, n_basis: int = 10, 
                 learning_rate: float = 1e-3, max_epochs: int = 1000,
                 patience: int = 50, device: str = 'cuda'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_basis = n_basis
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.patience = patience
        # 默认用 CPU，更稳且避免无 GPU 的警告
        self.device = device
        self.model = None
        # 训练期统计，用于固定化 z 的标准化
        self.z_mean = None
        self.z_std = None
    
    def _create_basis_functions(self, conf: np.ndarray) -> np.ndarray:
        """创建对 c 单调的基函数：logit(c) 并做标准化，提升数值稳定性"""
        c = np.clip(conf, 1e-4, 1 - 1e-4)
        z = np.log(c) - np.log(1.0 - c)  # logit(c)
        # 若训练期已统计均值方差，则使用同一标准化；否则估计并记录
        if self.z_mean is None or self.z_std is None:
            self.z_mean = float(z.mean())
            self.z_std = float(z.std() + 1e-6)
        z = (z - self.z_mean) / self.z_std
        knots = np.linspace(z.min(), z.max(), self.n_basis)
        basis = np.maximum(0.0, z[:, None] - knots[None, :])
        return basis
    
    def fit(self, features: Dict[str, np.ndarray], 
            soft_labels: np.ndarray, 
            hard_labels: np.ndarray,
            sample_weights: Optional[np.ndarray] = None):
        """训练单调网络校准器：φ 仅用 5 个特征的派生量"""
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        processed = self._preprocess_features(features, fit_mode=True)
    
        # 组装 φ：token_type, attn_q, pos_bin + tree_depth_norm + attn_norm + 可选 margin_norm
        token_onehot = np.eye(3)[processed['token_type']]
        attn_onehot = np.eye(3)[processed['attn_q']]
        pos_onehot = np.eye(2)[processed['pos_bin']]
    
        td = processed['tree_depth']
        td_min, td_max = td.min(), td.max()
        tree_norm = (td - td_min) / (td_max - td_min + 1e-8)
    
        ai = processed['avg_visual_attention_intensity']
        ai_min, ai_max = ai.min(), ai.max()
        attn_norm = (ai - ai_min) / (ai_max - ai_min + 1e-8)
    
        parts = [token_onehot, attn_onehot, pos_onehot, tree_norm.reshape(-1, 1), attn_norm.reshape(-1, 1)]
        if 'draft_margin' in processed:
            m = processed['draft_margin']; m_min, m_max = m.min(), m.max()
            margin_norm = (m - m_min) / (m_max - m_min + 1e-8)
            parts.append(margin_norm.reshape(-1, 1))
    
        phi = np.concatenate(parts, axis=1)
        self.phi_dim = int(phi.shape[1])
    
        # 单调基函数建立在 draft_conf
        conf = processed['draft_conf']
        basis = self._create_basis_functions(conf)
    
        # 张量化与训练（其余逻辑不变）
        phi_tensor = torch.FloatTensor(phi).to(self.device)
        basis_tensor = torch.FloatTensor(basis).to(self.device)
        labels_tensor = torch.FloatTensor(soft_labels).to(self.device)
        weights_tensor = torch.FloatTensor(sample_weights).to(self.device) if sample_weights is not None else None
    
        self.model = MonotonicMLP(phi_dim=self.phi_dim, n_basis=self.n_basis, hidden_dim=self.hidden_dim).to(self.device)
    
        with torch.no_grad():
            mu = float(labels_tensor.clamp(1e-4, 1-1e-4).mean().item())
            b = float(np.log(mu) - np.log(1.0 - mu))
            try:
                last_linear = self.model.beta_net[-1]
                nn.init.zeros_(last_linear.weight)
                nn.init.constant_(last_linear.bias, b)
            except Exception:
                pass
    
        self._train_model(phi_tensor, basis_tensor, labels_tensor, weights_tensor)
        self.is_fitted = True
        
    def _train_model(self, phi: torch.Tensor, basis: torch.Tensor, 
                     labels: torch.Tensor, weights: Optional[torch.Tensor]):
        """稳健训练：AdamW + BCE-with-logits + 梯度裁剪"""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=max(self.learning_rate, 1e-4), weight_decay=1e-4)
        best_loss = float('inf'); patience_left = self.patience
        n = phi.size(0); idx = torch.randperm(n)
        # 简单 90/10 切分作为“验证”
        split = max(int(n * 0.9), 1)
        train_idx, val_idx = idx[:split], idx[split:]
        phi_tr, basis_tr, y_tr = phi[train_idx], basis[train_idx], labels[train_idx]
        phi_val, basis_val, y_val = phi[val_idx], basis[val_idx], labels[val_idx]
        w_tr = weights[train_idx] if weights is not None else None
        w_val = weights[val_idx] if weights is not None else None

        for epoch in range(self.max_epochs):
            self.model.train()
            optimizer.zero_grad()
            # 前向：模型通常输出 logits；若输出在 [0,1]，则转为 logits 再做 BCE-with-logits
            out_tr = self.model(phi_tr, basis_tr)
            if out_tr.dim() == 2 and out_tr.shape[1] == 1:
                out_tr = out_tr.squeeze(1)
            if torch.all((out_tr >= 0.0) & (out_tr <= 1.0)):
                logits_tr = torch.logit(out_tr.clamp(1e-4, 1-1e-4))
            else:
                logits_tr = out_tr
            loss_tr = F.binary_cross_entropy_with_logits(logits_tr, y_tr, weight=w_tr, reduction='mean')
            loss_tr.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            # 验证
            self.model.eval()
            with torch.no_grad():
                out_val = self.model(phi_val, basis_val)
                if out_val.dim() == 2 and out_val.shape[1] == 1:
                    out_val = out_val.squeeze(1)
                if torch.all((out_val >= 0.0) & (out_val <= 1.0)):
                    logits_val = torch.logit(out_val.clamp(1e-4, 1-1e-4))
                else:
                    logits_val = out_val
                val_loss = F.binary_cross_entropy_with_logits(logits_val, y_val, weight=w_val, reduction='mean')

            print(f"Epoch {epoch}, Train Loss: {loss_tr.item():.4f}, Val Loss: {val_loss.item():.4f}")
            # 早停逻辑：修复变量名
            if val_loss.item() + 1e-6 < best_loss:
                best_loss = val_loss.item()
                patience_left = self.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    # 若起始就异常饱和，后续几轮会自动回落；否则触发早停保护
                    break
    
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Train Loss: {loss_tr.item():.4f}, Val Loss: {val_loss.item():.4f}")
        
        # 加载最佳模型
        if hasattr(self, 'best_state_dict'):
            self.model.load_state_dict(self.best_state_dict)
        
    def predict_proba(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """预测校准后概率：与 fit 相同的 5 特征派生逻辑"""
        processed = self._preprocess_features(features, fit_mode=False)
    
        conf = processed['draft_conf']
        basis = self._create_basis_functions(conf)
    
        token_onehot = np.eye(3)[processed['token_type']]
        attn_onehot = np.eye(3)[processed['attn_q']]
        pos_onehot = np.eye(2)[processed['pos_bin']]
    
        td = processed['tree_depth']; td_min, td_max = td.min(), td.max()
        tree_norm = (td - td_min) / (td_max - td_min + 1e-8)
    
        ai = processed['avg_visual_attention_intensity']; ai_min, ai_max = ai.min(), ai.max()
        attn_norm = (ai - ai_min) / (ai_max - ai_min + 1e-8)
    
        parts = [token_onehot, attn_onehot, pos_onehot, tree_norm.reshape(-1, 1), attn_norm.reshape(-1, 1)]
        if 'draft_margin' in processed:
            m = processed['draft_margin']; m_min, m_max = m.min(), m.max()
            margin_norm = (m - m_min) / (m_max - m_min + 1e-8)
            parts.append(margin_norm.reshape(-1, 1))
    
        phi = np.concatenate(parts, axis=1)
    
        phi_t = torch.FloatTensor(phi).to(self.device)
        basis_t = torch.FloatTensor(basis).to(self.device)
        self.model.eval()
        with torch.no_grad():
            out = self.model(phi_t, basis_t)
            if out.dim() == 2 and out.shape[1] == 1:
                out = out.squeeze(1)
            if torch.all((out >= 0.0) & (out <= 1.0)):
                logits = torch.logit(out.clamp(1e-4, 1-1e-4))
            else:
                logits = out
            probs = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4).detach().cpu().numpy()
        return probs


class MonotonicMLP(nn.Module):
    """单调MLP模型
    
    logit(p_tilde) = β(φ) + Σ_k γ_k(φ) * s_k(c)
    其中 γ_k(φ) ≥ 0 通过 softplus 确保
    """
    
    def __init__(self, phi_dim: int, n_basis: int, hidden_dim: int):
        super().__init__()
        self.phi_dim = phi_dim
        self.n_basis = n_basis
        self.hidden_dim = hidden_dim
        
        # β(φ) 网络 - 偏置项
        self.beta_net = nn.Sequential(
            nn.Linear(phi_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # γ_k(φ) 网络 - 单调系数（确保非负）
        self.gamma_net = nn.Sequential(
            nn.Linear(phi_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_basis)
        )
        
    def forward(self, phi: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        """
        Args:
            phi: 特征向量 (batch_size, phi_dim)
            basis: 基函数值 (batch_size, n_basis)
        
        Returns:
            logits: (batch_size,)
        """
        # 计算偏置项 β(φ)
        beta = self.beta_net(phi).squeeze(-1)  # (batch_size,)
        
        # 计算单调系数 γ_k(φ)，使用 softplus 确保非负
        gamma = F.softplus(self.gamma_net(phi))  # (batch_size, n_basis)
        
        # 计算单调项 Σ_k γ_k(φ) * s_k(c)
        monotonic_term = (gamma * basis).sum(dim=1)  # (batch_size,)
        
        # 最终logit
        logits = beta + monotonic_term
        
        return logits


def load_calibration_data(data_path: str) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """加载校准数据（简化版：直接从 JSON candidate_calibration_data 取字段，或直接从 NPZ 的同名键读取）"""
    import numpy as np
    if data_path.endswith('.json'):
        import json
        with open(data_path, 'r') as f:
            raw = json.load(f)
        data_list = raw['candidate_calibration_data'] if isinstance(raw, dict) else raw
        features = {}
        keys = [
            'draft_confidence',
            'tree_depth',
            'avg_visual_attention_intensity',
            'draft_margin',
            'token_category',
        ]
        if len(data_list) > 0:
            for k in keys:
                if k in data_list[0]:
                    features[k] = np.array([item[k] for item in data_list])
        soft_labels = np.array([item['base_confidence'] for item in data_list])
        hard_labels = np.array([item['base_top1_token'] for item in data_list])
        return features, soft_labels, hard_labels
    elif data_path.endswith('.npz'):
        data = np.load(data_path)
        features = {}
        for k in [
            'draft_confidence',
            'tree_depth',
            'avg_visual_attention_intensity',
            'draft_margin',
            'token_category',
        ]:
            if k in data:
                features[k] = data[k]
        soft_labels = data['soft_labels'] if 'soft_labels' in data else data['base_confidence']
        hard_labels = data['hard_labels'] if 'hard_labels' in data else data['base_top1_token']
        return features, soft_labels, hard_labels
    else:
        raise ValueError(f"Unsupported file format: {data_path}")

def calibrator_training(path: str, json_path: str):
    """使用 JSON 训练数据测试校准器训练流程（简化版）
    
    参数:
        path: 训练结果（模型与指标）保存的目标文件夹
        json_path: 训练数据的 JSON 文件路径
    """
    import os
    import numpy as np
    import json

    print("=" * 60)
    print("Testing Calibrator Training Pipeline (JSON)")
    print("=" * 60)

    # 使用外部提供的 JSON 训练集路径
    print(f"Using JSON training data: {json_path}")

    # 加载数据
    features, soft_labels, hard_labels = load_calibration_data(json_path)
    n_samples = len(soft_labels)
    print(f"Loaded {n_samples} samples")
    print(f"Feature keys: {list(features.keys())}")

    # 校准前：使用 draft_confidence 与 soft label 的拟合程度（soft MSE）
    if 'draft_confidence' in features:
        draft_conf = np.clip(features['draft_confidence'], 1e-6, 1 - 1e-6)
        before_soft_mse = np.mean((draft_conf - soft_labels) ** 2)
        print(f"Before calibration soft MSE (draft_conf vs base_confidence): {before_soft_mse:.6f}")
    else:
        draft_conf = None
        print("Warning: 'draft_confidence' not found in features; skipping BEFORE soft MSE.")

    # 分组 Isotonic 校准器
    print("\nTraining GroupedIsotonicCalibrator...")
    isotonic_cal = GroupedIsotonicCalibrator(min_samples_per_group=10)
    isotonic_cal.fit(features, soft_labels, hard_labels)
    iso_pred = isotonic_cal.predict_proba(features)
    iso_soft_mse = np.mean((iso_pred - soft_labels) ** 2)
    print(f"After (Isotonic) soft MSE: {iso_soft_mse:.6f}")

    # 单调网络校准器
    print("\nTraining MonotonicNetworkCalibrator...")
    monotonic_cal = MonotonicNetworkCalibrator(hidden_dim=32, n_basis=5, max_epochs=100, device='cpu')
    monotonic_cal.fit(features, soft_labels, hard_labels)
    mono_pred = monotonic_cal.predict_proba(features)
    mono_soft_mse = np.mean((mono_pred - soft_labels) ** 2)
    print(f"After (Monotonic) soft MSE: {mono_soft_mse:.6f}")
    # 打印简单的指标
    def mse(a, b): return float(np.mean((a - b) ** 2))
    def brier(y_true, y_prob): 
        return float(np.mean((y_prob - y_true) ** 2))
    print("\nMetrics summary:")
    if draft_conf is not None:
        print(f"  BEFORE brier: {brier(hard_labels, draft_conf):.6f}")
    print(f"  AFTER (Isotonic) brier: {brier(hard_labels, iso_pred):.6f}")
    print(f"  AFTER (Monotonic) brier: {brier(hard_labels, mono_pred):.6f}")
    # 训练结果保存到指定路径
    os.makedirs(path, exist_ok=True)
    iso_path = os.path.join(path, "grouped_isotonic_calibrator.pkl")
    mono_path = os.path.join(path, "monotonic_network_calibrator.pkl")
    isotonic_cal.save(iso_path)
    monotonic_cal.save(mono_path)
    metrics = {
        "before_soft_mse": float(before_soft_mse) if draft_conf is not None else None,
        "isotonic_soft_mse": float(iso_soft_mse),
        "monotonic_soft_mse": float(mono_soft_mse),
        "before_brier": float(brier(hard_labels, draft_conf)) if draft_conf is not None else None,
        "isotonic_brier": float(brier(hard_labels, iso_pred)),
        "monotonic_brier": float(brier(hard_labels, mono_pred)),
        "n_samples": int(n_samples),
        "feature_keys": list(features.keys()),
    }
    with open(os.path.join(path, "training_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("\n✓ Calibrator Training Pipeline (JSON) completed.")
    # 返回两个训练好的模型
    return isotonic_cal, monotonic_cal

if __name__ == "__main__":
    # 运行测试
    save_dir = "/root/Speculative_decoding/calibration_data/chartqa_train_40_val_0_test_60_total_100/calibrators"
    json_path = "/root/Speculative_decoding/calibration_data/chartqa_train_40_val_0_test_60_total_100/non_calibrator/training_calibration_data.json"
    calibrator_training(save_dir, json_path)

    # 运行对比
    # compare_before_after_calibration()