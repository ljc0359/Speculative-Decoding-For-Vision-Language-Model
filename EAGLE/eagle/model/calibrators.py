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
        """特征预处理和工程"""
        processed = {}
        
        # 1. Token类别编码
        token_categories = features['token_category']
        if fit_mode:
            self.token_category_map = {cat: i for i, cat in enumerate(['content', 'func_punct', 'number'])}
        
        processed['token_type'] = np.array([self.token_category_map.get(cat, 0) for cat in token_categories])
        
        # 2. 注意力强度三分位
        attn_intensity = features['avg_visual_attention_intensity']
        if fit_mode:
            self.attn_quantiles = np.quantile(attn_intensity, [0.33, 0.67])
        
        attn_q = np.zeros_like(attn_intensity, dtype=int)
        attn_q[attn_intensity <= self.attn_quantiles[0]] = 0  # Q1
        attn_q[(attn_intensity > self.attn_quantiles[0]) & (attn_intensity <= self.attn_quantiles[1])] = 1  # Q2
        attn_q[attn_intensity > self.attn_quantiles[1]] = 2  # Q3
        processed['attn_q'] = attn_q
        
        # 3. 位置分箱
        tree_position = features['tree_position']
        pos_bin = np.where(tree_position <= 2, 0, 1)  # P1_2: 0, P3plus: 1
        processed['pos_bin'] = pos_bin
        
        # 4. 保留原始特征
        processed['draft_margin'] = features['draft_margin']
        processed['tree_position'] = tree_position
        processed['avg_visual_attention_intensity'] = attn_intensity
        
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
        
        # Brier Score (越小越好)
        if sample_weights is not None:
            metrics['brier_score'] = np.average((pred_probs - soft_labels) ** 2, weights=sample_weights)
        else:
            metrics['brier_score'] = brier_score_loss(soft_labels, pred_probs)
        
        # AUROC (排序能力)
        try:
            metrics['auroc'] = roc_auc_score(hard_labels, pred_probs, sample_weight=sample_weights)
        except:
            metrics['auroc'] = 0.5
        
        # ECE (Expected Calibration Error)
        metrics['ece'] = self._compute_ece(pred_probs, hard_labels, sample_weights)
        
        # 决策加权ECE (wECE)
        if sample_weights is not None:
            metrics['wece'] = self._compute_ece(pred_probs, hard_labels, sample_weights)
        else:
            metrics['wece'] = metrics['ece']
        
        return metrics
    
    def _compute_ece(self, pred_probs: np.ndarray, true_labels: np.ndarray, 
                     sample_weights: Optional[np.ndarray] = None, n_bins: int = 10) -> float:
        """计算Expected Calibration Error"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        total_weight = 0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (pred_probs > bin_lower) & (pred_probs <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                if sample_weights is not None:
                    bin_weights = sample_weights[in_bin]
                    bin_weight = bin_weights.sum()
                    accuracy_in_bin = np.average(true_labels[in_bin], weights=bin_weights)
                    avg_confidence_in_bin = np.average(pred_probs[in_bin], weights=bin_weights)
                else:
                    bin_weight = in_bin.sum()
                    accuracy_in_bin = true_labels[in_bin].mean()
                    avg_confidence_in_bin = pred_probs[in_bin].mean()
                
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
        
    def fit(self, features: Dict[str, np.ndarray], 
            soft_labels: np.ndarray, 
            hard_labels: np.ndarray,
            sample_weights: Optional[np.ndarray] = None):
        """训练分组Isotonic校准器"""
        
        # 预处理特征
        processed_features = self._preprocess_features(features, fit_mode=True)
        
        # 按组训练
        n_samples = len(soft_labels)
        
        for token_type in range(3):  # content, func_punct, number
            for attn_q in range(3):  # Q1, Q2, Q3
                for pos_bin in range(2):  # P1_2, P3plus
                    group_key = self._create_group_key(token_type, attn_q, pos_bin)
                    
                    # 找到属于当前组的样本
                    group_mask = (
                        (processed_features['token_type'] == token_type) &
                        (processed_features['attn_q'] == attn_q) &
                        (processed_features['pos_bin'] == pos_bin)
                    )
                    
                    group_indices = np.where(group_mask)[0]
                    
                    if len(group_indices) < self.min_samples_per_group:
                        print(f"Warning: Group {group_key} has only {len(group_indices)} samples, "
                              f"minimum required: {self.min_samples_per_group}")
                        # 使用全局平均作为fallback
                        self.group_calibrators[group_key] = None
                        self.group_stats[group_key] = {
                            'n_samples': len(group_indices),
                            'fallback_prob': soft_labels.mean()
                        }
                        continue
                    
                    # 提取组内数据
                    group_margins = processed_features['draft_margin'][group_indices]
                    group_soft_labels = soft_labels[group_indices]
                    group_weights = sample_weights[group_indices] if sample_weights is not None else None
                    
                    # 按margin排序（Isotonic回归需要排序的输入）
                    sort_idx = np.argsort(group_margins)
                    sorted_margins = group_margins[sort_idx]
                    sorted_labels = group_soft_labels[sort_idx]
                    sorted_weights = group_weights[sort_idx] if group_weights is not None else None
                    
                    # 训练Isotonic回归
                    iso_reg = IsotonicRegression(
                        out_of_bounds=self.out_of_bounds,
                        increasing=True
                    )
                    
                    if sorted_weights is not None:
                        iso_reg.fit(sorted_margins, sorted_labels, sample_weight=sorted_weights)
                    else:
                        iso_reg.fit(sorted_margins, sorted_labels)
                    
                    self.group_calibrators[group_key] = iso_reg
                    self.group_stats[group_key] = {
                        'n_samples': len(group_indices),
                        'margin_range': (sorted_margins.min(), sorted_margins.max()),
                        'label_range': (sorted_labels.min(), sorted_labels.max())
                    }
        
        self.is_fitted = True
        print(f"Fitted {len([k for k, v in self.group_calibrators.items() if v is not None])} groups successfully")
        
    def predict_proba(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """预测校准后的概率"""
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before prediction")
        
        processed_features = self._preprocess_features(features, fit_mode=False)
        n_samples = len(features['draft_margin'])
        predictions = np.zeros(n_samples)
        
        for i in range(n_samples):
            token_type = processed_features['token_type'][i]
            attn_q = processed_features['attn_q'][i]
            pos_bin = processed_features['pos_bin'][i]
            margin = processed_features['draft_margin'][i]
            
            group_key = self._create_group_key(token_type, attn_q, pos_bin)
            
            if group_key in self.group_calibrators and self.group_calibrators[group_key] is not None:
                # 使用组内校准器
                predictions[i] = self.group_calibrators[group_key].predict([margin])[0]
            else:
                # 使用fallback
                if group_key in self.group_stats:
                    predictions[i] = self.group_stats[group_key]['fallback_prob']
                else:
                    predictions[i] = 0.5  # 默认值
        
        # 确保预测值在合理范围内
        predictions = np.clip(predictions, 1e-4, 1 - 1e-4)
        
        return predictions


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
        self.device = device
        self.model = None
        
    def _create_basis_functions(self, margins: np.ndarray) -> np.ndarray:
        """创建基函数 s_k(c)，按 c 递增"""
        margin_min, margin_max = margins.min(), margins.max()
        
        # 使用分段线性基函数
        knots = np.linspace(margin_min, margin_max, self.n_basis)
        basis = np.zeros((len(margins), self.n_basis))
        
        for i, knot in enumerate(knots):
            basis[:, i] = np.maximum(0, margins - knot)
        
        return basis
    
    def fit(self, features: Dict[str, np.ndarray], 
            soft_labels: np.ndarray, 
            hard_labels: np.ndarray,
            sample_weights: Optional[np.ndarray] = None):
        """训练单调网络校准器"""
        
        # 预处理特征
        processed_features = self._preprocess_features(features, fit_mode=True)
        
        # 准备输入特征
        # φ = [token_type_onehot, attn_q_onehot, pos_bin_onehot, tree_position_norm, attn_intensity_norm]
        n_samples = len(soft_labels)
        
        # One-hot编码
        token_onehot = np.eye(3)[processed_features['token_type']]  # (n, 3)
        attn_onehot = np.eye(3)[processed_features['attn_q']]       # (n, 3)
        pos_onehot = np.eye(2)[processed_features['pos_bin']]       # (n, 2)
        
        # 归一化连续特征
        tree_pos_norm = (processed_features['tree_position'] - processed_features['tree_position'].min()) / \
                       (processed_features['tree_position'].max() - processed_features['tree_position'].min() + 1e-8)
        attn_norm = (processed_features['avg_visual_attention_intensity'] - 
                    processed_features['avg_visual_attention_intensity'].min()) / \
                   (processed_features['avg_visual_attention_intensity'].max() - 
                    processed_features['avg_visual_attention_intensity'].min() + 1e-8)
        
        # 组合特征向量 φ
        phi = np.concatenate([
            token_onehot,           # 3 dims
            attn_onehot,           # 3 dims  
            pos_onehot,            # 2 dims
            tree_pos_norm.reshape(-1, 1),    # 1 dim
            attn_norm.reshape(-1, 1)         # 1 dim
        ], axis=1)  # Total: 10 dims
        
        # 创建基函数
        margins = processed_features['draft_margin']
        basis = self._create_basis_functions(margins)
        
        # 保存用于预测的统计信息
        self.margin_min = margins.min()
        self.margin_max = margins.max()
        self.tree_pos_min = processed_features['tree_position'].min()
        self.tree_pos_max = processed_features['tree_position'].max()
        self.attn_min = processed_features['avg_visual_attention_intensity'].min()
        self.attn_max = processed_features['avg_visual_attention_intensity'].max()
        
        # 转换为PyTorch张量
        phi_tensor = torch.FloatTensor(phi).to(self.device)
        basis_tensor = torch.FloatTensor(basis).to(self.device)
        labels_tensor = torch.FloatTensor(soft_labels).to(self.device)
        
        if sample_weights is not None:
            weights_tensor = torch.FloatTensor(sample_weights).to(self.device)
        else:
            weights_tensor = None
        
        # 创建模型
        self.model = MonotonicMLP(
            phi_dim=phi.shape[1], 
            n_basis=self.n_basis, 
            hidden_dim=self.hidden_dim
        ).to(self.device)

        # 训练
        self._train_model(phi_tensor, basis_tensor, labels_tensor, weights_tensor)
        
        self.is_fitted = True
        
    def _train_model(self, phi: torch.Tensor, basis: torch.Tensor, 
                     labels: torch.Tensor, weights: Optional[torch.Tensor]):
        """训练模型"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        best_loss = float('inf')
        patience_counter = 0
        
        # 划分训练/验证集
        n_samples = len(labels)
        indices = np.random.permutation(n_samples)
        split_idx = int(0.8 * n_samples)
        
        train_idx = indices[:split_idx]
        val_idx = indices[split_idx:]
        
        for epoch in range(self.max_epochs):
            # 训练
            self.model.train()
            optimizer.zero_grad()
            
            train_phi = phi[train_idx]
            train_basis = basis[train_idx]
            train_labels = labels[train_idx]
            train_weights = weights[train_idx] if weights is not None else None
            
            logits = self.model(train_phi, train_basis)
            probs = torch.sigmoid(logits)
            
            # Brier Score Loss
            loss = F.mse_loss(probs, train_labels, reduction='none')
            
            if train_weights is not None:
                loss = (loss * train_weights).mean()
            else:
                loss = loss.mean()
            
            loss.backward()
            optimizer.step()
            
            # 验证
            if len(val_idx) > 0:
                self.model.eval()
                with torch.no_grad():
                    val_phi = phi[val_idx]
                    val_basis = basis[val_idx]
                    val_labels = labels[val_idx]
                    val_weights = weights[val_idx] if weights is not None else None
                    
                    val_logits = self.model(val_phi, val_basis)
                    val_probs = torch.sigmoid(val_logits)
                    
                    val_loss = F.mse_loss(val_probs, val_labels, reduction='none')
                    if val_weights is not None:
                        val_loss = (val_loss * val_weights).mean()
                    else:
                        val_loss = val_loss.mean()
                    
                    if val_loss < best_loss:
                        best_loss = val_loss
                        patience_counter = 0
                        # 保存最佳模型
                        self.best_state_dict = self.model.state_dict().copy()
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= self.patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
        
        # 加载最佳模型
        if hasattr(self, 'best_state_dict'):
            self.model.load_state_dict(self.best_state_dict)
        
    def predict_proba(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """预测校准后的概率"""
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before prediction")
        
        processed_features = self._preprocess_features(features, fit_mode=False)
        
        # 准备输入特征（与训练时相同的处理）
        token_onehot = np.eye(3)[processed_features['token_type']]
        attn_onehot = np.eye(3)[processed_features['attn_q']]
        pos_onehot = np.eye(2)[processed_features['pos_bin']]
        
        tree_pos_norm = (processed_features['tree_position'] - self.tree_pos_min) / \
                       (self.tree_pos_max - self.tree_pos_min + 1e-8)
        attn_norm = (processed_features['avg_visual_attention_intensity'] - self.attn_min) / \
                   (self.attn_max - self.attn_min + 1e-8)
        
        phi = np.concatenate([
            token_onehot,
            attn_onehot,
            pos_onehot,
            tree_pos_norm.reshape(-1, 1),
            attn_norm.reshape(-1, 1)
        ], axis=1)
        
        # 创建基函数
        margins = processed_features['draft_margin']
        basis = self._create_basis_functions(margins)
        
        # 转换为张量并预测
        phi_tensor = torch.FloatTensor(phi).to(self.device)
        basis_tensor = torch.FloatTensor(basis).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            logits = self.model(phi_tensor, basis_tensor)
            probs = torch.sigmoid(logits)
        
        predictions = probs.cpu().numpy()
        predictions = np.clip(predictions, 1e-4, 1 - 1e-4)
        
        return predictions


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


# 工具函数
def load_calibration_data(data_path: str) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """加载校准数据"""
    if data_path.endswith('.npz'):
        data = np.load(data_path)
        features = {key: data[key] for key in data.files if key not in ['soft_labels', 'hard_labels']}
        soft_labels = data['soft_labels']
        hard_labels = data['hard_labels']
    elif data_path.endswith('.json'):
        import json
        with open(data_path, 'r') as f:
            data_list = json.load(f)
        
        # 转换为numpy数组
        features = {}
        for key in ['tree_position', 'draft_margin', 'avg_visual_attention_intensity', 'token_category']:
            features[key] = np.array([item[key] for item in data_list])
        
        soft_labels = np.array([item['base_confidence'] for item in data_list])
        hard_labels = np.array([item['base_top1_token'] for item in data_list])
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    return features, soft_labels, hard_labels


def compare_calibrators(data_path: str, test_split: float = 0.2):
    """比较两种校准器的性能"""
    
    # 加载数据
    features, soft_labels, hard_labels = load_calibration_data(data_path)
    
    # 划分训练/测试集
    n_samples = len(soft_labels)
    indices = np.random.permutation(n_samples)
    split_idx = int((1 - test_split) * n_samples)
    
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    
    train_features = {k: v[train_idx] for k, v in features.items()}
    test_features = {k: v[test_idx] for k, v in features.items()}
    
    train_soft = soft_labels[train_idx]
    train_hard = hard_labels[train_idx]
    test_soft = soft_labels[test_idx]
    test_hard = hard_labels[test_idx]
    
    # 训练分组Isotonic校准器
    print("Training Grouped Isotonic Calibrator...")
    isotonic_cal = GroupedIsotonicCalibrator()
    isotonic_cal.fit(train_features, train_soft, train_hard)
    
    # 训练单调网络校准器
    print("Training Monotonic Network Calibrator...")
    monotonic_cal = MonotonicNetworkCalibrator()
    monotonic_cal.fit(train_features, train_soft, train_hard)
    
    # 评估
    print("\nEvaluating on test set...")
    
    isotonic_metrics = isotonic_cal.evaluate(test_features, test_soft, test_hard)
    monotonic_metrics = monotonic_cal.evaluate(test_features, test_soft, test_hard)
    
    print("\nGrouped Isotonic Results:")
    for metric, value in isotonic_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nMonotonic Network Results:")
    for metric, value in monotonic_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return isotonic_cal, monotonic_cal, isotonic_metrics, monotonic_metrics


if __name__ == "__main__":
    # 示例用法
    data_path = "/root/Speculative_decoding/calibration_data/final_calibration_data.npz"
    
    try:
        isotonic_cal, monotonic_cal, iso_metrics, mono_metrics = compare_calibrators(data_path)
        
        # 保存校准器
        isotonic_cal.save("/root/Speculative_decoding/Speculative-Decoding-For-Vision-Language-Model/EAGLE/eagle/model/grouped_isotonic_calibrator.pkl")
        monotonic_cal.save("/root/Speculative_decoding/Speculative-Decoding-For-Vision-Language-Model/EAGLE/eagle/model/monotonic_network_calibrator.pkl")
        
        print("\nCalibrators saved successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure the calibration data file exists and has the correct format.")