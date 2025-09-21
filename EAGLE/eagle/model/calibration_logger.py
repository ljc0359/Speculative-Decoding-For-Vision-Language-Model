import json
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from scipy import interpolate

CALIBRATION_LOGGING_ENABLED = True

class CalibrationLogger:
    """
    用于记录draft model的confidence score和acceptance rate数据的日志器
    增强版：支持attention权重记录和跨模态注意力强度分析
    """
    
    def __init__(self, save_dir: str = "/root/Speculative_decoding/calibration_data"):
        self.save_dir = save_dir
        # 改进：存储每次draft的完整信息，包括attention权重
        self.draft_sessions = []  # 存储每次draft的{confidence_scores, tokens, accepted_length, attention_weights, cross_modal_attention}
        
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)
        
        # 初始化统计数据
        self.reset_stats()
    
    def reset_stats(self):
        """重置统计数据"""
        self.draft_sessions = []
        self.current_session = None
    
    def start_draft_session(self, img_start_idx: Optional[int] = None, img_end_idx: Optional[int] = None):
        """
        开始一个新的draft session
        
        Args:
            img_start_idx: 图像token的起始索引
            img_end_idx: 图像token的结束索引
        """
        if not CALIBRATION_LOGGING_ENABLED:
            return
            
        self.current_session = {
            'confidence_scores': [],
            'tokens': [],
            'accepted_length': 0,
            'img_start_idx': img_start_idx,
            'img_end_idx': img_end_idx,
            'attention_weights': None,  # 存储attention权重
            'cross_modal_attention': []  # 存储每个token的跨模态注意力强度
        }
    
    def log_draft_confidence(self, confidence_scores: torch.Tensor, draft_tokens: torch.Tensor):
        """
        记录draft model的confidence scores
        
        Args:
            confidence_scores: draft model对每个token的confidence score (log probabilities)
            draft_tokens: 对应的draft tokens
        """
        if hasattr(self, 'current_session') and self.current_session is not None:
            # 转换为概率值 (从log probability)
            probs = torch.exp(confidence_scores).cpu().numpy()
            tokens = draft_tokens.cpu().numpy()
            
            self.current_session['confidence_scores'] = probs.flatten()
            # 使用统一的字段名 'tokens'，同时保留 'draft_tokens' 以保持兼容性
            self.current_session['tokens'] = tokens.flatten()
            self.current_session['draft_tokens'] = tokens.flatten()  # 保持兼容性
    
    def calculate_cross_modal_attention(self, attention_weights: torch.Tensor, 
                                      img_start_idx: int, img_end_idx: int) -> List[float]:
        """
        计算每个token对图像token的跨模态注意力强度
        
        Args:
            attention_weights: 注意力权重张量 [batch_size, num_heads, seq_len, seq_len]
            img_start_idx: 图像token起始索引
            img_end_idx: 图像token结束索引
            
        Returns:
            每个token的跨模态注意力强度列表
        """
        if attention_weights is None or img_start_idx is None or img_end_idx is None:
            return []
        
        # 确保attention_weights是tensor
        if not isinstance(attention_weights, torch.Tensor):
            return []
        
        # 获取最后一层的attention权重
        if len(attention_weights.shape) == 4:  # [batch, heads, seq_len, seq_len]
            attn = attention_weights[0]  # 取第一个batch
        else:
            attn = attention_weights
        
        # 对所有head求平均
        if len(attn.shape) == 3:  # [heads, seq_len, seq_len]
            attn = attn.mean(dim=0)  # [seq_len, seq_len]
        
        cross_modal_scores = []
        seq_len = attn.shape[0]
        
        # 计算每个token对图像区域的注意力强度
        for token_idx in range(seq_len):
            if img_start_idx < seq_len and img_end_idx <= seq_len:
                # 对图像token区域的注意力权重求和/均值
                img_attention = attn[token_idx, img_start_idx:img_end_idx].sum().item()
                cross_modal_scores.append(img_attention)
            else:
                cross_modal_scores.append(0.0)
        
        return cross_modal_scores
    
    def log_attention_weights(self, attention_weights: torch.Tensor):
        """
        记录attention权重并计算跨模态注意力强度
        
        Args:
            attention_weights: 注意力权重张量
        """
        if not CALIBRATION_LOGGING_ENABLED or self.current_session is None:
            return
        
        self.current_session['attention_weights'] = attention_weights
        
        # 计算跨模态注意力强度
        if (self.current_session['img_start_idx'] is not None and 
            self.current_session['img_end_idx'] is not None):
            cross_modal_scores = self.calculate_cross_modal_attention(
                attention_weights,
                self.current_session['img_start_idx'],
                self.current_session['img_end_idx']
            )
            self.current_session['cross_modal_attention'] = cross_modal_scores
    
    def log_acceptance(self, accepted_length: int, draft_tokens=None, best_candidate=None):
        """记录acceptance结果"""
        if not CALIBRATION_LOGGING_ENABLED or self.current_session is None:
            return
        
        # 确保accepted_length是Python int类型
        if hasattr(accepted_length, 'item'):
            accepted_length = accepted_length.item()
        elif hasattr(accepted_length, 'cpu'):
            accepted_length = accepted_length.cpu().item()
            
        self.current_session['accepted_length'] = int(accepted_length)
        
        # 记录额外的acceptance信息（如果提供）
        if draft_tokens is not None:
            self.current_session['draft_tokens'] = draft_tokens.cpu().tolist() if hasattr(draft_tokens, 'cpu') else draft_tokens
        if best_candidate is not None:
            self.current_session['best_candidate'] = best_candidate.cpu().tolist() if hasattr(best_candidate, 'cpu') else best_candidate
        
        # 将当前session添加到历史记录中
        self.draft_sessions.append(self.current_session.copy())
        self.current_session = None

        # print("add draft sessions")
    
    def get_token_level_data(self):
        """
        提取token级别的数据用于分析
        
        Returns:
            List[Dict]: 每个token的数据，包含confidence, is_accepted, token, cross_modal_attention
        """
        token_data = []
        
        for session in self.draft_sessions:
            confidence_scores = session.get('confidence_scores', [])
            # 修复：使用正确的字段名，支持两种字段名以保持兼容性
            tokens = session.get('draft_tokens', session.get('tokens', []))
            accepted_length = session.get('accepted_length', 0)
            cross_modal_attention = session.get('cross_modal_attention', [])
            
            # 确保数据类型正确
            if isinstance(accepted_length, torch.Tensor):
                accepted_length = int(accepted_length.item())
            elif not isinstance(accepted_length, int):
                accepted_length = int(accepted_length) if accepted_length is not None else 0
            
            # 确保confidence_scores和tokens都是列表或数组
            if isinstance(confidence_scores, torch.Tensor):
                confidence_scores = confidence_scores.cpu().numpy()
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.cpu().numpy()
            
            # 处理tokens可能是嵌套列表的情况
            if isinstance(tokens, (list, np.ndarray)) and len(tokens) > 0:
                # 检查第一个元素是否是列表
                if isinstance(tokens[0], (list, np.ndarray)):
                    # 如果是嵌套列表，展平它
                    tokens = np.concatenate([np.array(t).flatten() for t in tokens])
                else:
                    # 确保是一维数组
                    tokens = np.array(tokens).flatten()
            else:
                tokens = np.array([])
            
            # 处理confidence_scores可能是嵌套的情况
            if isinstance(confidence_scores, (list, np.ndarray)) and len(confidence_scores) > 0:
                if isinstance(confidence_scores[0], (list, np.ndarray)):
                    # 如果是嵌套列表，展平它
                    confidence_scores = np.concatenate([np.array(c).flatten() for c in confidence_scores])
                else:
                    # 确保是一维数组
                    confidence_scores = np.array(confidence_scores).flatten()
            else:
                confidence_scores = np.array([])
            
            # 确保长度一致
            if len(confidence_scores) > 0 and len(tokens) > 0:
                min_length = min(len(confidence_scores), len(tokens))
                confidence_scores = confidence_scores[:min_length]
                tokens = tokens[:min_length]
                
                for i, (conf, token) in enumerate(zip(confidence_scores, tokens)):
                    is_accepted = bool(i < accepted_length)
                    cross_modal_score = cross_modal_attention[i] if i < len(cross_modal_attention) else 0.0
                    
                    # 确保所有数据都是Python原生类型
                    try:
                        if isinstance(conf, (torch.Tensor, np.ndarray)):
                            conf = float(conf.item() if hasattr(conf, 'item') else conf)
                        else:
                            conf = float(conf)
                    except (ValueError, TypeError):
                        conf = 0.0
                    
                    try:
                        if isinstance(token, (torch.Tensor, np.ndarray)):
                            token = int(token.item() if hasattr(token, 'item') else token)
                        elif isinstance(token, (list, tuple)):
                            # 如果token仍然是列表，取第一个元素
                            token = int(token[0]) if len(token) > 0 else 0
                        else:
                            token = int(token)
                    except (ValueError, TypeError):
                        token = 0
                    
                    try:
                        if isinstance(cross_modal_score, (torch.Tensor, np.ndarray)):
                            cross_modal_score = float(cross_modal_score.item() if hasattr(cross_modal_score, 'item') else cross_modal_score)
                        else:
                            cross_modal_score = float(cross_modal_score)
                    except (ValueError, TypeError):
                        cross_modal_score = 0.0
                    
                    token_data.append({
                        'confidence': conf,
                        'is_accepted': is_accepted,
                        'token': token,
                        'cross_modal_attention': cross_modal_score
                    })
        
        return token_data
    
    def analyze_by_cross_modal_attention(self, num_quantiles: int = 5) -> Dict[str, Any]:
        """
        按跨模态注意力强度分位数分析校准情况
        
        Args:
            num_quantiles: 分位数数量
            
        Returns:
            分析结果字典
        """
        token_data = self.get_token_level_data()
        if not token_data:
            return {}
        
        # 提取跨模态注意力分数
        cross_modal_scores = [item['cross_modal_attention'] for item in token_data]
        confidences = [item['confidence'] for item in token_data]
        acceptances = [item['is_accepted'] for item in token_data]
        
        # 计算分位数阈值
        quantiles = np.linspace(0, 1, num_quantiles + 1)
        thresholds = np.quantile(cross_modal_scores, quantiles)
        
        results = {}
        for i in range(num_quantiles):
            # 找到属于当前分位数的token
            if i == num_quantiles - 1:  # 最后一个分位数包含最大值
                mask = (np.array(cross_modal_scores) >= thresholds[i]) & (np.array(cross_modal_scores) <= thresholds[i+1])
            else:
                mask = (np.array(cross_modal_scores) >= thresholds[i]) & (np.array(cross_modal_scores) < thresholds[i+1])
            
            if np.sum(mask) == 0:
                continue
            
            quantile_confidences = np.array(confidences)[mask]
            quantile_acceptances = np.array(acceptances)[mask]
            
            # 计算该分位数的校准指标
            ece = self.calculate_ece(quantile_confidences, quantile_acceptances)
            avg_confidence = np.mean(quantile_confidences)
            avg_accuracy = np.mean(quantile_acceptances)
            
            results[f'Q{i+1}'] = {
                'range': f'[{thresholds[i]:.4f}, {thresholds[i+1]:.4f}]',
                'count': int(np.sum(mask)),
                'avg_cross_modal_attention': np.mean(np.array(cross_modal_scores)[mask]),
                'avg_confidence': avg_confidence,
                'avg_accuracy': avg_accuracy,
                'ece': ece,
                'confidence_scores': quantile_confidences.tolist(),
                'acceptance_labels': quantile_acceptances.tolist()
            }
        
        return results
    
    def plot_cross_modal_attention_comprehensive_analysis(self, save_path: Optional[str] = None, num_quantiles: int = 5, num_bins: int = 20) -> List[str]:
        """
        绘制跨模态注意力的综合分析图，包含三个独立的图：
        1. 按注意力分位数的接受率分析
        2. 每个注意力分位数的条件可靠性图
        3. 置信度×注意力分位数的二维热力图
        
        Args:
            save_path: 保存路径前缀（不包含文件名）
            num_quantiles: 注意力分位数数量 (默认5，对应Q1-Q5)
            num_bins: 置信度分桶数量 (默认20)
            
        Returns:
            保存的三个图片路径列表
        """
        token_data = self.get_token_level_data()
        if not token_data:
            print("No data available for cross-modal attention analysis")
            return []
        
        # 提取数据
        cross_modal_scores = np.array([item['cross_modal_attention'] for item in token_data])
        confidences = np.array([item['confidence'] for item in token_data])
        acceptances = np.array([item['is_accepted'] for item in token_data])
        
        # 计算等频分位数阈值
        quantiles = np.linspace(0, 1, num_quantiles + 1)
        thresholds = np.quantile(cross_modal_scores, quantiles)
        
        # 为每个token分配分位数标签
        quantile_labels = np.zeros(len(cross_modal_scores), dtype=int)
        for i in range(num_quantiles):
            if i == num_quantiles - 1:  # 最后一个分位数包含最大值
                mask = (cross_modal_scores >= thresholds[i]) & (cross_modal_scores <= thresholds[i+1])
            else:
                mask = (cross_modal_scores >= thresholds[i]) & (cross_modal_scores < thresholds[i+1])
            quantile_labels[mask] = i
        
        # 准备保存路径
        if save_path is None:
            save_dir = self.save_dir
        else:
            save_dir = save_path
        
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)
        
        saved_paths = []
        
        # ========== 图1: 按注意力分位数的接受率分析 ==========
        plt.figure(figsize=(12, 8))
        quantile_names = [f'Q{i+1}' for i in range(num_quantiles)]
        acceptance_rates = []
        quantile_counts = []
        
        for i in range(num_quantiles):
            mask = quantile_labels == i
            if np.sum(mask) > 0:
                acceptance_rate = np.mean(acceptances[mask])
                acceptance_rates.append(acceptance_rate)
                quantile_counts.append(np.sum(mask))
            else:
                acceptance_rates.append(0)
                quantile_counts.append(0)
        
        bars = plt.bar(quantile_names, acceptance_rates, color='skyblue', alpha=0.7, edgecolor='navy')
        plt.xlabel('Attention Quantiles (Q1=Weak Visual Dependency, Q5=Strong Visual Dependency)', fontsize=14)
        plt.ylabel('Acceptance Rate', fontsize=14)
        plt.title('Acceptance Rate Analysis by Cross-Modal Attention Quantiles', fontsize=16, fontweight='bold')
        plt.ylim(0, 1)
        
        # 在柱状图上添加数值标签
        for bar, rate, count in zip(bars, acceptance_rates, quantile_counts):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{rate:.3f}\n(n={count})', ha='center', va='bottom', fontsize=12)
        
        # 添加网格
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存第一个图
        save_path_1 = os.path.join(save_dir, "cross_modal_attention_acceptance_rates.png")
        plt.savefig(save_path_1, dpi=300, bbox_inches='tight')
        plt.close()
        saved_paths.append(save_path_1)
        
        # ========== 图2: 每个注意力分位数的条件可靠性图 ==========
        plt.figure(figsize=(12, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, num_quantiles))
        
        for i in range(num_quantiles):
            mask = quantile_labels == i
            if np.sum(mask) < 10:  # 跳过样本太少的分位数
                continue
                
            q_confidences = confidences[mask]
            q_acceptances = acceptances[mask]
            
            # 创建置信度分桶
            bin_edges = np.linspace(0, 1, num_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_acceptance_rates = []
            bin_counts = []
            
            for j in range(num_bins):
                bin_mask = (q_confidences >= bin_edges[j]) & (q_confidences < bin_edges[j+1])
                if j == num_bins - 1:  # 最后一个bin包含1.0
                    bin_mask = (q_confidences >= bin_edges[j]) & (q_confidences <= bin_edges[j+1])
                
                if np.sum(bin_mask) > 0:
                    bin_acceptance_rate = np.mean(q_acceptances[bin_mask])
                    bin_acceptance_rates.append(bin_acceptance_rate)
                    bin_counts.append(np.sum(bin_mask))
                else:
                    bin_acceptance_rates.append(np.nan)
                    bin_counts.append(0)
            
            # 只绘制有数据的点
            valid_mask = ~np.isnan(bin_acceptance_rates)
            if np.sum(valid_mask) > 0:
                plt.plot(bin_centers[valid_mask], np.array(bin_acceptance_rates)[valid_mask], 
                        'o-', color=colors[i], label=f'Q{i+1}', alpha=0.8, markersize=6, linewidth=2)
        
        # 添加完美校准线
        plt.plot([0, 1], [0, 1], 'r--', alpha=0.7, linewidth=3, label='Perfect Calibration')
        plt.xlabel('Draft Confidence', fontsize=14)
        plt.ylabel('True Acceptance Rate', fontsize=14)
        plt.title('Conditional Reliability Diagram by Attention Quantiles', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.tight_layout()
        
        # 保存第二个图
        save_path_2 = os.path.join(save_dir, "cross_modal_attention_reliability_diagram.png")
        plt.savefig(save_path_2, dpi=300, bbox_inches='tight')
        plt.close()
        saved_paths.append(save_path_2)
        
        # ========== 图3: 二维热力图：置信度×注意力分位数 ==========
        plt.figure(figsize=(14, 8))
        
        # 创建二维网格
        confidence_bins = np.linspace(0, 1, num_bins + 1)
        heatmap_data = np.zeros((num_quantiles, num_bins))
        heatmap_counts = np.zeros((num_quantiles, num_bins))
        
        for i in range(num_quantiles):
            for j in range(num_bins):
                # 找到属于当前网格的数据点
                quantile_mask = quantile_labels == i
                if j == num_bins - 1:
                    confidence_mask = (confidences >= confidence_bins[j]) & (confidences <= confidence_bins[j+1])
                else:
                    confidence_mask = (confidences >= confidence_bins[j]) & (confidences < confidence_bins[j+1])
                
                combined_mask = quantile_mask & confidence_mask
                
                if np.sum(combined_mask) > 0:
                    acceptance_rate = np.mean(acceptances[combined_mask])
                    heatmap_data[i, j] = acceptance_rate
                    heatmap_counts[i, j] = np.sum(combined_mask)
                else:
                    heatmap_data[i, j] = np.nan
        
        # 创建热力图
        im = plt.imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
        
        # 设置坐标轴标签
        plt.xticks(np.arange(0, num_bins, 4), [f'{confidence_bins[i]:.1f}' for i in range(0, num_bins, 4)])
        plt.yticks(np.arange(num_quantiles), [f'Q{i+1}' for i in range(num_quantiles)])
        
        plt.xlabel('Draft Confidence', fontsize=14)
        plt.ylabel('Attention Quantiles', fontsize=14)
        plt.title('Confidence × Attention Quantiles Heatmap\n(Color = True Acceptance Rate)', fontsize=16, fontweight='bold')
        
        # 添加颜色条
        cbar = plt.colorbar(im, shrink=0.8)
        cbar.set_label('True Acceptance Rate', fontsize=13)
        
        # 在热力图上添加数值标签（仅显示有足够样本的格子）
        for i in range(num_quantiles):
            for j in range(num_bins):
                if heatmap_counts[i, j] >= 5:  # 只显示样本数>=5的格子
                    text = f'{heatmap_data[i, j]:.2f}\n({int(heatmap_counts[i, j])})'
                    plt.text(j, i, text, ha="center", va="center", 
                            color="white" if heatmap_data[i, j] < 0.5 else "black", fontsize=10)
        
        plt.tight_layout()
        
        # 保存第三个图
        save_path_3 = os.path.join(save_dir, "cross_modal_attention_heatmap.png")
        plt.savefig(save_path_3, dpi=300, bbox_inches='tight')
        plt.close()
        saved_paths.append(save_path_3)
        
        # 打印统计信息
        print(f"跨模态注意力分析图已保存:")
        print(f"  1. 接受率分析: {save_path_1}")
        print(f"  2. 可靠性图: {save_path_2}")
        print(f"  3. 热力图: {save_path_3}")
        print(f"总样本数: {len(token_data)}")
        print("各分位数样本分布:")
        for i in range(num_quantiles):
            count = np.sum(quantile_labels == i)
            avg_attention = np.mean(cross_modal_scores[quantile_labels == i]) if count > 0 else 0
            avg_acceptance = np.mean(acceptances[quantile_labels == i]) if count > 0 else 0
            print(f"  Q{i+1}: {count} 样本, 平均注意力={avg_attention:.4f}, 平均接受率={avg_acceptance:.3f}")
        
        return saved_paths

    def _make_json_serializable(self, obj):
        """
        递归地将对象转换为JSON可序列化的格式
        """
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        else:
            return obj

    def save_data(self, filename_prefix: str = "calibration_data"):
        """
        保存收集到的数据
        
        Args:
            filename_prefix: 文件名前缀
        """
        if not self.draft_sessions:
            print("No data to save")
            return None, None
        
        # 提取token级别的数据
        token_data = self.get_token_level_data()
        
        # 清理draft_sessions中的不可序列化对象
        cleaned_draft_sessions = []
        for session in self.draft_sessions:
            cleaned_session = session.copy()
            # 移除或转换不可序列化的attention_weights
            if 'attention_weights' in cleaned_session:
                # 不保存完整的attention weights到JSON中，因为太大
                # 只保存一些统计信息
                attn_weights = cleaned_session['attention_weights']
                if attn_weights is not None:
                    if isinstance(attn_weights, torch.Tensor):
                        attn_weights = attn_weights.detach().cpu()
                    cleaned_session['attention_weights'] = {
                        'shape': list(attn_weights.shape) if hasattr(attn_weights, 'shape') else None,
                        'mean': float(attn_weights.mean()) if hasattr(attn_weights, 'mean') else None,
                        'std': float(attn_weights.std()) if hasattr(attn_weights, 'std') else None
                    }
                else:
                    cleaned_session['attention_weights'] = None
            
            # 确保其他字段也是可序列化的
            cleaned_session = self._make_json_serializable(cleaned_session)
            cleaned_draft_sessions.append(cleaned_session)
        
        # 准备保存的数据
        data = {
            'draft_sessions': cleaned_draft_sessions,
            'token_data': token_data,
            'summary': {
                'total_sessions': len(self.draft_sessions),
                'total_tokens': len(token_data),
                'avg_confidence': float(np.mean([item['confidence'] for item in token_data])) if token_data else 0,
                'avg_acceptance_rate': float(np.mean([item['is_accepted'] for item in token_data])) if token_data else 0,
                'avg_cross_modal_attention': float(np.mean([item['cross_modal_attention'] for item in token_data])) if token_data else 0
            }
        }
        
        # 保存为JSON文件
        json_path = os.path.join(self.save_dir, f"{filename_prefix}.json")
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        # 保存为numpy文件（便于后续分析）
        if token_data:
            confidence_scores = np.array([item['confidence'] for item in token_data])
            acceptance_labels = np.array([item['is_accepted'] for item in token_data])
            tokens = np.array([item['token'] for item in token_data])
            cross_modal_attention = np.array([item['cross_modal_attention'] for item in token_data])
            
            np_path = os.path.join(self.save_dir, f"{filename_prefix}.npz")
            np.savez(np_path, 
                    confidence_scores=confidence_scores,
                    acceptance_labels=acceptance_labels,
                    tokens=tokens,
                    cross_modal_attention=cross_modal_attention)
        
        print(f"Calibration data saved to {json_path}")
        print(f"Total draft sessions: {len(self.draft_sessions)}")
        print(f"Total token samples: {len(token_data)}")
        
        return json_path, np_path if token_data else None

    def calculate_oce_uce(self, confidence_scores, acceptance_labels, num_bins=20):
        """
        计算OCE（过度自信误差）和UCE（信心不足误差）
        
        Args:
            confidence_scores: 置信度分数数组
            acceptance_labels: 接受标签数组
            num_bins: 分箱数量
            
        Returns:
            dict: 包含oce, uce, bin_confidences, bin_acceptance_rates, bin_counts等信息的字典
        """
        confidence_scores = np.array(confidence_scores)
        acceptance_labels = np.array(acceptance_labels)
        
        # 创建等宽分箱
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_indices = np.digitize(confidence_scores, bin_boundaries) - 1
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)
        
        bin_confidences = []
        bin_accuracies = []
        bin_counts = []
        
        oce = 0.0  # 过度自信误差
        uce = 0.0  # 信心不足误差
        total_samples = len(confidence_scores)
        
        for i in range(num_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_conf = np.mean(confidence_scores[mask])
                bin_acc = np.mean(acceptance_labels[mask])
                bin_count = np.sum(mask)
                
                bin_confidences.append(bin_conf)
                bin_accuracies.append(bin_acc)
                bin_counts.append(bin_count)
                
                # 计算OCE和UCE
                weight = bin_count / total_samples
                if bin_conf > bin_acc:  # 过度自信
                    oce += weight * (bin_conf - bin_acc)
                else:  # 信心不足
                    uce += weight * (bin_acc - bin_conf)
            else:
                bin_confidences.append(0)
                bin_accuracies.append(0)
                bin_counts.append(0)
        
        return {
            'oce': float(oce),
            'uce': float(uce),
            'bin_boundaries': bin_boundaries.tolist(),
            'bin_confidences': bin_confidences,
            'bin_acceptance_rates': bin_accuracies,  # 注意这里使用bin_acceptance_rates作为key
            'bin_counts': bin_counts
        }

    def calculate_ece(self, confidence_scores, acceptance_labels, num_bins=20):
        """
        计算ECE（期望校准误差）
        
        Args:
            confidence_scores: 置信度分数数组
            acceptance_labels: 接受标签数组
            num_bins: 分箱数量
            
        Returns:
            float: ECE值
        """
        confidence_scores = np.array(confidence_scores)
        acceptance_labels = np.array(acceptance_labels)
        
        # 创建等宽分箱
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_indices = np.digitize(confidence_scores, bin_boundaries) - 1
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)
        
        ece = 0.0
        total_samples = len(confidence_scores)
        
        for i in range(num_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_confidence = np.mean(confidence_scores[mask])
                bin_accuracy = np.mean(acceptance_labels[mask])
                bin_count = np.sum(mask)
                
                ece += (bin_count / total_samples) * abs(bin_confidence - bin_accuracy)
        
        return ece

    def plot_colored_reliability_diagram(self, confidence_scores, acceptance_labels, 
                                       num_bins=20, save_path=None):
        """
        绘制着色的校准曲线图（Reliability Diagram + Area）
        
        Args:
            confidence_scores: 置信度分数数组
            acceptance_labels: 接受标签数组
            num_bins: 分箱数量
            save_path: 保存路径
        """
        # 计算OCE和UCE
        oce_uce_data = self.calculate_oce_uce(confidence_scores, acceptance_labels, num_bins)
        
        bin_confidences = oce_uce_data['bin_confidences']
        bin_acceptance_rates = oce_uce_data['bin_acceptance_rates']
        oce = oce_uce_data['oce']
        uce = oce_uce_data['uce']
        
        # 创建图表
        plt.figure(figsize=(10, 8))
        
        # 创建更密集的点用于平滑曲线和填充
        if len(bin_confidences) > 1:
            # 使用插值创建平滑曲线
            x_smooth = np.linspace(0, 1, 1000)
            
            # 添加端点以确保曲线覆盖整个范围
            extended_conf = np.concatenate([[0], bin_confidences, [1]])
            extended_acc = np.concatenate([[0], bin_acceptance_rates, [1]])
            
            # 排序以确保插值正确
            sort_idx = np.argsort(extended_conf)
            extended_conf = extended_conf[sort_idx]
            extended_acc = extended_acc[sort_idx]
            
            # 使用线性插值创建平滑曲线
            f_interp = interpolate.interp1d(extended_conf, extended_acc, 
                                          kind='linear', bounds_error=False, 
                                          fill_value='extrapolate')
            y_smooth = f_interp(x_smooth)
            
            # 限制y值在[0,1]范围内
            y_smooth = np.clip(y_smooth, 0, 1)
            
            # 填充UCE区域（曲线在对角线上方，蓝色）
            plt.fill_between(x_smooth, y_smooth, x_smooth, 
                           where=(y_smooth >= x_smooth), 
                           color='lightblue', alpha=0.6, 
                           label=f'UCE (Underconfidence): {uce:.4f}')
            
            # 填充OCE区域（曲线在对角线下方，红色）
            plt.fill_between(x_smooth, y_smooth, x_smooth, 
                           where=(y_smooth <= x_smooth), 
                           color='lightcoral', alpha=0.6, 
                           label=f'OCE (Overconfidence): {oce:.4f}')
            
            # 绘制平滑的校准曲线
            plt.plot(x_smooth, y_smooth, 'b-', linewidth=3, alpha=0.8, 
                    label='Reliability Curve')
        
        # 绘制理想校准线（对角线）
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.8, 
                label='Perfect Calibration')
        
        # 绘制原始数据点
        plt.scatter(bin_confidences, bin_acceptance_rates, 
                   c='darkblue', s=80, alpha=0.8, zorder=5, 
                   label='Binned Data Points')
        
        # 设置图表属性
        plt.xlabel('Confidence Score', fontsize=14, fontweight='bold')
        plt.ylabel('Acceptance Rate', fontsize=14, fontweight='bold')
        plt.title('Colored Reliability Diagram\n(OCE/UCE Visualization)', 
                 fontsize=16, fontweight='bold')
        plt.legend(loc='upper left', fontsize=11)
        plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        
        # 添加统计信息文本框
        textstr = f'OCE: {oce:.4f}\nUCE: {uce:.4f}\nECE: {oce + uce:.4f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=12,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        # 保存图表
        if save_path is None:
            save_path = os.path.join(self.save_dir, "colored_reliability_diagram.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Colored reliability diagram saved to: {save_path}")
        
        return save_path, {'oce': oce, 'uce': uce, 'ece': oce + uce}

    def get_calibration_stats(self, num_bins: int = 20, save_figure: bool = True, figure_path: str = None):
        """
        计算校准统计数据并绘制acceptance rate和confidence的图
        
        Args:
            num_bins: 分箱数量
            save_figure: 是否保存图表
            figure_path: 图表保存路径
            
        Returns:
            Dict containing calibration statistics and ECE
        """
        token_data = self.get_token_level_data()
        
        if not token_data:
            return {}
        
        # 提取confidence scores和acceptance labels
        confidence_scores = np.array([item['confidence'] for item in token_data])
        acceptance_labels = np.array([item['is_accepted'] for item in token_data])
        
        # 计算总体acceptance rate (移到这里，在绘图之前)
        overall_acceptance_rate = float(np.mean(acceptance_labels))
        
        # 创建confidence score bins
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_indices = np.digitize(confidence_scores, bin_boundaries) - 1
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)
        
        # 计算每个bin的统计数据
        bin_stats = {}
        bin_confidences = []
        bin_acceptance_rates = []
        bin_counts = []
        
        for i in range(num_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_confidence = confidence_scores[mask]
                bin_acceptance = acceptance_labels[mask]
                
                avg_confidence = float(np.mean(bin_confidence))
                avg_acceptance = float(np.mean(bin_acceptance))  # 真正的token级别acceptance rate
                count = int(np.sum(mask))
                
                bin_stats[f'bin_{i}'] = {
                    'bin_range': [float(bin_boundaries[i]), float(bin_boundaries[i+1])],
                    'count': count,
                    'avg_confidence': avg_confidence,
                    'avg_acceptance_rate': avg_acceptance,
                    'min_confidence': float(np.min(bin_confidence)),
                    'max_confidence': float(np.max(bin_confidence))
                }
                
                bin_confidences.append(avg_confidence)
                bin_acceptance_rates.append(avg_acceptance)
                bin_counts.append(count)
        
        # 计算ECE (Expected Calibration Error)
        bin_confidences = np.array(bin_confidences)
        bin_acceptance_rates = np.array(bin_acceptance_rates)
        bin_counts = np.array(bin_counts)
        
        total_samples = np.sum(bin_counts)
        ece = np.sum(bin_counts / total_samples * np.abs(bin_confidences - bin_acceptance_rates))
        
        # 绘制原有的图表
        if save_figure and len(bin_confidences) > 0:
            plt.figure(figsize=(15, 6))
            
            # 子图1: Calibration Plot - 使用条形图样式
            plt.subplot(1, 2, 1)
            
            # 计算bin的宽度和中心位置
            bin_width = 1.0 / num_bins
            bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
            
            # 绘制理想校准线
            plt.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.7, label='Perfect Calibration')
            
            # 绘制条形图显示每个bin的acceptance rate
            bars = plt.bar(bin_centers[:len(bin_acceptance_rates)], bin_acceptance_rates, 
                          width=bin_width * 0.8, alpha=0.7, color='steelblue', 
                          edgecolor='navy', linewidth=1, label='Observed Acceptance Rate')
            
            # 在每个条形上方显示样本数量
            for i, (center, rate, count) in enumerate(zip(bin_centers[:len(bin_acceptance_rates)], 
                                                            bin_acceptance_rates, bin_counts)):
                plt.text(center, rate + 0.02, f'{count}', ha='center', va='bottom', 
                        fontsize=9, fontweight='bold')
            
            plt.xlabel('Confidence Score', fontsize=12, fontweight='bold')
            plt.ylabel('Token Acceptance Rate', fontsize=12, fontweight='bold')
            plt.title(f'Token-level Calibration\n(ECE: {ece:.4f})', fontsize=14, fontweight='bold')
            plt.legend(loc='upper left', fontsize=10)
            plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            plt.xlim(-0.05, 1.05)
            # 修复这里的数组布尔判断问题
            max_acceptance_rate = float(np.max(bin_acceptance_rates)) if len(bin_acceptance_rates) > 0 else 0.0
            plt.ylim(0, max(1.1, max_acceptance_rate * 1.1))
            
            # 子图2: Confidence Score Distribution
            plt.subplot(1, 2, 2)
            
            # 绘制confidence score分布的条形图
            bars2 = plt.bar(bin_centers[:len(bin_counts)], bin_counts, 
                           width=bin_width * 0.8, alpha=0.7, color='lightgreen', 
                           edgecolor='darkgreen', linewidth=1)
            
            # 在每个条形上方显示数量
            for center, count in zip(bin_centers[:len(bin_counts)], bin_counts):
                plt.text(center, count + max(bin_counts) * 0.01, f'{count}', 
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            plt.xlabel('Confidence Score', fontsize=12, fontweight='bold')
            plt.ylabel('Token Count', fontsize=12, fontweight='bold')
            plt.title('Confidence Score Distribution', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            plt.xlim(-0.05, 1.05)
            
            plt.tight_layout()
            
            # 保存图表
            if figure_path is None:
                figure_path = os.path.join(self.save_dir, "token_level_calibration.png")
            
            plt.savefig(figure_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"Token-level calibration plot saved to: {figure_path}")
        
        # 绘制新的着色校准曲线图
        colored_diagram_path = None
        oce_uce_stats = {}
        if save_figure and len(bin_confidences) > 0:
            colored_diagram_path, oce_uce_stats = self.plot_colored_reliability_diagram(
                confidence_scores, acceptance_labels, num_bins)
        
        return {
            'bin_stats': bin_stats,
            'overall_acceptance_rate': overall_acceptance_rate,
            'total_samples': len(token_data),
            'total_sessions': len(self.draft_sessions),
            'num_bins': num_bins,
            'ece': float(ece),
            'oce': oce_uce_stats.get('oce', 0.0),
            'uce': oce_uce_stats.get('uce', 0.0),
            'figure_path': figure_path if save_figure else None,
            'colored_diagram_path': colored_diagram_path
        }

# 全局logger实例
_global_logger = None

def get_calibration_logger():
    """获取全局的校准日志器实例"""
    global _global_logger
    if _global_logger is None:
        _global_logger = CalibrationLogger()
    return _global_logger

def reset_calibration_logger():
    """重置全局的校准日志器"""
    global _global_logger
    if _global_logger is not None:
        _global_logger.reset_stats()

# 创建全局实例
calibration_logger = get_calibration_logger()