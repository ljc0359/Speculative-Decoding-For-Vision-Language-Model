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
        print(f"[debug] calibration logger path: {save_dir}")
        # 改进：存储每次draft的完整信息，包括attention权重
        self.draft_sessions = []  # 存储每次draft的{confidence_scores, tokens, accepted_length, attention_weights, cross_modal_attention}
        
        # 确保保存目录存在

        if save_dir == None:
            save_dir = "/root/Speculative_decoding/calibration_data/baseline"
            
        os.makedirs(save_dir, exist_ok=True)
        
        # 初始化统计数据
        self.reset_stats()
        
        # 添加候选校准数据存储
        self.candidate_calibration_data = []
    
    def reset_stats(self):
        """重置统计数据"""
        self.draft_sessions = []
        self.current_session = None
        self.candidate_calibration_data = []
    
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
    
    def log_draft_confidence(self, path_confidence_scores: torch.Tensor, local_confidence_scores: torch.Tensor, 
                           draft_tokens: torch.Tensor, tree_positions: torch.Tensor, tree_depths: torch.Tensor, 
                           parent_positions: torch.Tensor):
        """
        记录draft model的多种confidence信息和树位置信息
        
        Args:
            path_confidence_scores: 路径累计confidence scores (log probabilities)
            local_confidence_scores: 单步token confidence scores (log probabilities) 
            draft_tokens: 对应的draft tokens
            tree_positions: token在树中的BFS位置编号 (P1, P2, ...)
            tree_depths: token在树中的深度 (1, 2, 3, ...)
            parent_positions: 父节点的位置编号 (0表示根节点的子节点)
        """
        if hasattr(self, 'current_session') and self.current_session is not None:
            # 转换为概率值 (从log probability)
            path_probs = torch.exp(path_confidence_scores).cpu().numpy()
            local_probs = torch.exp(local_confidence_scores).cpu().numpy()
            
            # 记录多种置信度信息
            self.current_session['path_confidence_scores'] = path_probs.flatten()  # 路径累计置信度
            self.current_session['local_confidence_scores'] = local_probs.flatten()  # 单步置信度
            
            # 记录树位置信息
            self.current_session['tree_positions'] = tree_positions.cpu().numpy().flatten()  # BFS位置编号
            self.current_session['tree_depths'] = tree_depths.cpu().numpy().flatten()  # 树深度
            self.current_session['parent_positions'] = parent_positions.cpu().numpy().flatten()  # 父节点位置
            
            # 记录token信息（用于调试和验证）
            self.current_session['draft_tokens'] = draft_tokens.cpu().numpy().flatten()
    
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

    def calculate_cross_modal_attention(self, attention_weights: torch.Tensor, 
                                      img_start_idx: int, img_end_idx: int) -> List[float]:
        """
        计算推测解码中候选token对图像token的跨模态注意力强度
        
        在推测解码中，attention_weights的结构是：
        - 形状: [batch_size, num_heads, candidate_seq_len, full_context_len]
        - candidate_seq_len: 候选token数量（约60个）
        - full_context_len: 完整上下文长度（包含原始输入和图像token）
        
        Args:
            attention_weights: 注意力权重张量，候选token对完整上下文的attention
            img_start_idx: 图像token在完整上下文中的起始索引
            img_end_idx: 图像token在完整上下文中的结束索引
            
        Returns:
            每个候选token的跨模态注意力强度列表
        """
        
        if attention_weights is None or img_start_idx is None or img_end_idx is None:
            return []
        
        # 确保attention_weights是tensor
        if not isinstance(attention_weights, torch.Tensor):
            return []
        
        if torch.isnan(attention_weights).any():
            return []
        if torch.isinf(attention_weights).any():
            return []
        
        try:
            if len(attention_weights.shape) == 4:
                batch_size, num_heads, candidate_seq_len, full_context_len = attention_weights.shape
                attn = attention_weights[0]
            elif len(attention_weights.shape) == 3:
                num_heads, candidate_seq_len, full_context_len = attention_weights.shape
                attn = attention_weights
            elif len(attention_weights.shape) == 2:
                candidate_seq_len, full_context_len = attention_weights.shape
                attn = attention_weights   
            else:
                return []
            
            # 对多个head求平均（如果需要）
            if len(attn.shape) == 3:  # [num_heads, candidate_seq_len, full_context_len]
                attn = attn.mean(dim=0)  # [candidate_seq_len, full_context_len]
                # print(f"DEBUG: After head averaging, shape: {attn.shape}")
            
            candidate_seq_len, full_context_len = attn.shape
            
            # 验证图像token索引的有效性
            if img_start_idx < 0 or img_end_idx <= img_start_idx:
                # print(f"DEBUG: Invalid image token indices: start={img_start_idx}, end={img_end_idx}")
                return [0.0] * candidate_seq_len
            
            if img_end_idx > full_context_len:
                # print(f"DEBUG: Image token end index {img_end_idx} exceeds context length {full_context_len}")
                # 调整到有效范围
                img_end_idx = min(img_end_idx, full_context_len)
                if img_start_idx >= img_end_idx:
                    # print("DEBUG: No valid image tokens after adjustment")
                    return [0.0] * candidate_seq_len
            
            # print(f"DEBUG: Using image token range: [{img_start_idx}:{img_end_idx}] in context of length {full_context_len}")
            
            # 计算每个候选token对图像区域的注意力强度
            cross_modal_scores = []
            
            # 提取候选token对图像token的attention
            img_attention_matrix = attn[:, img_start_idx:img_end_idx]  # [candidate_seq_len, img_token_len]
            # print(f"DEBUG: Image attention matrix shape: {img_attention_matrix.shape}")
            
            # 对每个候选token计算其对图像区域的总attention
            for token_idx in range(candidate_seq_len):
                # 对该token对所有图像token的attention求和
                img_attention_sum = img_attention_matrix[token_idx].sum().item()
                cross_modal_scores.append(img_attention_sum)
                
                # 打印前几个token的详细信息
                if token_idx < 5:
                    img_attention_mean = img_attention_matrix[token_idx].mean().item()
                    img_attention_max = img_attention_matrix[token_idx].max().item()
                    # print(f"DEBUG: candidate_token_{token_idx} -> img_attention: sum={img_attention_sum:.6f}, mean={img_attention_mean:.6f}, max={img_attention_max:.6f}")
            
            # 统计信息
            if cross_modal_scores:
                scores_tensor = torch.tensor(cross_modal_scores)
            
            return cross_modal_scores
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            return []
    
    def log_candidate_calibration_data(self, calibration_data):
        """
        记录候选校准数据
        
        Args:
            calibration_data: 包含候选校准信息的字典列表，每个字典包含：
                - layer: 层索引
                - position_in_layer: 在该层中的位置
                - candidate_token: 候选token ID
                - draft_confidence: draft model的置信度
                - base_confidence: base model的置信度
                - tree_position: 在候选树中的位置
                - tree_depth: 在候选树中的深度
                - parent_position: 父节点在候选树中的位置
                - base_top1_token: base model的top1 token是否匹配 (1/0)
                - draft_margin: draft model的margin (top1_prob - top2_prob)
                - base_margin: base model的margin (top1_prob - top2_prob)
                - avg_visual_attention_intensity: 平均视觉注意力强度
        """
        if not CALIBRATION_LOGGING_ENABLED or self.current_session is None:
            return
        
        if not isinstance(calibration_data, list):
            calibration_data = [calibration_data]
        
        # 将candidate calibration数据添加到当前session
        if 'candidate_calibration_data' not in self.current_session:
            self.current_session['candidate_calibration_data'] = []
        
        self.current_session['candidate_calibration_data'].extend(calibration_data)
        print(f"[CALIBRATION] Logged {len(calibration_data)} candidate calibration data points")
    
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
            # 统一成扁平list
            if hasattr(draft_tokens, 'cpu'):
                tokens = draft_tokens.cpu().numpy().flatten().tolist()
            else:
                # list/np.ndarray等
                tokens = draft_tokens.flatten().tolist() if hasattr(draft_tokens, 'flatten') else list(draft_tokens)
            self.current_session['tokens'] = tokens
            self.current_session['draft_tokens'] = tokens  # 保持兼容性
            
            # 与confidence_scores长度对齐（支持新的多种置信度字段）
            path_probs = self.current_session.get('path_confidence_scores', None)
            local_probs = self.current_session.get('local_confidence_scores', None)
            # 向后兼容旧字段
            legacy_probs = self.current_session.get('confidence_scores', None)
            
            # 优先使用新字段，如果没有则使用旧字段
            probs = path_probs if path_probs is not None else legacy_probs
            
            if probs is not None:
                # 转为list以便处理
                probs_list = probs.tolist() if not isinstance(probs, list) else probs
                min_len = min(len(tokens), len(probs_list))
                if len(tokens) != len(probs_list):
                    # 截断到一致长度
                    tokens = tokens[:min_len]
                    probs_list = probs_list[:min_len]
                    self.current_session['tokens'] = tokens
                    self.current_session['draft_tokens'] = tokens
                    
                    # 同时截断所有置信度相关字段
                    if path_probs is not None:
                        self.current_session['path_confidence_scores'] = probs_list
                    if local_probs is not None:
                        local_probs_list = local_probs.tolist() if not isinstance(local_probs, list) else local_probs
                        self.current_session['local_confidence_scores'] = local_probs_list[:min_len]
                    if legacy_probs is not None:
                        self.current_session['confidence_scores'] = probs_list
                    
                    # 同时截断树位置信息
                    for field in ['tree_positions', 'tree_depths', 'parent_positions']:
                        if field in self.current_session and self.current_session[field] is not None:
                            field_data = self.current_session[field]
                            if isinstance(field_data, (list, np.ndarray)) and len(field_data) > min_len:
                                self.current_session[field] = field_data[:min_len]
                
                # 生成 per-token acceptance labels：前 accepted_length 个为1，其余为0
                acceptance_labels = [1 if i < accepted_length else 0 for i in range(min_len)]
                self.current_session['acceptance_labels'] = acceptance_labels
            else:
                # 没有confidence_scores也生成labels，但后续分析会发现缺少配对
                acceptance_labels = [1 if i < accepted_length else 0 for i in range(len(tokens))]
                self.current_session['acceptance_labels'] = acceptance_labels
        
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
            List[Dict]: 每个token的数据，包含path_confidence, local_confidence, is_accepted, token, cross_modal_attention, tree_position, tree_depth, parent_position
        """
        token_data = []
        
        for session in self.draft_sessions:
            # 支持新的多种置信度字段，同时保持向后兼容
            path_confidence_scores = session.get('path_confidence_scores', session.get('confidence_scores', []))
            local_confidence_scores = session.get('local_confidence_scores', session.get('confidence_scores', []))
            # 修复：使用正确的字段名，支持两种字段名以保持兼容性
            tokens = session.get('draft_tokens', session.get('tokens', []))
            accepted_length = session.get('accepted_length', 0)
            cross_modal_attention = session.get('cross_modal_attention', [])
            
            # 新增的树位置信息
            tree_positions = session.get('tree_positions', [])
            tree_depths = session.get('tree_depths', [])
            parent_positions = session.get('parent_positions', [])
            
            # 确保数据类型正确
            if isinstance(accepted_length, torch.Tensor):
                accepted_length = int(accepted_length.item())
            elif not isinstance(accepted_length, int):
                accepted_length = int(accepted_length) if accepted_length is not None else 0
            
            # 确保confidence_scores和tokens都是列表或数组
            if isinstance(path_confidence_scores, torch.Tensor):
                path_confidence_scores = path_confidence_scores.cpu().numpy()
            if isinstance(local_confidence_scores, torch.Tensor):
                local_confidence_scores = local_confidence_scores.cpu().numpy()
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
            
            # 处理path_confidence_scores可能是嵌套的情况
            if isinstance(path_confidence_scores, (list, np.ndarray)) and len(path_confidence_scores) > 0:
                if isinstance(path_confidence_scores[0], (list, np.ndarray)):
                    # 如果是嵌套列表，展平它
                    path_confidence_scores = np.concatenate([np.array(c).flatten() for c in path_confidence_scores])
                else:
                    # 确保是一维数组
                    path_confidence_scores = np.array(path_confidence_scores).flatten()
            else:
                path_confidence_scores = np.array([])
            
            # 处理local_confidence_scores可能是嵌套的情况
            if isinstance(local_confidence_scores, (list, np.ndarray)) and len(local_confidence_scores) > 0:
                if isinstance(local_confidence_scores[0], (list, np.ndarray)):
                    # 如果是嵌套列表，展平它
                    local_confidence_scores = np.concatenate([np.array(c).flatten() for c in local_confidence_scores])
                else:
                    # 确保是一维数组
                    local_confidence_scores = np.array(local_confidence_scores).flatten()
            else:
                local_confidence_scores = np.array([])
            
            # 确保长度一致
            if len(path_confidence_scores) > 0 and len(tokens) > 0:
                min_length = min(len(path_confidence_scores), len(local_confidence_scores), len(tokens))
                path_confidence_scores = path_confidence_scores[:min_length]
                local_confidence_scores = local_confidence_scores[:min_length]
                tokens = tokens[:min_length]
                
                # 截断其他数组到相同长度
                cross_modal_attention = cross_modal_attention[:min_length] if len(cross_modal_attention) > min_length else cross_modal_attention
                tree_positions = tree_positions[:min_length] if len(tree_positions) > min_length else tree_positions
                tree_depths = tree_depths[:min_length] if len(tree_depths) > min_length else tree_depths
                parent_positions = parent_positions[:min_length] if len(parent_positions) > min_length else parent_positions
                
                for i in range(min_length):
                    path_conf = path_confidence_scores[i] if i < len(path_confidence_scores) else 0.0
                    local_conf = local_confidence_scores[i] if i < len(local_confidence_scores) else 0.0
                    token = tokens[i] if i < len(tokens) else 0
                    is_accepted = bool(i < accepted_length)
                    cross_modal_score = cross_modal_attention[i] if i < len(cross_modal_attention) else 0.0
                    tree_position = tree_positions[i] if i < len(tree_positions) else 0
                    tree_depth = tree_depths[i] if i < len(tree_depths) else 0
                    parent_position = parent_positions[i] if i < len(parent_positions) else 0
                    
                    # 确保所有数据都是Python原生类型
                    try:
                        if isinstance(path_conf, (torch.Tensor, np.ndarray)):
                            path_conf = float(path_conf.item() if hasattr(path_conf, 'item') else path_conf)
                        else:
                            path_conf = float(path_conf)
                    except (ValueError, TypeError):
                        path_conf = 0.0
                    
                    try:
                        if isinstance(local_conf, (torch.Tensor, np.ndarray)):
                            local_conf = float(local_conf.item() if hasattr(local_conf, 'item') else local_conf)
                        else:
                            local_conf = float(local_conf)
                    except (ValueError, TypeError):
                        local_conf = 0.0
                    
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
                    
                    try:
                        tree_position = int(tree_position)
                    except (ValueError, TypeError):
                        tree_position = 0
                    
                    try:
                        tree_depth = int(tree_depth)
                    except (ValueError, TypeError):
                        tree_depth = 0
                    
                    try:
                        parent_position = int(parent_position)
                    except (ValueError, TypeError):
                        parent_position = 0
                    
                    token_data.append({
                        'path_confidence': path_conf,
                        'local_confidence': local_conf,
                        'confidence': path_conf,  # 保持向后兼容
                        'is_accepted': is_accepted,
                        'token': token,
                        'cross_modal_attention': cross_modal_score,
                        'tree_position': tree_position,
                        'tree_depth': tree_depth,
                        'parent_position': parent_position
                    })
        
        return token_data
    
    def analyze_by_cross_modal_attention(self, num_quantiles: int = 5, use_equal_frequency_confidence_bins: bool = False) -> Dict[str, Any]:
        """
        按跨模态注意力强度分位数分析校准情况
        
        Args:
            num_quantiles: 分位数数量
            use_equal_frequency_confidence_bins: 是否对置信度使用等频分箱
            
        Returns:
            分析结果字典
        """
        token_data = self.get_token_level_data()
        if not token_data:
            return {}
        
        cross_modal_scores = [item['cross_modal_attention'] for item in token_data]
        # 优先使用path_confidence，如果没有则使用旧的confidence字段
        confidences = [item.get('path_confidence', item.get('confidence', 0.0)) for item in token_data]
        acceptances = [item['is_accepted'] for item in token_data]
        
        # 改为基于排序索引的等频分箱，避免阈值重复导致的空箱
        scores = np.array(cross_modal_scores, dtype=float)
        N = len(scores)
        if N == 0:
            return {}
        
        # 如果样本数少于分位数数量，缩小分位数数量
        actual_quantiles = min(num_quantiles, N)
        order = np.argsort(scores)
        bin_idx = np.linspace(0, N, actual_quantiles + 1).astype(int)
        quantile_labels = np.zeros(N, dtype=int)
        for i in range(actual_quantiles):
            start, end = bin_idx[i], bin_idx[i + 1]
            quantile_labels[order[start:end]] = i
        
        results = {}
        for i in range(actual_quantiles):
            mask = (quantile_labels == i)
            if np.sum(mask) == 0:
                continue
            
            quantile_confidences = np.array(confidences)[mask]
            quantile_acceptances = np.array(acceptances)[mask]
            quantile_scores = scores[mask]
            
            # 计算ECE（置信度分箱仍按配置使用等频或等宽）
            ece = self.calculate_ece(quantile_confidences, quantile_acceptances, 
                                     use_equal_frequency=use_equal_frequency_confidence_bins)
            avg_confidence = float(np.mean(quantile_confidences))
            avg_accuracy = float(np.mean(quantile_acceptances))
            
            # 用该分位数内的最小/最大注意力作为范围展示
            range_str = f'[{float(np.min(quantile_scores)):.4f}, {float(np.max(quantile_scores)):.4f}]'
            
            results[f'Q{i+1}'] = {
                'range': range_str,
                'count': int(np.sum(mask)),
                'avg_cross_modal_attention': float(np.mean(quantile_scores)),
                'avg_confidence': avg_confidence,
                'avg_accuracy': avg_accuracy,
                'ece': ece,
                'confidence_scores': quantile_confidences.tolist(),
                'acceptance_labels': quantile_acceptances.tolist(),
                'binning_method': 'equal_frequency' if use_equal_frequency_confidence_bins else 'equal_width'
            }
        
        return results
    
    def plot_cross_modal_attention_comprehensive_analysis(
        self,
        save_path: Optional[str] = None, 
        num_quantiles: int = 3,
        num_bins: int = 20,
        confidence_binning: str = "equal_frequency"  # "equal_frequency" | "equal_width" | "both"
    ) -> List[str]:
        """
        绘制跨模态注意力的综合分析图，包含三个独立的图：
        1. 按注意力分位数的接受率分析（与置信度分箱无关，仅按注意力分位数）
        2. 每个注意力分位数的条件可靠性图（可选：等频/等距分箱）
        3. 置信度×注意力分位数的二维热力图（可选：等频/等距分箱）

        Args:
            save_path: 保存目录（不包含文件名）
            num_quantiles: 注意力分位数数量 (默认5，对应Q1-Q5)
            num_bins: 置信度分桶数量 (默认20)
            confidence_binning: 置信度分箱策略:
                - "equal_frequency": 等频分箱（默认）
                - "equal_width": 等距分箱
                - "both": 两种都画（文件名会带后缀）

        Returns:
            保存的图片路径列表
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt

        # ---------- 数据准备 ----------
        token_data = self.get_token_level_data()
        if not token_data:
            print("No data available for cross-modal attention analysis")
            return []

        cross_modal_scores = np.array([item['cross_modal_attention'] for item in token_data], dtype=float)
        confidences = np.array([item.get('path_confidence', item.get('confidence', 0.0)) for item in token_data], dtype=float)
        acceptances = np.array([item['is_accepted'] for item in token_data], dtype=float)

        N = len(cross_modal_scores)
        if N == 0:
            print("No data available for cross-modal attention analysis")
            return []

        # ---------- 注意力分位数（等频，不会空箱） ----------
        actual_quantiles = min(num_quantiles, N)
        order = np.argsort(cross_modal_scores)
        bin_idx = np.linspace(0, N, actual_quantiles + 1).astype(int)
        quantile_labels = np.zeros(N, dtype=int)
        for i in range(actual_quantiles):
            start, end = bin_idx[i], bin_idx[i + 1]
            quantile_labels[order[start:end]] = i

        # ---------- 保存目录 ----------
        save_dir = self.save_dir if save_path is None else save_path
        os.makedirs(save_dir, exist_ok=True)
        saved_paths = []

        # ---------- 图1：按注意力分位数的接受率（与置信度分箱策略无关） ----------
        plt.figure(figsize=(12, 8))
        quantile_names = [f'Q{i+1}' for i in range(actual_quantiles)]
        acceptance_rates = []
        quantile_counts = []

        for i in range(actual_quantiles):
            mask = (quantile_labels == i)
            if np.sum(mask) > 0:
                acceptance_rates.append(float(np.mean(acceptances[mask])))
                quantile_counts.append(int(np.sum(mask)))
            else:
                acceptance_rates.append(0.0)
                quantile_counts.append(0)

        bars = plt.bar(quantile_names, acceptance_rates, alpha=0.7, edgecolor='black')
        plt.xlabel('Attention Quantiles (Q1=Weak Visual Dependency, Q5=Strong Visual Dependency)', fontsize=14)
        plt.ylabel('Acceptance Rate', fontsize=14)
        plt.title('Acceptance Rate Analysis by Cross-Modal Attention Quantiles', fontsize=16, fontweight='bold')
        plt.ylim(0, 1)

        for bar, rate, count in zip(bars, acceptance_rates, quantile_counts):
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., h + 0.01, f'{rate:.3f}\n(n={count})',
                    ha='center', va='bottom', fontsize=12)

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path_1 = os.path.join(save_dir, "cross_modal_attention_acceptance_rates.png")
        plt.savefig(save_path_1, dpi=300, bbox_inches='tight')
        plt.close()
        saved_paths.append(save_path_1)

        # ---------- 分箱策略工具 ----------
        def compute_global_bins(conf_arr: np.ndarray, bins: int, method: str) -> np.ndarray:
            """返回严格覆盖[min, max]的全局分箱边界（升序，长度 = bins + 1）"""
            if conf_arr.size == 0:
                # 退化：给一个[0,1]区间
                return np.linspace(0.0, 1.0, bins + 1)

            cmin, cmax = float(np.min(conf_arr)), float(np.max(conf_arr))
            if method == "equal_frequency":
                # 等频：使用分位数；去重后若太少，降级为等距
                q = np.linspace(0, 1, bins + 1)
                edges = np.quantile(conf_arr, q)
                edges = np.unique(edges)
                if edges.size < 3:  # 去重后过少，回退等距
                    edges = np.linspace(cmin, cmax, bins + 1)
                else:
                    edges[0] = cmin
                    edges[-1] = cmax
            elif method == "equal_width":
                edges = np.linspace(cmin, cmax, bins + 1)
            else:
                raise ValueError(f"Unknown binning method: {method}")
            # 确保严格单调非降（避免数值抖动导致相等）
            # 若仍出现重复边界，少数空箱是允许的；下游逻辑会跳过空箱
            return edges

        # 决定需要运行的分箱方法
        if confidence_binning not in ("equal_frequency", "equal_width", "both"):
            raise ValueError("confidence_binning must be 'equal_frequency', 'equal_width', or 'both'")

        binning_methods = (
            ["equal_frequency", "equal_width"] if confidence_binning == "both"
            else [confidence_binning]
        )

        # ---------- 针对每种置信度分箱方法，分别画 图2 与 图3 ----------
        for method in binning_methods:
            global_bin_edges = compute_global_bins(confidences, num_bins, method)
            global_bin_centers = (global_bin_edges[:-1] + global_bin_edges[1:]) / 2.0
            actual_num_bins = len(global_bin_edges) - 1
            method_tag = "EF" if method == "equal_frequency" else "EW"
            method_name = "Global Equal-Frequency" if method == "equal_frequency" else "Equal-Width"

            # ---- 图2：可靠性图（按注意力分位数条件化 + 全局置信度分箱） ----
            plt.figure(figsize=(12, 8))
            colors = plt.cm.viridis(np.linspace(0, 1, actual_quantiles))

            for qi in range(actual_quantiles):
                mask_q = (quantile_labels == qi)
                if np.sum(mask_q) < 10:
                    continue

                q_confidences = confidences[mask_q]
                q_acceptances = acceptances[mask_q]

                bin_acceptance_rates = []
                for j in range(actual_num_bins):
                    # 注意最后一个 bin 包含右端点
                    if j == actual_num_bins - 1:
                        bin_mask = (q_confidences >= global_bin_edges[j]) & (q_confidences <= global_bin_edges[j+1])
                    else:
                        bin_mask = (q_confidences >= global_bin_edges[j]) & (q_confidences <  global_bin_edges[j+1])

                    if np.any(bin_mask):
                        bin_acceptance_rates.append(float(np.mean(q_acceptances[bin_mask])))
                    else:
                        bin_acceptance_rates.append(np.nan)

                # 只绘制非 NaN
                br = np.array(bin_acceptance_rates, dtype=float)
                valid = ~np.isnan(br)
                if np.any(valid):
                    plt.plot(global_bin_centers[valid], br[valid], 'o-', color=colors[qi],
                            label=f'Q{qi+1}', alpha=0.85, markersize=6, linewidth=2)

            plt.plot([0, 1], [0, 1], 'r--', alpha=0.7, linewidth=3, label='Perfect Calibration')
            plt.xlabel('Draft Confidence', fontsize=14)
            plt.ylabel('True Acceptance Rate', fontsize=14)
            plt.title(f'Conditional Reliability Diagram by Attention Quantiles\n({method_name} Confidence Binning)',
                    fontsize=16, fontweight='bold')
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.tight_layout()

            save_path_2 = os.path.join(save_dir, f"cross_modal_attention_reliability_diagram_{method_tag}.png")
            plt.savefig(save_path_2, dpi=300, bbox_inches='tight')
            plt.close()
            saved_paths.append(save_path_2)

            # ---- 图3：二维热力图（注意力分位数 × 全局置信度分箱） ----
            plt.figure(figsize=(14, 8))
            heatmap_data = np.zeros((actual_quantiles, actual_num_bins))
            heatmap_counts = np.zeros((actual_quantiles, actual_num_bins))

            for qi in range(actual_quantiles):
                quantile_mask = (quantile_labels == qi)
                if not np.any(quantile_mask):
                    heatmap_data[qi, :] = np.nan
                    continue

                for j in range(actual_num_bins):
                    if j == actual_num_bins - 1:
                        confidence_mask = (confidences >= global_bin_edges[j]) & (confidences <= global_bin_edges[j+1])
                    else:
                        confidence_mask = (confidences >= global_bin_edges[j]) & (confidences <  global_bin_edges[j+1])

                    combined = quantile_mask & confidence_mask
                    cnt = int(np.sum(combined))
                    if cnt > 0:
                        heatmap_data[qi, j] = float(np.mean(acceptances[combined]))
                        heatmap_counts[qi, j] = cnt
                    else:
                        heatmap_data[qi, j] = np.nan

            im = plt.imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
            # X 轴：为了可读性，采样少量刻度
            step = max(1, actual_num_bins // 5)
            x_tick_indices = np.arange(0, actual_num_bins, step)
            x_tick_labels = [f'{global_bin_edges[i]:.2f}' for i in x_tick_indices]
            plt.xticks(x_tick_indices, x_tick_labels)
            plt.yticks(np.arange(actual_quantiles), [f'Q{i+1}' for i in range(actual_quantiles)])

            plt.xlabel('Draft Confidence', fontsize=14)
            plt.ylabel('Attention Quantiles', fontsize=14)
            plt.title(f'Confidence × Attention Quantiles Heatmap\n({method_name} Confidence Binning, Color = True Acceptance Rate)',
                    fontsize=16, fontweight='bold')

            cbar = plt.colorbar(im, shrink=0.8)
            cbar.set_label('True Acceptance Rate', fontsize=13)

            # 标注数值（样本数>=5）
            for qi in range(actual_quantiles):
                for j in range(actual_num_bins):
                    if heatmap_counts[qi, j] >= 5 and not np.isnan(heatmap_data[qi, j]):
                        txt = f'{heatmap_data[qi, j]:.2f}\n({int(heatmap_counts[qi, j])})'
                        plt.text(j, qi, txt, ha="center", va="center",
                                color="white" if heatmap_data[qi, j] < 0.5 else "black", fontsize=10)

            plt.tight_layout()
            save_path_3 = os.path.join(save_dir, f"cross_modal_attention_heatmap_{method_tag}.png")
            plt.savefig(save_path_3, dpi=300, bbox_inches='tight')
            plt.close()
            saved_paths.append(save_path_3)

            # ---- 统计信息 ----
            print(f"[{method_name}] 跨模态注意力分析图已保存：")
            print(f"  - 可靠性图: {save_path_2}")
            print(f"  - 热力图:   {save_path_3}")

        # 全局统计输出（一次即可）
        print(f"总样本数: {N}")
        print(f"全局置信度范围: {confidences.min():.4f} - {confidences.max():.4f}")
        print("各分位数样本分布:")
        for i in range(actual_quantiles):
            count = int(np.sum(quantile_labels == i))
            avg_attention = float(np.mean(cross_modal_scores[quantile_labels == i])) if count > 0 else 0.0
            avg_acceptance = float(np.mean(acceptances[quantile_labels == i])) if count > 0 else 0.0
            q_confidences = confidences[quantile_labels == i]
            conf_range = f"{float(q_confidences.min()):.3f}-{float(q_confidences.max()):.3f}" if count > 0 else "N/A"
            print(f"  Q{i+1}: {count} 样本, 平均注意力={avg_attention:.4f}, 平均接受率={avg_acceptance:.3f}, 置信度范围={conf_range}")

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

    def log_calibrator_scores(self, layer_idx: int, original_scores: torch.Tensor, 
                             calibrated_probs: torch.Tensor, calibrated_log_scores: torch.Tensor, 
                             candidate_count: int):
        """
        记录校准前后的分数变化
        
        Args:
            layer_idx: 层索引 (0 表示第一层，1+ 表示后续层)
            original_scores: 校准前的原始分数
            calibrated_probs: 校准器输出的概率值 (0-1范围)
            calibrated_log_scores: 取对数后的校准分数
            candidate_count: 候选数量
        """
        if not CALIBRATION_LOGGING_ENABLED:
            return
        
        try:
            # 转换为numpy数组便于序列化
            original_np = original_scores.detach().cpu().numpy()
            calibrated_probs_np = calibrated_probs.detach().cpu().numpy()
            calibrated_log_np = calibrated_log_scores.detach().cpu().numpy()
            
            # 计算概率值的差异（原始分数 vs 校准概率）
            # 注意：这里比较的是原始logits和校准后的概率值，可能需要转换
            # 将原始分数转换为概率以便比较
            original_probs_np = torch.softmax(original_scores, dim=-1).detach().cpu().numpy()
            prob_diff = calibrated_probs_np - original_probs_np
            
            # 计算对数分数的差异
            log_score_diff = calibrated_log_np - original_np
            
            # 准备记录数据
            calibrator_data = {
                'layer': layer_idx,
                'candidate_count': candidate_count,
                'original_scores': original_np.tolist(),
                'original_probs': original_probs_np.tolist(),  # 原始分数转换的概率
                'calibrated_probs': calibrated_probs_np.tolist(),  # 校准器输出的概率 (0-1)
                'calibrated_log_scores': calibrated_log_np.tolist(),  # 取对数后的分数
                'prob_differences': prob_diff.tolist(),  # 概率差异
                'log_score_differences': log_score_diff.tolist(),  # 对数分数差异
                'statistics': {
                    'original_scores_mean': float(np.mean(original_np)),
                    'original_scores_std': float(np.std(original_np)),
                    'original_probs_mean': float(np.mean(original_probs_np)),
                    'original_probs_std': float(np.std(original_probs_np)),
                    'calibrated_probs_mean': float(np.mean(calibrated_probs_np)),
                    'calibrated_probs_std': float(np.std(calibrated_probs_np)),
                    'calibrated_log_mean': float(np.mean(calibrated_log_np)),
                    'calibrated_log_std': float(np.std(calibrated_log_np)),
                    'prob_diff_mean': float(np.mean(prob_diff)),
                    'prob_diff_std': float(np.std(prob_diff)),
                    'log_diff_mean': float(np.mean(log_score_diff)),
                    'log_diff_std': float(np.std(log_score_diff)),
                    'max_prob_increase': float(np.max(prob_diff)),
                    'max_prob_decrease': float(np.min(prob_diff))
                }
            }
            
            # 读取现有数据或创建新文件
            calibrator_file = os.path.join(self.save_dir, "calibrator_scores.json")
            
            if os.path.exists(calibrator_file):
                with open(calibrator_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            else:
                existing_data = {'calibrator_score_changes': []}
            
            # 添加新数据
            existing_data['calibrator_score_changes'].append(calibrator_data)
            
            # 保存到文件
            with open(calibrator_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)
            
            print(f"[CalibrationLogger] Logged calibrator scores for layer {layer_idx} to {calibrator_file}")
            
        except Exception as e:
            print(f"[CalibrationLogger] Error logging calibrator scores: {e}")

    def get_candidate_calibration_data(self):
        """
        提取所有candidate calibration数据
        
        Returns:
            List[Dict]: 包含所有candidate的calibration数据
        """
        all_candidate_data = []
        
        for session in self.draft_sessions:
            if 'candidate_calibration_data' in session:
                all_candidate_data.extend(session['candidate_calibration_data'])
        
        return all_candidate_data

    def save_data(self, filename_prefix: str = "calibration_data"):
        """保存校准数据到文件"""
        if not self.draft_sessions:
            print("No calibration data to save")
            return
        
        # 从draft sessions中提取token数据
        token_data = self.get_token_level_data()
        
        # 获取candidate calibration数据
        candidate_calibration_data = self.get_candidate_calibration_data()
        
        # 清理draft sessions数据以便JSON序列化
        cleaned_draft_sessions = []
        for session in self.draft_sessions:
            cleaned_session = {}
            for key, value in session.items():
                if key == 'attention_weights':
                    # 不保存完整的attention weights到JSON中，因为太大
                    # 只保存一些统计信息
                    attn_weights = value
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
                else:
                    cleaned_session[key] = value
            
            # 确保其他字段也是可序列化的
            cleaned_session = self._make_json_serializable(cleaned_session)
            cleaned_draft_sessions.append(cleaned_session)
        
        # 准备保存的数据
        data = {
            'draft_sessions': cleaned_draft_sessions,
            'token_data': token_data,
            'candidate_calibration_data': candidate_calibration_data,  # 新增candidate calibration数据
            'summary': {
                'total_sessions': len(self.draft_sessions),
                'total_tokens': len(token_data),
                'total_candidates': len(candidate_calibration_data),  # 新增candidate数量统计
                'avg_confidence': float(np.mean([item.get('path_confidence', item.get('confidence', 0.0)) for item in token_data])) if token_data else 0,
                'avg_acceptance_rate': float(np.mean([item['is_accepted'] for item in token_data])) if token_data else 0,
                'avg_cross_modal_attention': float(np.mean([item['cross_modal_attention'] for item in token_data])) if token_data else 0,
                'avg_draft_candidate_confidence': float(np.mean([item.get('draft_confidence', 0.0) for item in candidate_calibration_data])) if candidate_calibration_data else 0,
                'avg_base_candidate_confidence': float(np.mean([item.get('base_confidence', 0.0) for item in candidate_calibration_data])) if candidate_calibration_data else 0,
                'avg_base_top1_match_rate': float(np.mean([item.get('base_top1_token', 0) for item in candidate_calibration_data])) if candidate_calibration_data else 0,
                'avg_draft_margin': float(np.mean([item.get('draft_margin', 0.0) for item in candidate_calibration_data])) if candidate_calibration_data else 0,
                'avg_base_margin': float(np.mean([item.get('base_margin', 0.0) for item in candidate_calibration_data])) if candidate_calibration_data else 0,
                'avg_visual_attention_intensity': float(np.mean([item.get('avg_visual_attention_intensity', 0.0) for item in candidate_calibration_data])) if candidate_calibration_data else 0  # 新增：平均视觉注意力强度统计
            }
        }
        
        # 保存为JSON文件
        json_path = os.path.join(self.save_dir, f"{filename_prefix}.json")
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        # 保存为numpy文件（便于后续分析）
        if token_data:
            # 支持新的多种置信度字段
            path_confidence_scores = np.array([item.get('path_confidence', item.get('confidence', 0.0)) for item in token_data])
            local_confidence_scores = np.array([item.get('local_confidence', item.get('confidence', 0.0)) for item in token_data])
            acceptance_labels = np.array([item['is_accepted'] for item in token_data])
            tokens = np.array([item['token'] for item in token_data])
            cross_modal_attention = np.array([item['cross_modal_attention'] for item in token_data])
            
            # 新增的树位置信息
            tree_positions = np.array([item.get('tree_position', 0) for item in token_data])
            tree_depths = np.array([item.get('tree_depth', 0) for item in token_data])
            parent_positions = np.array([item.get('parent_position', 0) for item in token_data])
            
            # 保存candidate calibration数据到numpy文件
            save_dict = {
                'path_confidence_scores': path_confidence_scores,
                'local_confidence_scores': local_confidence_scores,
                'confidence_scores': path_confidence_scores,  # 保持向后兼容
                'acceptance_labels': acceptance_labels,
                'tokens': tokens,
                'cross_modal_attention': cross_modal_attention,
                'tree_positions': tree_positions,
                'tree_depths': tree_depths,
                'parent_positions': parent_positions
            }
            
            # 如果有candidate calibration数据，也保存到numpy文件
            if candidate_calibration_data:  
                candidate_draft_confidences = np.array([item.get('draft_confidence', 0.0) for item in candidate_calibration_data])  
                candidate_base_confidences = np.array([item.get('base_confidence', 0.0) for item in candidate_calibration_data])
                candidate_tokens = np.array([item['candidate_token'] for item in candidate_calibration_data])
                candidate_layers = np.array([item['layer'] for item in candidate_calibration_data])
                candidate_tree_positions = np.array([item['tree_position'] for item in candidate_calibration_data])
                candidate_tree_depths = np.array([item['tree_depth'] for item in candidate_calibration_data])
                candidate_parent_positions = np.array([item['parent_position'] for item in candidate_calibration_data])
                
                # 新增字段
                candidate_base_top1_tokens = np.array([item.get('base_top1_token', 0) for item in candidate_calibration_data])
                candidate_draft_margins = np.array([item.get('draft_margin', 0.0) for item in candidate_calibration_data])
                candidate_base_margins = np.array([item.get('base_margin', 0.0) for item in candidate_calibration_data])
                candidate_avg_visual_attention = np.array([item.get('avg_visual_attention_intensity', 0.0) for item in candidate_calibration_data])  # 新增：平均视觉注意力强度
                
                save_dict.update({
                    'candidate_draft_confidences': candidate_draft_confidences,
                    'candidate_base_confidences': candidate_base_confidences,
                    'candidate_tokens': candidate_tokens,
                    'candidate_layers': candidate_layers,
                    'candidate_tree_positions': candidate_tree_positions,
                    'candidate_tree_depths': candidate_tree_depths,
                    'candidate_parent_positions': candidate_parent_positions,
                    # 新增字段
                    'candidate_base_top1_tokens': candidate_base_top1_tokens,
                    'candidate_draft_margins': candidate_draft_margins,
                    'candidate_base_margins': candidate_base_margins,
                    'candidate_avg_visual_attention': candidate_avg_visual_attention,  # 新增：平均视觉注意力强度
                    
                    # 添加校准器训练所需的字段
                    'soft_labels': candidate_base_confidences,  # soft_labels = base_confidence
                    'hard_labels': candidate_base_top1_tokens,  # hard_labels = base_top1_token (0或1)
                    
                    # 添加校准器所需的特征字段
                    'tree_position': candidate_tree_positions,
                    'draft_margin': candidate_draft_margins,
                    'avg_visual_attention_intensity': candidate_avg_visual_attention,
                    'token_category': np.array([item.get('token_category', 'content') for item in candidate_calibration_data])
                })
            
            np_path = os.path.join(self.save_dir, f"{filename_prefix}.npz")
            np.savez_compressed(np_path, **save_dict)
            
            print(f"Calibration data saved to {json_path} and {np_path}")
            print(f"Total tokens: {len(token_data)}, Total candidates: {len(candidate_calibration_data)}")
            if candidate_calibration_data:
                print(f"Average visual attention intensity: {data['summary']['avg_visual_attention_intensity']:.4f}")
        else:
            print("No token data to save")

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

    def calculate_ece(self, confidence_scores, acceptance_labels, num_bins=20, use_equal_frequency=True):
        """
        计算ECE（期望校准误差）
        
        Args:
            confidence_scores: 置信度分数数组
            acceptance_labels: 接受标签数组
            num_bins: 分箱数量
            use_equal_frequency: 是否使用等频分箱（推荐True）
            
        Returns:
            float: ECE值
        """
        confidence_scores = np.array(confidence_scores)
        acceptance_labels = np.array(acceptance_labels)
        
        if use_equal_frequency:
            # 使用等频分箱：每个分箱包含相等数量的样本
            quantiles = np.linspace(0, 1, num_bins + 1)
            bin_boundaries = np.quantile(confidence_scores, quantiles)
            # 确保边界值的唯一性，避免重复边界
            bin_boundaries = np.unique(bin_boundaries)
            if len(bin_boundaries) < num_bins + 1:
                # 如果唯一边界数量不足，回退到等距分箱
                print(f"Warning: 置信度值重复过多，回退到等距分箱。唯一边界数: {len(bin_boundaries)}")
                bin_boundaries = np.linspace(0, 1, num_bins + 1)
            
            bin_indices = np.digitize(confidence_scores, bin_boundaries) - 1
            bin_indices = np.clip(bin_indices, 0, len(bin_boundaries) - 2)
            actual_num_bins = len(bin_boundaries) - 1
        else:
            # 原始的等距分箱
            bin_boundaries = np.linspace(0, 1, num_bins + 1)
            bin_indices = np.digitize(confidence_scores, bin_boundaries) - 1
            bin_indices = np.clip(bin_indices, 0, num_bins - 1)
            actual_num_bins = num_bins
        
        ece = 0.0
        total_samples = len(confidence_scores)
        
        for i in range(actual_num_bins):
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
        
        # 提取confidence scores和acceptance labels（支持新的置信度字段）
        # 优先使用path_confidence，如果没有则使用旧的confidence字段
        confidence_scores = np.array([item.get('path_confidence', item.get('confidence', 0.0)) for item in token_data])
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

def get_calibration_logger(save_dir=None):
    """获取全局的校准日志器实例"""
    global _global_logger
    if _global_logger is None:
        _global_logger = CalibrationLogger(save_dir)
    return _global_logger

def reset_calibration_logger():
    """重置全局的校准日志器"""
    global _global_logger
    if _global_logger is not None:
        _global_logger.reset_stats()

# 创建全局实例
# calibration_logger = get_calibration_logger()