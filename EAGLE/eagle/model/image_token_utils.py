import torch
from typing import List, Tuple, Optional

IMAGE_TOKEN_INDEX = -200

def map_placeholders_to_spans(input_ids_b: torch.Tensor, image_feats_b_list: List[torch.Tensor]) -> List[Tuple[int, int]]:
    """
    将input_ids中的图像占位符映射到展开后序列中的实际spans
    
    Args:
        input_ids_b: (S,) 单条序列的 ids（含 -200 占位符）
        image_feats_b_list: List[Tensor]，每个占位符对应一段图像特征，长度 L_i = feats.size(0)
    
    Returns:
        spans: List[(start, end)] 在"展开后的序列坐标系"里的图像token位置
    """
    placeholder_pos = (input_ids_b == IMAGE_TOKEN_INDEX).nonzero(as_tuple=False).squeeze(-1).tolist()
    
    # 如果没有图像占位符，返回空列表
    if not placeholder_pos:
        return []
    
    # 确保placeholder_pos是列表
    if isinstance(placeholder_pos, int):
        placeholder_pos = [placeholder_pos]
    
    spans = []
    cursor = 0  # 展开后序列的当前位置
    last_text_pos = 0
    
    for j, p in enumerate(placeholder_pos):
        # 先推进占位符之前的文本长度
        cursor += (p - last_text_pos)
        
        # 获取该占位符对应的图像token数
        if j < len(image_feats_b_list):
            L_img = int(image_feats_b_list[j].shape[0])
        else:
            # 如果image_feats_b_list长度不够，使用默认值（LLaVA通常是576）
            L_img = 576
            
        start = cursor
        end = cursor + L_img
        spans.append((start, end))
        cursor = end
        last_text_pos = p + 1
    
    return spans

def spans_to_mask(spans: List[Tuple[int, int]], total_len: int) -> torch.Tensor:
    """
    根据spans生成图像token的mask
    
    Args:
        spans: List[(start, end)] 图像token的位置范围
        total_len: 总序列长度
    
    Returns:
        mask: (total_len,) bool tensor，True表示图像token位置
    """
    mask = torch.zeros(total_len, dtype=torch.bool)
    for s, e in spans:
        if s < total_len and e <= total_len:
            mask[s:e] = True
    return mask

def get_image_token_spans_from_embeddings(
    input_ids: torch.Tensor, 
    inputs_embeds: torch.Tensor,
    batch_idx: int = 0
) -> Tuple[Optional[int], Optional[int]]:
    """
    从已经展开的embeddings中推断图像token的位置
    这是一个备用方法，当无法直接获取image_features时使用
    
    Args:
        input_ids: 原始input_ids (batch_size, seq_len)
        inputs_embeds: 展开后的embeddings (batch_size, expanded_seq_len, hidden_size)
        batch_idx: 要处理的batch索引
    
    Returns:
        (img_start_idx, img_end_idx): 图像token在展开序列中的起止位置
    """
    # 检查是否有图像占位符
    if IMAGE_TOKEN_INDEX not in input_ids[batch_idx]:
        return None, None
    
    # 计算原始序列长度和展开后长度的差异
    original_len = input_ids.shape[1]
    expanded_len = inputs_embeds.shape[1]
    
    # 如果长度相同，说明没有图像token
    if original_len == expanded_len:
        return None, None
    
    # 找到第一个图像占位符的位置
    placeholder_pos = (input_ids[batch_idx] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
    if len(placeholder_pos) == 0:
        return None, None
    
    first_placeholder = placeholder_pos[0].item()
    
    # 假设图像token被插入到第一个占位符的位置
    # 图像token数量 = 展开后长度 - 原始长度 + 占位符数量
    num_placeholders = len(placeholder_pos)
    total_image_tokens = expanded_len - original_len + num_placeholders
    
    # 简化假设：所有图像token连续放置在第一个占位符位置
    img_start_idx = first_placeholder
    img_end_idx = first_placeholder + total_image_tokens
    
    return img_start_idx, img_end_idx

def calculate_image_token_positions_for_calibration(
    input_ids: torch.Tensor,
    inputs_embeds: Optional[torch.Tensor] = None,
    image_features: Optional[List[torch.Tensor]] = None,
    batch_idx: int = 0
) -> Tuple[Optional[int], Optional[int]]:
    """
    为calibration logging计算正确的图像token位置
    
    Args:
        input_ids: 原始input_ids，包含-200占位符
        inputs_embeds: 展开后的embeddings（可选）
        image_features: 图像特征列表（可选）
        batch_idx: batch索引
    
    Returns:
        (img_start_idx, img_end_idx): 展开后序列中的图像token位置
    """
    # 方法1：如果有image_features，使用精确的span映射
    if image_features is not None and len(image_features) > 0:
        try:
            spans = map_placeholders_to_spans(input_ids[batch_idx], image_features)
            if spans:
                # 返回第一个span的开始和最后一个span的结束
                img_start_idx = spans[0][0]
                img_end_idx = spans[-1][1]
                return img_start_idx, img_end_idx
        except Exception as e:
            print(f"Warning: Failed to use image_features for span mapping: {e}")
    
    # 方法2：如果有inputs_embeds，从长度差异推断
    if inputs_embeds is not None:
        try:
            return get_image_token_spans_from_embeddings(input_ids, inputs_embeds, batch_idx)
        except Exception as e:
            print(f"Warning: Failed to infer from embeddings: {e}")
    
    # 方法3：回退到简单的占位符位置（不准确，但至少有个位置）
    if IMAGE_TOKEN_INDEX in input_ids[batch_idx]:
        placeholder_pos = (input_ids[batch_idx] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
        if len(placeholder_pos) > 0:
            first_placeholder = placeholder_pos[0].item()
            # 使用默认的576个图像token
            return first_placeholder, first_placeholder + 576
    
    return None, None