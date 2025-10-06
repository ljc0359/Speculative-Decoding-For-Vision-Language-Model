# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LLaMA model."""
import copy
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import math
import re
import string
from typing import List, Optional, Tuple, Union
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS


from eagle.model.configs import EConfig, Qwen2VLConfig
from eagle.model.choices import *
from eagle.model.utils import prepare_logits_processor

# 添加calibration logger导入
try:
    from eagle.model.calibration_logger import get_calibration_logger
    CALIBRATION_LOGGING_ENABLED = True
except ImportError:
    CALIBRATION_LOGGING_ENABLED = False
    print("Warning: Calibration logging not available")

# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
        input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)
    
# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class I(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.ones(1, dtype=torch.float32))

    def forward(self, x):
        return x + self.dummy - self.dummy  # (also tried x+self.dummy)


def len_list(x, n):
    return [i for i in x if len(i) <= n]


class Model(nn.Module):
    def __init__(self, config, load_emb=False, path=None, bias=True, total_tokens=63, depth=5, top_k=8, threshold=1.0, train_embed=False, decouple=False):
        super().__init__()

        self.gradient_checkpointing = True
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.decouple = decouple


        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        if load_emb:
            from eagle.train.main_deepspeed import get_embed
            tensor = get_embed(path)
            self.embed_tokens.weight.data = tensor

        self.top_k = top_k
        self.total_tokens = total_tokens - 1
        self.depth = depth
        self.threshold = math.log(threshold)

        from eagle.model.ea_llama_model import LlamaDecoderLayer
        from eagle.model.ea_qwen2vl_model import Qwen2VLDecoderLayer, Qwen2VLRotaryEmbedding

        if config.model_type == "llava":
            self.layers = nn.ModuleList([LlamaDecoderLayer(config, index) for index in range(config.num_hidden_layers)])
            self.rotary_emb = None 
        elif config.model_type == "qwen2_vl":
            self.layers = nn.ModuleList([Qwen2VLDecoderLayer(config, index) for index in range(config.num_hidden_layers)])
            self.rotary_emb = Qwen2VLRotaryEmbedding(config=config)

        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=bias)
        self.act = ACT2FN[config.hidden_act]
        self.logsoftmax = nn.LogSoftmax(dim=-1)

        if not train_embed:
            for param in self.embed_tokens.parameters():
                param.requires_grad = False

    def init_tree(self):
        self.tree_mask_init = torch.eye(self.top_k, device=self.embed_tokens.weight.device)[None, None]
        self.position_ids = torch.zeros(self.top_k, device=self.embed_tokens.weight.device, dtype=torch.long)
        self.tree_mask_init = self.tree_mask_init.to(self.embed_tokens.weight.device)

    def reset(self):
        self.tree_mask = None

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                # inputs_embeds.dtype,
                torch.float32,  # [MODIFIED] force to cast to float32
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, torch.float32, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        # [MODIFIED] add tree mask
        if hasattr(self, "tree_mask") and self.tree_mask is not None:
            tree_mask = self.tree_mask
            _, _, tree_shape0, tree_shape1 = tree_mask.shape
            combined_attention_mask[:, :, -tree_shape0:, -tree_shape1:][
                tree_mask == 0
                ] = torch.finfo(torch.float32).min

        return combined_attention_mask

    def forward(
            self,
            hidden_states,
            input_ids,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            image_tokens_num: Optional[list] = None,
            std=None,
    ):
        batch_size, seq_length, _ = hidden_states.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0
        if inputs_embeds != None:
            inputs_embeds = inputs_embeds.to(hidden_states.device)
        # with torch.no_grad():
        from eagle.model.utils import temp_cache

        if any(p.requires_grad for p in self.embed_tokens.parameters()):
            # Iterate over each input_ids in the batch
            new_inputs_embeds_list = []
            for batch_idx in range(input_ids.size(0)):
                single_input_ids = input_ids[batch_idx]  # Extract current sample's input_ids
                if -200 in single_input_ids:
                    # Extract the part that needs processing
                    ori_input_ids = single_input_ids[image_tokens_num[batch_idx] - 1:]
                    indice = (ori_input_ids == -200).nonzero(as_tuple=False).flatten().to(ori_input_ids.device)
                    # Mask to exclude -200
                    mask = (ori_input_ids != -200)
                    ori_input_ids = ori_input_ids[mask]  # Reshape to 1D
                    # Get embeddings
                    inputs_ids_embeds = self.embed_tokens(ori_input_ids)  # Embed only the part without -200
                    indice = indice.item()  # Index position
                    # Extract a segment of length 576
                    extracted_segment = inputs_embeds[batch_idx, indice:indice + image_tokens_num[batch_idx]]
                    # Insert back the extracted segment into ori_input_ids embeddings
                    inputs_ids_embeds = torch.cat([
                        inputs_ids_embeds[:indice],     # Part before insertion
                        extracted_segment,              # Extracted segment
                        inputs_ids_embeds[indice:]      # Part after insertion
                    ], dim=0).unsqueeze(0)
                    new_inputs_embeds_list.append(inputs_ids_embeds)

                elif 151652 in single_input_ids:
                    indice = (single_input_ids == 151652).nonzero(as_tuple=False).flatten().to(single_input_ids.device) + 1
                    extracted_segment = inputs_embeds[batch_idx, indice:indice + image_tokens_num[batch_idx]]
                    single_inputs_embeds = self.embed_tokens(single_input_ids.unsqueeze(0))
                    single_inputs_embeds[0, indice : indice + image_tokens_num[batch_idx]] = extracted_segment
                    new_inputs_embeds_list.append(single_inputs_embeds)
                else:
                    # If no -200 is present, directly get embeddings
                    single_inputs_embeds = self.embed_tokens(single_input_ids.unsqueeze(0))
                    new_inputs_embeds_list.append(single_inputs_embeds)

            # Concatenate the processed results into new inputs_embeds
            inputs_embeds = torch.cat(new_inputs_embeds_list, dim=0)

        elif temp_cache.use_msd:
            if -200 in input_ids:    # llava
                indice = torch.where(input_ids == -200)[1].item()
                inputs_ids_embeds = self.embed_tokens(
                    torch.cat((input_ids[:,:indice], input_ids[:,indice+1:]), dim=1)
                )
                inputs_embeds = torch.cat(
                    (inputs_ids_embeds[:,:indice], inputs_embeds[:, indice:indice + image_tokens_num[0]], inputs_ids_embeds[:,indice:]), dim=1
                )
            elif 151652 in input_ids and image_tokens_num != None:    # qwen2_vl
                indice = torch.where(input_ids == 151652)[1].item() + 1
                inputs_ids_embeds = self.embed_tokens(
                    torch.cat((input_ids[:,:indice], input_ids[:,indice + image_tokens_num[0]:]), dim=1)
                )
                inputs_embeds = torch.cat(
                    (inputs_ids_embeds[:,:indice], inputs_embeds[:, indice:indice + image_tokens_num[0]], inputs_ids_embeds[:,indice:]), dim=1
                )

            else:
                inputs_embeds = self.embed_tokens(input_ids)
        else:
            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)


        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if position_ids is None:
            device = hidden_states.device if hidden_states is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        #position_ids=position_ids//4
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=hidden_states.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), hidden_states, past_key_values_length
        )

        inputs_embeds = inputs_embeds.to(hidden_states.dtype)

        if self.decouple: # train
            for i in range(input_ids.shape[0]):
                if -200 in input_ids[i]:
                    ori_input_ids = input_ids[i][image_tokens_num[i] - 1:]
                    indice = torch.where(ori_input_ids == -200)[0].item()
                    temp_image_tensor = inputs_embeds[i][indice : indice + image_tokens_num[i]].clone()
                    inputs_embeds[i][indice] = inputs_embeds[i][indice + image_tokens_num[i]]
                    hidden_states[i] = self.fc(torch.cat((inputs_embeds[i], hidden_states[i]), dim=-1))
                    hidden_states[i][indice : indice + image_tokens_num[i]] = temp_image_tensor
                elif 151652 in input_ids[i]:
                    ori_input_ids = input_ids[i]
                    indice = torch.where(ori_input_ids == 151652)[0].item() + 1
                    temp_image_tensor = inputs_embeds[i][indice : indice + image_tokens_num[i]].clone()
                    inputs_embeds[i][indice] = inputs_embeds[i][indice + image_tokens_num[i]]
                    hidden_states[i] = self.fc(torch.cat((inputs_embeds[i], hidden_states[i]), dim=-1))
                    hidden_states[i][indice : indice + image_tokens_num[i]] = temp_image_tensor
                else:
                    hidden_states[i] = self.fc(torch.cat((inputs_embeds[i], hidden_states[i]), dim=-1))
        elif temp_cache.use_msd:
            input_ids = input_ids[0]
            inputs_embeds = inputs_embeds[0]
            hidden_states = hidden_states[0]
            if -200 in input_ids:
                ori_input_ids = input_ids[image_tokens_num[0] - 1:] if input_ids[0] == 0 else input_ids
                indice = torch.where(ori_input_ids == -200)[0].item()
                temp_image_tensor = inputs_embeds[indice : indice + image_tokens_num[0]].clone()
                new_hidden_state = hidden_states.clone()
                inputs_embeds[indice] = inputs_embeds[indice + image_tokens_num[0]]
                cat_tensor = torch.cat((inputs_embeds, new_hidden_state), dim=-1)
                new_hidden_state = self.fc(cat_tensor)
                new_hidden_state[indice : indice + image_tokens_num[0]] = temp_image_tensor
            elif 151652 in input_ids and image_tokens_num != None:
                ori_input_ids = input_ids
                indice = torch.where(ori_input_ids == 151652)[0].item() + 1
                temp_image_tensor = inputs_embeds[indice : indice + image_tokens_num[0]].clone()
                new_hidden_state = hidden_states.clone()
                inputs_embeds[indice] = inputs_embeds[indice + image_tokens_num[0]]
                cat_tensor = torch.cat((inputs_embeds, new_hidden_state), dim=-1)
                new_hidden_state = self.fc(cat_tensor)
                new_hidden_state[indice : indice + image_tokens_num[0]] = temp_image_tensor
            else:
                new_hidden_state = self.fc(torch.cat((inputs_embeds, hidden_states), dim=-1))

            hidden_states = new_hidden_state.unsqueeze(0)
        else:
            inputs_embeds = inputs_embeds.to(hidden_states.device)
            hidden_states = self.fc(torch.cat((inputs_embeds, hidden_states), dim=-1))

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None  # 添加注意力权重收集
        next_decoder_cache = () if use_cache else None

        if self.rotary_emb is not None:
            # qwen2vl
            if position_ids.dim() == 2:
                position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
        else:
            position_embeddings = None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, past_key_value, output_attentions, None, position_embeddings)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                # print(f"[DEBUG] Layer outputs length: {len(layer_outputs)}")
                # print(f"[DEBUG] Layer outputs[1] type: {type(layer_outputs[1])}")
                # if hasattr(layer_outputs[1], 'shape'):
                #     print(f"[DEBUG] Layer outputs[1] shape: {layer_outputs[1].shape}")
                all_self_attns += (layer_outputs[1],)  # 收集注意力权重

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # 根据不同的参数组合返回相应的值
        if use_cache:
            if output_attentions:
                if output_hidden_states:
                    return hidden_states, next_decoder_cache, all_hidden_states, all_self_attns
                else:
                    return hidden_states, next_decoder_cache, None, all_self_attns
            else:
                if output_hidden_states:
                    return hidden_states, next_decoder_cache, all_hidden_states
                else:
                    return hidden_states, next_decoder_cache
        else:
            if output_attentions:
                if output_hidden_states:
                    return hidden_states, all_hidden_states, all_self_attns
                else:
                    return hidden_states, None, all_self_attns
            else:
                if output_hidden_states:
                    return hidden_states, all_hidden_states
                else:
                    return hidden_states

    def reset_kv(self):
        self.stable_kv = None

    def _collect_calibration_data_safely(self, base_model, original_context, inputs_embeds, 
                                   context_past_key_values, topk_index, topk_p, 
                                   layer_positions, layer_depths, layer_parents, 
                                   layer_idx, frontier_paths=None, attentions=None,
                                   img_start_idx=None, img_end_idx=None, train_calibrator=False):
        """
        安全地收集候选校准数据，避免状态污染
        
        Args:
            train_calibrator: 是否为训练校准器模式
                - True: 收集完整校准数据（包括base model预测）
                - False: 只收集必要信息（token_category, avg_visual_attention_intensity, tree_position, draft_margin）
        
        Returns:
            calibration_data: 校准数据列表，内容根据train_calibrator参数而不同
        """
        import copy
        calibration_data = []
        
        # Token分类方法
        def categorize_token_simple(token_id, tokenizer):
            """
            简化的token分类方法，按照content, func_punct, number三类分类
            
            Args:
                token_id: Token ID
                tokenizer: 分词器
                
            Returns:
                str: Token category ('content', 'func_punct', 'number')
            """
            try:
                # Convert token ID to text
                token_text = tokenizer.decode([token_id], skip_special_tokens=False)
                
                # Remove possible prefix spaces
                token_text = token_text.strip()
                
                # Numbers (包括纯数字和小数)
                if token_text.isdigit() or re.match(r'^\d+\.?\d*$', token_text):
                    return 'number'
                
                # Punctuation and special tokens
                if (all(c in string.punctuation for c in token_text) or 
                    token_text.startswith('<') and token_text.endswith('>') or
                    not token_text or token_text.isspace()):
                    return 'func_punct'
                
                # Function words list (常见英文功能词)
                function_words = {
                    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                    'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
                    'before', 'after', 'above', 'below', 'between', 'among', 'under', 'over',
                    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                    'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
                    'can', 'must', 'shall', 'ought', 'need', 'dare', 'used',
                    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
                    'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs',
                    'this', 'that', 'these', 'those', 'here', 'there', 'where', 'when', 'why', 'how',
                    'what', 'which', 'who', 'whom', 'whose', 'if', 'unless', 'until', 'while', 'since',
                    'because', 'so', 'as', 'than', 'then', 'now', 'just', 'only', 'also', 'even',
                    'still', 'yet', 'already', 'again', 'once', 'twice', 'always', 'never', 'often',
                    'sometimes', 'usually', 'rarely', 'hardly', 'almost', 'quite', 'very', 'too',
                    'much', 'many', 'more', 'most', 'less', 'least', 'few', 'little', 'some', 'any',
                    'all', 'both', 'each', 'every', 'either', 'neither', 'none', 'no', 'not'
                }
                
                # Check if it's a function word
                clean_token = token_text.lower().strip(' ')  # Remove possible subword prefix
                if clean_token in function_words:
                    return 'func_punct'
                
                # Other cases are classified as content words
                return 'content'
                
            except Exception as e:
                # print(f"⚠️ Token classification failed (ID: {token_id}): {e}")
                return 'content'  # 默认分类为content
        
        # 获取tokenizer (从base_model中获取)
        tokenizer = getattr(base_model, 'tokenizer', None)
        if tokenizer is None:
            # 尝试从其他可能的位置获取tokenizer
            tokenizer = getattr(base_model, 'llm', None)
            if tokenizer is not None:
                tokenizer = getattr(tokenizer, 'tokenizer', None)
        
        # 计算平均视觉注意力强度的辅助函数
        def calculate_avg_visual_attention(attention_weights, img_start, img_end, candidate_idx=None):
            """
            计算候选token对图像区域的平均注意力强度
            
            Args:
                attention_weights: 注意力权重张量
                img_start: 图像token开始位置
                img_end: 图像token结束位置
                candidate_idx: 候选token索引（如果为None，计算所有token的平均）
            
            Returns:
                float: 平均注意力强度
            """
            try:
                if attention_weights is None or img_start is None or img_end is None:
                    return 0.0
                
                if img_start >= img_end:
                    return 0.0
                
                # 处理不同维度的attention tensor
                if len(attention_weights.shape) == 4:
                    # [batch_size, num_heads, seq_len, context_len]
                    attn = attention_weights[0]  # 取第一个batch
                elif len(attention_weights.shape) == 3:
                    # [num_heads, seq_len, context_len]
                    attn = attention_weights
                elif len(attention_weights.shape) == 2:
                    # [seq_len, context_len]
                    attn = attention_weights
                else:
                    return 0.0
                
                # 对多个head求平均（如果需要）
                if len(attn.shape) == 3:  # [num_heads, seq_len, context_len]
                    attn = attn.mean(dim=0)  # [seq_len, context_len]
                
                seq_len, context_len = attn.shape
                
                # 验证图像token索引的有效性
                if img_start < 0 or img_end <= img_start or img_end > context_len:
                    return 0.0
                
                # 如果指定了候选索引，只计算该候选的注意力
                if candidate_idx is not None:
                    if candidate_idx >= seq_len:
                        return 0.0
                    # 计算该候选token对图像区域的平均注意力
                    img_attention = attn[candidate_idx, img_start:img_end]
                    result = img_attention.mean().item()
                    return result
                else:
                    # 计算所有候选token对图像区域的平均注意力
                    img_attention_matrix = attn[:, img_start:img_end]  # [seq_len, img_token_len]
                    result = img_attention_matrix.mean().item()
                    return result
                    
            except Exception as e:
                # print(f"⚠️ Visual attention calculation failed: {e}")
                return 0.0
        
        # 根据train_calibrator模式决定是否需要base model预测
        if train_calibrator:
            # 训练模式：需要完整的校准数据，包括base model预测
            # 保存模型的原始状态
            original_training_state = base_model.training
            base_model.eval()  # 确保在评估模式下
        
        try:
            if layer_idx == 0:
                # 第0层的候选校准
                if train_calibrator:
                    # 训练模式：需要base model预测
                    with torch.no_grad():
                        # 使用原始context和KV缓存进行base model预测
                        outputs, orig, _ = base_model(
                            inputs_embeds=inputs_embeds,
                            past_key_values=context_past_key_values,
                            output_orig=True
                        )
                        base_logits = orig[:, -1]  # [1, vocab_size]
                        base_probs0 = torch.softmax(base_logits, dim=-1)
                        
                        # 计算base model的top1 token和margin
                        base_sorted_probs, base_sorted_indices = torch.sort(base_probs0[0], descending=True)
                        base_top1_token = base_sorted_indices[0].item()
                        base_top1_prob = base_sorted_probs[0].item()
                        base_top2_prob = base_sorted_probs[1].item() if len(base_sorted_probs) > 1 else 0.0
                        base_margin = base_top1_prob - base_top2_prob
                        
                        # 计算draft margin
                        draft_sorted_probs, draft_sorted_indices = torch.sort(torch.exp(topk_p[0]), descending=True)
                        draft_top1_prob = draft_sorted_probs[0].item()
                        draft_top2_prob = draft_sorted_probs[1].item() if len(draft_sorted_probs) > 1 else 0.0
                        draft_margin = draft_top1_prob - draft_top2_prob
                
                for child_idx in range(topk_index.shape[1]):
                    candidate_token_id = topk_index[0, child_idx].item()
                    
                    # 计算该候选token的平均视觉注意力强度
                    attention_tensor = attentions[0] if isinstance(attentions, (tuple, list)) and len(attentions) > 0 else attentions
                    avg_visual_attention = calculate_avg_visual_attention(
                        attention_tensor, img_start_idx, img_end_idx, candidate_idx=child_idx
                    )
                    
                    # 分类token
                    token_category = categorize_token_simple(candidate_token_id, tokenizer) if tokenizer is not None else 'content'
                    
                    if train_calibrator:
                        # 训练模式：收集完整数据
                        base_prob = base_probs0[0, candidate_token_id].item()
                        
                        calibration_data.append({
                            'layer': layer_idx,
                            'position_in_layer': child_idx,
                            'candidate_token': candidate_token_id,
                            'draft_confidence': torch.exp(topk_p[0, child_idx]).item(),
                            'base_confidence': base_prob,
                            'tree_position': layer_positions[child_idx].item(),
                            'tree_depth': layer_depths[child_idx].item(),
                            'parent_position': layer_parents[child_idx].item(),
                            'base_top1_token': int(candidate_token_id == base_top1_token),
                            'draft_margin': draft_margin,
                            'base_margin': base_margin,
                            'avg_visual_attention_intensity': avg_visual_attention,
                            'token_category': token_category
                        })
                    else:
                        # 推理模式：只收集必要信息
                        # 计算draft margin
                        draft_sorted_probs, draft_sorted_indices = torch.sort(torch.exp(topk_p[0]), descending=True)
                        draft_top1_prob = draft_sorted_probs[0].item()
                        draft_top2_prob = draft_sorted_probs[1].item() if len(draft_sorted_probs) > 1 else 0.0
                        draft_margin = draft_top1_prob - draft_top2_prob
                        
                        calibration_data.append({
                            'candidate_token': candidate_token_id,
                            'tree_position': layer_positions[child_idx].item(),
                            'draft_margin': draft_margin,
                            'avg_visual_attention_intensity': avg_visual_attention,
                            'token_category': token_category
                        })
            else:
                # 后续层的候选校准
                if frontier_paths is None:
                    raise ValueError("frontier_paths is required for non-zero layers")
                
                top_k = topk_index.shape[0]
                
                if train_calibrator:
                    # 训练模式：需要base model预测
                    # 为每个父路径计算base model预测
                    base_predictions = {}
                    
                    for parent_idx in range(top_k):
                        parent_path = frontier_paths[parent_idx]
                        
                        # 构建包含父路径的完整context
                        parent_context_ids = torch.cat([
                            original_context,
                            torch.tensor(parent_path, device=original_context.device).unsqueeze(0)
                        ], dim=1)
                        
                        # 构建对应的embeds
                        parent_path_embeds = self.embed_tokens(torch.tensor(parent_path, device=original_context.device))
                        parent_context_embeds = torch.cat([
                            inputs_embeds[:, 1:],  # 去掉第一个token的embed
                            parent_path_embeds.unsqueeze(0)
                        ], dim=1)
                        
                        with torch.no_grad():
                            # 注意：这里使用 None 作为 past_key_values，因为不同父前缀需要独立计算
                            outputs_i, orig_i, _ = base_model(
                                input_ids=parent_context_ids,
                                past_key_values=None,  # 不复用KV缓存，避免状态污染
                                output_orig=True,
                                inputs_embeds=parent_context_embeds
                            )
                            base_logits_i = orig_i[:, -1]  # [1, vocab_size]
                            base_probs_i = torch.softmax(base_logits_i, dim=-1)
                            
                            # 计算 base model 的 top-k 用于 margin 计算
                            base_sorted_probs_i, base_sorted_indices_i = torch.sort(base_probs_i[0], descending=True)
                            base_top1_token_i = base_sorted_indices_i[0].item()
                            base_top1_prob_i = base_sorted_probs_i[0].item()
                            base_top2_prob_i = base_sorted_probs_i[1].item() if len(base_sorted_probs_i) > 1 else 0.0
                            base_margin_i = base_top1_prob_i - base_top2_prob_i
                            
                            base_predictions[parent_idx] = {
                                'probs': base_probs_i,
                                'top1_token': base_top1_token_i,
                                'margin': base_margin_i
                            }
                
                for parent_idx in range(top_k):
                    if train_calibrator:
                        # 计算 draft model 的 margin (对于当前父节点)
                        draft_sorted_probs_i, draft_sorted_indices_i = torch.sort(torch.exp(topk_p[parent_idx]), descending=True)
                        draft_top1_prob_i = draft_sorted_probs_i[0].item()
                        draft_top2_prob_i = draft_sorted_probs_i[1].item() if len(draft_sorted_probs_i) > 1 else 0.0
                        draft_margin_i = draft_top1_prob_i - draft_top2_prob_i
                        
                        base_pred = base_predictions[parent_idx]
                        base_probs_i = base_pred['probs']
                        base_top1_token_i = base_pred['top1_token']
                        base_margin_i = base_pred['margin']
                    
                    for child_idx in range(topk_index.shape[1]):
                        candidate_token_id = topk_index[parent_idx, child_idx].item()
                        
                        # 计算该候选token的平均视觉注意力强度
                        candidate_attention_idx = child_idx
                        attention_tensor = attentions[0] if isinstance(attentions, (tuple, list)) and len(attentions) > 0 else attentions
                        avg_visual_attention = calculate_avg_visual_attention(
                            attention_tensor, img_start_idx, img_end_idx, candidate_idx=candidate_attention_idx
                        )
                        
                        # 分类token
                        token_category = categorize_token_simple(candidate_token_id, tokenizer) if tokenizer is not None else 'content'
                        
                        if train_calibrator:
                            # 训练模式：收集完整数据
                            base_prob = base_probs_i[0, candidate_token_id].item()
                            
                            calibration_data.append({
                                'layer': layer_idx,
                                'position_in_layer': parent_idx * topk_index.shape[1] + child_idx,
                                'candidate_token': candidate_token_id,
                                'draft_confidence': torch.exp(topk_p[parent_idx, child_idx]).item(),
                                'base_confidence': base_prob,
                                'tree_position': layer_positions[parent_idx * topk_index.shape[1] + child_idx].item() if layer_positions is not None else None,
                                'tree_depth': layer_depths[parent_idx * topk_index.shape[1] + child_idx].item() if layer_depths is not None else None,
                                'parent_position': layer_parents[parent_idx * topk_index.shape[1] + child_idx].item() if layer_parents is not None else None,
                                'base_top1_token': int(candidate_token_id == base_top1_token_i),
                                'draft_margin': draft_margin_i,
                                'base_margin': base_margin_i,
                                'avg_visual_attention_intensity': avg_visual_attention,
                                'token_category': token_category
                            })
                        else:
                            # 推理模式：只收集必要信息
                            # 计算 draft model 的 margin (对于当前父节点)
                            draft_sorted_probs_i, draft_sorted_indices_i = torch.sort(torch.exp(topk_p[parent_idx]), descending=True)
                            draft_top1_prob_i = draft_sorted_probs_i[0].item()
                            draft_top2_prob_i = draft_sorted_probs_i[1].item() if len(draft_sorted_probs_i) > 1 else 0.0
                            draft_margin_i = draft_top1_prob_i - draft_top2_prob_i
                            
                            calibration_data.append({
                                'candidate_token': candidate_token_id,
                                'tree_position': layer_positions[parent_idx * topk_index.shape[1] + child_idx].item() if layer_positions is not None else None,
                                'draft_margin': draft_margin_i,
                                'avg_visual_attention_intensity': avg_visual_attention,
                                'token_category': token_category
                            })
    
        except Exception as e:
            print(f"[ERROR] Calibration failed for layer {layer_idx}: {str(e)}")
            import traceback
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
        finally:
            # 恢复模型的原始训练状态（仅在训练模式下）
            if train_calibrator:
                base_model.train(original_training_state)
        
        return calibration_data

    @torch.no_grad()
    def topK_genrate(self, hidden_states, input_ids, head, logits_processor, inputs_embeds=None, enable_candidate_calibration=False, base_model=None, context_past_key_values=None, train_calibrator=False):
        input_ids = input_ids.to(hidden_states.device)
        total_tokens = self.total_tokens
        depth = self.depth
        top_k = self.top_k
        sample_token = input_ids[:, -1]
    
        if inputs_embeds is not None:
            new_embed = self.embed_tokens(sample_token).unsqueeze(dim=0).to(inputs_embeds.device)
            inputs_embeds = torch.cat((inputs_embeds[:, 1:], new_embed), dim=1)
        
        input_ids = input_ids[:, 1:]
        from eagle.model.utils import temp_cache
        
        scores_list = []
        parents_list = []
        ss_token = []
    
        len_posi = input_ids.shape[1]
    
        if (input_ids == -200).any():
            len_posi += 575
    
        self.reset()
        # 存储candidate calibration数据
        candidate_calibration_data = []
        
        # 保存原始context用于candidate calibration
        original_context = input_ids.clone()
        
        # 获取图像token位置信息（用于注意力计算）
        img_start_idx, img_end_idx = None, None
        if CALIBRATION_LOGGING_ENABLED and enable_candidate_calibration:
            from eagle.model.image_token_utils import calculate_image_token_positions_for_calibration
            original_input_ids = input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids
            img_start_idx, img_end_idx = calculate_image_token_positions_for_calibration(
                input_ids=original_input_ids,
                inputs_embeds=inputs_embeds,
                image_features=None,
                batch_idx=0
            )
            # 由于input_ids已经去掉了第一个token，需要调整位置
            if img_start_idx is not None:
                img_start_idx = max(0, img_start_idx - 1)
            if img_end_idx is not None:
                img_end_idx = max(0, img_end_idx - 1)
            # print(f"[DEBUG] Image token positions for attention: [{img_start_idx}:{img_end_idx}]")
        
        if hasattr(self, "stable_kv") and self.stable_kv is not None:
            kv_len = self.stable_kv[0][0].shape[2]
            if -200 in input_ids:
                kv_len -= 575
            
            # 在候选校准模式下获取注意力权重
            if enable_candidate_calibration:
                out_hidden, past_key_values, _, attentions = self(
                    hidden_states, 
                    input_ids=input_ids[:, kv_len:],
                    past_key_values=self.stable_kv, 
                    use_cache=True, 
                    inputs_embeds=inputs_embeds,
                    output_attentions=True
                )
            else:
                out_hidden, past_key_values = self(
                    hidden_states, 
                    input_ids=input_ids[:, kv_len:],
                    past_key_values=self.stable_kv, 
                    use_cache=True, 
                    inputs_embeds=inputs_embeds
                )
                attentions = None
        else:
            image_tokens_num = []
            if -200 in input_ids:
                image_tokens_num = [576]
            elif 151652 in input_ids:
                image_tokens_num = [(input_ids == 151655).sum().item()]
            
            # 在候选校准模式下获取注意力权重
            if enable_candidate_calibration:
                out_hidden, past_key_values, _, attentions = self(
                    hidden_states, 
                    input_ids=input_ids, 
                    use_cache=True, 
                    inputs_embeds=inputs_embeds, 
                    image_tokens_num=image_tokens_num,
                    output_attentions=True
                )
            else:
                out_hidden, past_key_values = self(
                    hidden_states, 
                    input_ids=input_ids, 
                    use_cache=True, 
                    inputs_embeds=inputs_embeds, 
                    image_tokens_num=image_tokens_num
                )
                attentions = None
    
        self.stable_kv = past_key_values
        if CALIBRATION_LOGGING_ENABLED:
            logger = get_calibration_logger()
            
            # 使用新的图像token位置计算方法
            from eagle.model.image_token_utils import calculate_image_token_positions_for_calibration
            
            # 尝试从inputs_embeds推断图像token位置
            original_input_ids = input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids
            
            img_start_idx_log, img_end_idx_log = calculate_image_token_positions_for_calibration(
                input_ids=original_input_ids,
                inputs_embeds=inputs_embeds,
                image_features=None,
                batch_idx=0
            )
            
            # 由于input_ids已经去掉了第一个token，需要调整位置
            if img_start_idx_log is not None:
                img_start_idx_log = max(0, img_start_idx_log - 1)
            if img_end_idx_log is not None:
                img_end_idx_log = max(0, img_end_idx_log - 1)
            # print("img_start_idx, img_end_idx", img_start_idx_log, img_end_idx_log)
            logger.start_draft_session(img_start_idx=img_start_idx_log, img_end_idx=img_end_idx_log)
    
        # 初始化用于后续层的 KV 缓存，避免 UnboundLocalError
        past_key_values = self.stable_kv
    
        last_hidden = out_hidden[:, -1]
        last_headout = head(last_hidden)
        last_p = self.logsoftmax(last_headout)
    
        # --- 第 0 层 ---
        top = torch.topk(last_p, top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values
        scores = topk_p[0]
        scores_list.append(scores[None])
    
        # 记录第0层的局部分数和位置信息（保持原有）
        local_scores_list = []
        token_list = []
        position_labels_list = []
        depth_labels_list = []
        parent_labels_list = []
        next_position_idx = 1
    
        local_scores_list.append(topk_p)
        token_list.append(topk_index)
    
        layer_positions = torch.arange(next_position_idx, next_position_idx + top_k, device=topk_index.device)
        layer_depths = torch.ones(top_k, device=topk_index.device)
        layer_parents = torch.zeros(top_k, device=topk_index.device)
    
        position_labels_list.append(layer_positions)
        depth_labels_list.append(layer_depths)
        parent_labels_list.append(layer_parents)
        next_position_idx += top_k
    
        frontier_paths = [[topk_index[0, p].item()] for p in range(top_k)]
    
        # 使用安全的候选校准函数 - 第0层
        if enable_candidate_calibration and base_model is not None:
            if context_past_key_values is None:
                raise RuntimeError("topK_genrate: context_past_key_values 为 None；请从 initialize_tree 传入初始化的 KVCache。")
        
            layer_0_data = self._collect_calibration_data_safely(
                base_model=base_model,
                original_context=original_context,
                inputs_embeds=inputs_embeds,
                context_past_key_values=context_past_key_values,
                topk_index=topk_index,
                topk_p=topk_p,
                layer_positions=layer_positions,
                layer_depths=layer_depths,
                layer_parents=layer_parents,
                layer_idx=0,
                attentions=attentions,  # 传递注意力权重
                img_start_idx=img_start_idx,
                img_end_idx=img_end_idx,
                train_calibrator=train_calibrator  # 传递训练模式参数
            )
            candidate_calibration_data.extend(layer_0_data)
    
        # 设备统一到 hidden_states.device，避免受 scores 异常影响
        parents_list.append(torch.zeros(1, dtype=torch.long, device=hidden_states.device))
        ss_token.append(topk_index)
        input_ids = topk_index
        input_hidden = last_hidden[None].repeat(1, top_k, 1)
        tree_mask = self.tree_mask_init
        topk_cs_index = torch.arange(top_k, device=self.embed_tokens.weight.device)
    
        # 后续层的处理
        for i in range(depth):
            self.tree_mask = tree_mask
            position_ids = len_posi + self.position_ids
    
            out_hidden, past_key_values, _, layer_attentions = self(
                input_hidden,
                input_ids=input_ids,
                past_key_values=past_key_values,
                position_ids=position_ids,
                use_cache=True,
                output_attentions=True
            )
            len_posi += 1
    
            bias1 = top_k if i > 0 else 0
            bias2 = max(0, i - 1)
            bias = 1 + top_k ** 2 * bias2 + bias1
            parents = (topk_cs_index + bias)
            parents_list.append(parents)
    
            last_headout = head(out_hidden[0])
            last_p = self.logsoftmax(last_headout)
    
            # --- 之后每一层 ---
            top = torch.topk(last_p, top_k, dim=-1)
            topk_index, topk_p = top.indices, top.values
    
            # 记录树信息（保持原有）
            local_scores_list.append(topk_p)
            token_list.append(topk_index)
    
            current_depth = i + 2
            layer_positions = torch.arange(next_position_idx, next_position_idx + top_k * top_k, device=topk_index.device)
            layer_depths = torch.full((top_k * top_k,), current_depth, device=topk_index.device)
    
            parent_positions_for_layer = []
            for parent_idx in range(top_k):
                if i == 0:
                    parent_pos = position_labels_list[0][parent_idx].item()
                else:
                    parent_pos = next_position_idx - top_k * top_k + parent_idx * top_k
                parent_positions_for_layer.extend([parent_pos] * top_k)
            layer_parents = torch.tensor(parent_positions_for_layer, device=topk_index.device)
    
            position_labels_list.append(layer_positions)
            depth_labels_list.append(layer_depths)
            parent_labels_list.append(layer_parents)
    
            # 使用安全的候选校准函数 - 后续层
            if enable_candidate_calibration and base_model is not None:
                layer_i_data = self._collect_calibration_data_safely(
                    base_model=base_model,
                    original_context=original_context,
                    inputs_embeds=inputs_embeds,
                    context_past_key_values=None,  # 后续层不使用KV缓存
                    topk_index=topk_index,
                    topk_p=topk_p,
                    layer_positions=layer_positions,
                    layer_depths=layer_depths,
                    layer_parents=layer_parents,
                    layer_idx=i + 1,
                    frontier_paths=frontier_paths,
                    attentions=layer_attentions,  # 传递注意力权重
                    img_start_idx=img_start_idx,
                    img_end_idx=img_end_idx,
                    train_calibrator=train_calibrator  # 传递训练模式参数
                )
                candidate_calibration_data.extend(layer_i_data)
    
            cu_scores = topk_p + scores[:, None]
            topk_cs = torch.topk(cu_scores.view(-1), top_k, dim=-1)
            topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
            scores = topk_cs_p
    
            out_ids = topk_cs_index // top_k
            input_hidden = out_hidden[:, out_ids]
            input_ids = topk_index.view(-1)[topk_cs_index][None]
    
            ss_token.append(topk_index)
            scores_list.append(cu_scores)
    
            # 更新前沿路径
            new_frontier_paths = []
            for selected_idx in topk_cs_index.tolist():
                parent_idx = selected_idx // top_k
                child_idx = selected_idx % top_k
                selected_token = topk_index[parent_idx, child_idx].item()
                new_path = frontier_paths[parent_idx] + [selected_token]
                new_frontier_paths.append(new_path)
            frontier_paths = new_frontier_paths
    
            next_position_idx += top_k * top_k
            tree_mask = torch.cat((tree_mask[:, :, out_ids], self.tree_mask_init), dim=3)
    
        # 获取所有候选的分数和token
        scores_flat = torch.cat(scores_list, dim=0).reshape(-1)        # 累计分数
        local_scores_flat = torch.cat(local_scores_list, dim=0).reshape(-1)  # 局部分数
        tokens_flat = torch.cat(token_list, dim=0).reshape(-1)         # tokens
        positions_flat = torch.cat(position_labels_list, dim=0).reshape(-1)  # 位置
        depths_flat = torch.cat(depth_labels_list, dim=0).reshape(-1)        # 深度
        parents_flat = torch.cat(parent_labels_list, dim=0).reshape(-1)      # 父节点位置
        
        ss_token_list = torch.cat(ss_token, dim=0).view(-1)
        
        # 选择最终的draft tokens
        top_scores = torch.topk(scores_flat, total_tokens, dim=-1)
        top_scores_index = top_scores.indices
        top_scores_index = torch.sort(top_scores_index).values
    
        draft_tokens = ss_token_list[top_scores_index]
        draft_tokens = torch.cat((sample_token, draft_tokens), dim=0)
        
        # 记录最终被选中的draft tokens的所有信息
        if CALIBRATION_LOGGING_ENABLED:
            logger = get_calibration_logger()
            # 获取最终选中的tokens的各种信息（不包括sample_token）
            selected_path_scores = scores_flat[top_scores_index]      # 路径累计分数
            selected_local_scores = local_scores_flat[top_scores_index]  # 局部分数
            selected_tokens = draft_tokens[1:]                        # 排除sample_token
            selected_positions = positions_flat[top_scores_index]     # 树位置
            selected_depths = depths_flat[top_scores_index]           # 树深度
            selected_parents = parents_flat[top_scores_index]         # 父节点位置
            
            # 记录所有信息
            logger.log_draft_confidence(
                path_confidence_scores=selected_path_scores,
                local_confidence_scores=selected_local_scores,
                draft_tokens=selected_tokens,
                tree_positions=selected_positions,
                tree_depths=selected_depths,
                parent_positions=selected_parents
            )
            
            # 如果启用candidate calibration，记录所有candidate的数据
            if enable_candidate_calibration and candidate_calibration_data:
                logger.log_candidate_calibration_data(candidate_calibration_data)
                print(f"[DEBUG] Recorded {len(candidate_calibration_data)} candidate calibration samples")
                print(f"[DEBUG] Sample calibration data: {candidate_calibration_data[:3]}")

        draft_parents = torch.cat(parents_list, dim=0)[top_scores_index // top_k].long()
        mask_index = torch.searchsorted(top_scores_index, draft_parents - 1, right=False)
        mask_index[draft_parents == 0] = -1
        mask_index = mask_index + 1
        mask_index_list = mask_index.tolist()
        
        tree_mask = torch.eye(total_tokens + 1).bool()
        tree_mask[:, 0] = True
        for i in range(total_tokens):
            tree_mask[i + 1].add_(tree_mask[mask_index_list[i]])
        
        tree_position_ids = torch.sum(tree_mask, dim=1) - 1

        tree_mask = tree_mask.float()[None, None]
        draft_tokens = draft_tokens[None]

        del parents_list, scores_list, ss_token, ss_token_list, draft_parents
        del local_scores_list, token_list, position_labels_list, depth_labels_list, parent_labels_list

        max_depth = torch.max(tree_position_ids) + 1
        noleaf_index = torch.unique(mask_index).tolist()
        noleaf_num = len(noleaf_index) - 1
        leaf_num = total_tokens - noleaf_num

        retrieve_indices = torch.zeros(leaf_num, max_depth.item(), dtype=torch.long) - 1
        retrieve_indices = retrieve_indices.tolist()

        rid = 0
        position_ids_list = tree_position_ids.tolist()

        for i in range(total_tokens + 1):
            if i not in noleaf_index:
                cid = i
                depth = position_ids_list[i]
                for j in reversed(range(depth + 1)):
                    retrieve_indices[rid][j] = cid
                    cid = mask_index_list[cid - 1]
                rid += 1

        if logits_processor is not None:
            maxitem = total_tokens + 5

            def custom_sort(lst):
                sort_keys = []
                for i in range(len(lst)):
                    sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
                return sort_keys

            retrieve_indices = sorted(retrieve_indices, key=custom_sort)

        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
        del mask_index, mask_index_list, noleaf_index, noleaf_num, leaf_num, max_depth, rid
        tree_position_ids = tree_position_ids.to(hidden_states.device)
        
        # print(f"[DEBUG] topK_genrate completed successfully")
        
        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids

    @torch.no_grad()
    def acc(self, data, head, max_length=5):
        hidden_states = data["hidden_states"]
        input_ids = data["input_ids"]
        # attention_mask=data["attention_mask"]
        loss_mask = data["loss_mask"]
        sample_mask = data["sample_mask"]
        target = data["target"]
        total = [0 for _ in range(max_length)]
        correct = [0 for _ in range(max_length)]
        bs, sl = hidden_states.shape[0], hidden_states.shape[1]
        target_headout = head(target)
        hidden_states_headout = head(hidden_states)

        for i in range(bs):
            for j in range(sl):
                if loss_mask[i, j] == 0:
                    continue
                single_hidden_states = hidden_states[i, :j]
                single_input_ids = input_ids[i, :j]

                single_hidden_states = single_hidden_states[None, :, :]
                single_input_ids = single_input_ids[None, :]
                for k in range(max_length):
                    tmp_in_target_headout = hidden_states_headout[i, single_hidden_states.shape[1] - 1]
                    tmp_out_target_headout = target_headout[i, single_hidden_states.shape[1] - 1]
                    target_in_token = torch.argmax(tmp_in_target_headout)
                    target_out_token = torch.argmax(tmp_out_target_headout)
                    tmp_token = input_ids[i, single_hidden_states.shape[1] - 1]
                    tmp_sample_mask = sample_mask[i, single_hidden_states.shape[1] - 1]
                    if not (target_in_token == tmp_token):
                        break
                    out_hidden = self(single_hidden_states, input_ids=single_input_ids)
                    last_hidden = out_hidden[:, -1]
                    last_headout = head(last_hidden)
                    token = torch.argmax(last_headout)
                    total[k] += 1
                    if token == target_out_token:
                        correct[k] += 1
                    else:
                        for kk in range(k, max_length):
                            total[kk] += 1
                        break

                    single_hidden_states = torch.cat((single_hidden_states, out_hidden[:, -1:]), dim=1)
                    single_input_ids = torch.cat(
                        (single_input_ids, torch.tensor([[token]]).to(single_input_ids.device)), dim=1)

        acc = [correct[i] / total[i] for i in range(len(correct))]
        return acc


class Vhead(nn.Module):
    def __init__(self, ins=6566, outs=32000):
        super().__init__()
        self.fc = nn.Linear(ins, outs, bias=False)

    def forward(self, x):
        return self.fc(x)


import torch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    config = EConfig.from_pretrained('config.json')
    model = Model(config, load_emb=False)
    print(model)