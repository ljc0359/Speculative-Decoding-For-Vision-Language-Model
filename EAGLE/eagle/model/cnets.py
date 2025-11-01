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
    def __init__(self, config, load_emb=False, path=None, bias=True, total_tokens=63, depth=5, top_k=8, threshold=1.0, train_embed=False, decouple=False, draft_temperature=1.0):
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
        self.draft_temperature = draft_temperature  # Temperature scaling for draft model logits

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
                            'layer': layer_idx,
                            'tree_position': layer_positions[child_idx].item(),
                            'tree_depth': layer_depths[child_idx].item() if layer_depths is not None else 0,
                            'parent_position': layer_parents[child_idx].item() if layer_parents is not None else -1,
                            'candidate_token': candidate_token_id,
                            'draft_confidence': torch.exp(topk_p[0, child_idx]).item(),
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
                                'layer': layer_idx,
                                'tree_depth': layer_depths[parent_idx * topk_index.shape[1] + child_idx].item() if layer_depths is not None else None,
                                'parent_position': layer_parents[parent_idx * topk_index.shape[1] + child_idx].item() if layer_parents is not None else None,
                                'candidate_token': candidate_token_id,
                                'draft_confidence': torch.exp(topk_p[parent_idx, child_idx]).item(),
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
    def topK_genrate(
        self,
        hidden_states,
        input_ids,
        head,
        logits_processor,
        inputs_embeds=None,
        enable_candidate_calibration=False,
        base_model=None,
        context_past_key_values=None,
        train_calibrator=False,
        use_calibrator=False,
        calibrator=None,
        alpha=1,
        nodes: Optional[int] = 500,
        threshold: Optional[float] =  0.05,
        max_depth: Optional[int] = 20,
        print_time: bool = False,
    ):
        # print("train_calibrator", train_calibrator)
        
        # 重置树掩码状态
        self.reset()

        # -------- OPT-Tree 参数接入：预算与阈值覆盖 --------
        effective_total_tokens = self.total_tokens if nodes is None else max(1, int(nodes) - 1)
        effective_max_depth = self.depth if max_depth is None else int(max_depth)
        # 阈值直接使用，不取对数（因为权重增量本身就在对数空间）
        effective_threshold = self.threshold if threshold is None else float(threshold)

        # -------- 辅助函数：上下文步前向 --------
        def run_context_forward(h_states, ids, embeds, want_attn=False):
            image_tokens_num = []
            if -200 in ids:
                image_tokens_num = [576]
            elif 151652 in ids:
                image_tokens_num = [(ids == 151655).sum().item()]

            if hasattr(self, "stable_kv") and self.stable_kv is not None:
                kv_len = self.stable_kv[0][0].shape[2]
                if -200 in ids:
                    kv_len -= 575
                if want_attn:
                    out_h, pkv, _, attn = self(
                        h_states,
                        input_ids=ids[:, kv_len:],
                        past_key_values=self.stable_kv,
                        use_cache=True,
                        inputs_embeds=embeds,
                        output_attentions=True,
                    )
                else:
                    out_h, pkv = self(
                        h_states,
                        input_ids=ids[:, kv_len:],
                        past_key_values=self.stable_kv,
                        use_cache=True,
                        inputs_embeds=embeds,
                    )
                    attn = None
            else:
                if want_attn:
                    out_h, pkv, _, attn = self(
                        h_states,
                        input_ids=ids,
                        use_cache=True,
                        inputs_embeds=embeds,
                        image_tokens_num=image_tokens_num,
                        output_attentions=True,
                    )
                else:
                    out_h, pkv = self(
                        h_states,
                        input_ids=ids,
                        use_cache=True,
                        inputs_embeds=embeds,
                        image_tokens_num=image_tokens_num,
                    )
                    attn = None
            return out_h, pkv, attn

        # -------- 辅助函数：校准器重排（暂时禁用，保留接口）--------
        def select_with_calibrator(pre_idx, pre_scores_logp, per_row, layer_features):
            # 暂时禁用校准器功能，直接返回 top-k 选择
            chosen = torch.topk(pre_scores_logp, per_row, dim=-1)
            return pre_idx.gather(dim=-1, index=chosen.indices), pre_scores_logp.gather(dim=-1, index=chosen.indices)
        
        # -------- 初始化输入与 KV --------
        input_ids = input_ids.to(hidden_states.device)
        top_k = min(self.top_k, effective_total_tokens)
        
        # 恢复上下文与 KV 初始化
        sample_token = input_ids[:, -1]
        if inputs_embeds is not None:
            new_embed = self.embed_tokens(sample_token).unsqueeze(0).to(inputs_embeds.device)
            inputs_embeds = torch.cat((inputs_embeds[:, 1:], new_embed), dim=1)
        input_ids = input_ids[:, 1:]
        pos_base_len = input_ids.shape[1]
        if (input_ids == -200).any():
            pos_base_len += 575
        
        # 暂时禁用校准数据收集
        want_attn = False
        out_hidden, past_key_values, attentions = run_context_forward(hidden_states, input_ids, inputs_embeds, want_attn=want_attn)
        self.stable_kv = past_key_values
        
        # -------- OPT-Tree 核心实现：全局前沿贪心选择 --------
        
        # 初始化 OPT-Tree 权重矩阵和全局状态
        weight_matrix = torch.zeros([effective_max_depth, top_k], device=hidden_states.device)
        input_ids_matrix = torch.zeros([effective_max_depth, top_k], dtype=torch.long, device=hidden_states.device)
        parents_matrix = torch.zeros([effective_max_depth, top_k], dtype=torch.long, device=hidden_states.device)
        
        current_depth = 0
        global_weight_sum = 0.0
        
        # 初始层：获取第一层候选
        last_hidden = out_hidden[:, -1]
        last_logits = head(last_hidden)
        
        # 应用温度缩放到草稿模型 logits
        if hasattr(self, 'draft_temperature') and self.draft_temperature != 1.0:
            last_logits = last_logits / self.draft_temperature
        
        # 使用 softmax 而不是 log_softmax（与 opt_eagle 一致）
        last_probs = torch.softmax(last_logits, dim=-1, dtype=torch.float32)
        
        # 初始层选择 top-k
        init_top = torch.topk(last_probs[0], top_k, dim=-1)
        init_indices, init_weights = init_top.indices, init_top.values
        
        # 存储到权重矩阵
        weight_matrix[current_depth] = init_weights
        input_ids_matrix[current_depth] = init_indices
        parents_matrix[current_depth] = torch.arange(top_k, device=hidden_states.device)
        
        current_depth += 1
        
        # 初始化树掩码和位置
        self.init_tree()
        tree_mask = self.tree_mask_init
        current_ids = init_indices.unsqueeze(0)
        current_hidden = last_hidden.unsqueeze(0).repeat(1, top_k, 1)
        
        # -------- 逐层扩展：全局前沿贪心选择 --------
        for layer_i in range(effective_max_depth - 1):
            if current_depth >= effective_max_depth:
                break
                
            # 前向传播当前层
            self.tree_mask = tree_mask
            position_ids = pos_base_len + self.position_ids
            
            out_h, past_key_values = self(
                current_hidden,
                input_ids=current_ids,
                past_key_values=past_key_values,
                position_ids=position_ids,
                use_cache=True,
            )
            pos_base_len += 1
            
            # 获取每个节点的 logits
            layer_logits = head(out_h[0])  # [top_k, vocab_size]
            
            # 获取当前层的 logits 并转换为概率
            if hasattr(self, 'draft_temperature') and self.draft_temperature != 1.0:
                layer_logits = layer_logits / self.draft_temperature
            
            # 使用 softmax 获取概率（与 opt_eagle 一致）
            layer_probs = torch.softmax(layer_logits, dim=-1, dtype=torch.float32)
            
            # 获取每个父节点的 top-k 候选
            candidates_probs, candidates_ids = torch.topk(layer_probs, top_k, dim=-1)  # [top_k, top_k]
            
            # 计算路径权重：父节点权重 × 子节点概率（概率空间乘法）
            parent_weights = weight_matrix[current_depth - 1].unsqueeze(1)  # [top_k, 1]
            path_weights = parent_weights * candidates_probs  # [top_k, top_k] (probability space)
            
            # 全局前沿贪心选择：在所有候选中选择 top-k
            flat_weights = path_weights.view(-1)  # [top_k * top_k]
            flat_ids = candidates_ids.view(-1)  # [top_k * top_k]
            
            global_top_weights, global_top_idx = torch.topk(flat_weights, top_k, dim=-1)
            selected_ids = flat_ids[global_top_idx]
            selected_parents = global_top_idx // top_k
            
            # 存储到权重矩阵
            weight_matrix[current_depth] = global_top_weights
            input_ids_matrix[current_depth] = selected_ids
            parents_matrix[current_depth] = selected_parents
            
            # 计算全局权重和 E[A] 估计（与 opt_eagle 一致）
            if current_depth > 0:
                # 计算当前层的全局最优权重和
                historical_weights = weight_matrix[:current_depth].view(-1)  # 展平历史权重
                top_historical_weights, _ = torch.topk(historical_weights, min(effective_total_tokens, len(historical_weights)), dim=-1)
                new_global_weight_sum = top_historical_weights.sum().item()
                
                # 阈值驱动的动态终止：检查权重增量是否足够大（opt_eagle 逻辑）
                weight_increment = new_global_weight_sum - global_weight_sum
                if weight_increment <= effective_threshold:
                    print(f"OPT-Tree: 动态终止于深度 {current_depth}, 权重增量 {weight_increment:.4f} <= 阈值 {effective_threshold:.4f}")
                    break
                
                global_weight_sum = new_global_weight_sum
            else:
                # 第一层：初始化全局权重和
                global_weight_sum = global_top_weights.sum().item()
            
            current_depth += 1
            
            # 更新下一层的输入
            current_ids = selected_ids.unsqueeze(0)
            current_hidden = out_h[:, selected_parents]
            
            # 更新树掩码（简化版本，用于注意力）
            # 这里需要根据父子关系构建新的掩码
            # 为简化，我们使用基本的因果掩码
            
        # -------- 最终树构建：基于权重矩阵重建最优路径 --------
        
        # 从权重矩阵中选择全局最优路径
        final_depth = current_depth
        all_weights = weight_matrix[:final_depth].view(-1)
        all_positions = torch.arange(len(all_weights), device=hidden_states.device)
        
        # 选择全局最优节点（使用 effective_total_tokens 而不是 top_k）
        final_top_weights, final_top_positions = torch.topk(all_weights, min(effective_total_tokens, len(all_weights)), dim=-1)
        
        # 解码位置到层和节点索引
        final_layers = final_top_positions // top_k
        final_nodes = final_top_positions % top_k
        
        # 构建最终的 token 序列
        draft_tokens_list = [sample_token.item()]
        parent_pointers = [0]  # sample_token 的父指针为 0
        
        # 按层排序并构建路径
        sorted_indices = torch.argsort(final_layers)
        final_layers = final_layers[sorted_indices]
        final_nodes = final_nodes[sorted_indices]
        
        for i, (layer_idx, node_idx) in enumerate(zip(final_layers, final_nodes)):
            token_id = input_ids_matrix[layer_idx, node_idx].item()
            draft_tokens_list.append(token_id)
            
            if layer_idx == 0:
                parent_pointers.append(0)  # 第一层的父节点是 sample_token
            else:
                # 找到父节点在已构建序列中的位置
                parent_node_idx = parents_matrix[layer_idx, node_idx].item()
                parent_layer_idx = layer_idx - 1
                
                # 在已构建的序列中找到对应的父节点
                parent_pos = 0
                for j in range(i):
                    if final_layers[j] == parent_layer_idx and final_nodes[j] == parent_node_idx:
                        parent_pos = j + 1
                        break
                parent_pointers.append(parent_pos)
        
        # 构建最终输出
        total_tokens = len(draft_tokens_list) - 1  # 不包括 sample_token
        draft_tokens = torch.tensor(draft_tokens_list, dtype=torch.long, device=hidden_states.device).unsqueeze(0)
        
        # 构建树掩码
        tree_mask_bool = torch.eye(len(draft_tokens_list)).bool()
        tree_mask_bool[:, 0] = True  # 所有节点都能看到 sample_token
        
        for i in range(1, len(draft_tokens_list)):
            parent_idx = parent_pointers[i]
            tree_mask_bool[i] = tree_mask_bool[i] | tree_mask_bool[parent_idx]
        
        tree_position_ids = (torch.sum(tree_mask_bool, dim=1) - 1).to(hidden_states.device)
        tree_mask = tree_mask_bool.float().unsqueeze(0).unsqueeze(0)
        
        # 构建 retrieve_indices
        max_depth_val = torch.max(tree_position_ids).item() + 1
        noleaf_indices = torch.unique(torch.tensor(parent_pointers[1:])).tolist()  # 非叶子节点
        noleaf_indices = [0] + [idx for idx in noleaf_indices if idx != 0]  # 包含 sample_token
        
        leaf_indices = [i for i in range(len(draft_tokens_list)) if i not in noleaf_indices]
        leaf_num = len(leaf_indices)
        
        retrieve_indices = torch.full((leaf_num, max_depth_val), -1, dtype=torch.long)
        
        for rid, leaf_idx in enumerate(leaf_indices):
            current_idx = leaf_idx
            depth = tree_position_ids[leaf_idx].item()
            
            for j in reversed(range(depth + 1)):
                retrieve_indices[rid, j] = current_idx
                if current_idx > 0:
                    current_idx = parent_pointers[current_idx]
                else:
                    break
        
        # 排序 retrieve_indices（如果需要）
        if logits_processor is not None:
            maxitem = len(draft_tokens_list) + 5
            def custom_sort(lst):
                return [v if v >= 0 else maxitem for v in lst]
            retrieve_indices = retrieve_indices[sorted(range(len(retrieve_indices)), 
                                                    key=lambda i: custom_sort(retrieve_indices[i].tolist()))]
        
        print(f"OPT-Tree: 生成了 {total_tokens} 个草稿 tokens，最终深度 {final_depth}，全局权重和 {global_weight_sum:.4f}")
        
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
                    last_headout = head(last_headout)
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