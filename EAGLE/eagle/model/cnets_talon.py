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
from typing import List, Optional, Tuple, Union
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
import os, sys
try:
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
    if _repo_root not in sys.path:
        sys.path.append(_repo_root)
    from MSD.TALON.utils import node_scores, mc_node_scores_from_logits, mc_stats_from_logits, dual_uncertainty_alignment
except Exception:
    node_scores = None
import os, sys
try:
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
    if _repo_root not in sys.path:
        sys.path.append(_repo_root)
    from MSD.TALON.utils import node_scores, mc_node_scores_from_logits, mc_stats_from_logits, dual_uncertainty_alignment
except Exception:
    node_scores = None
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS


from eagle.model.configs import EConfig, Qwen2VLConfig
from eagle.model.choices import *
from eagle.model.utils import prepare_logits_processor




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
    def __init__(self, config, load_emb=False, path=None, bias=True, total_tokens=63, depth=5, top_k=8, threshold=1.0, train_embed=False, decouple=False, use_uncertainty_scoring=True, uncertainty_stride: int =1, score_a: float = 1.0, score_b=0.1, score_c: float = 0.0, score_d: float = 0.4, use_js: bool = False, use_epi: bool = False, reorder_leaves: bool = False, use_mc_alea_epi: bool = True, mc_samples: int = 8, mc_noise_std: float = 0.3, mc_temperature: float = 1.0, mc_kind: str = "gauss", epi_threshold: float = 5.0, alea_threshold: float = 5.0, epi_center: float = 0.5, alea_center: float = 0.5, exploit_bonus: float = 3.0, explore_penalty: float = -0.3, balance_factor: float = 0.5, uncertain_penalty: float = -0.8):
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
        self.use_uncertainty_scoring = use_uncertainty_scoring
        self.uncertainty_stride = max(1, int(uncertainty_stride))
        self.score_a = float(score_a)
        self.score_b = float(score_b)
        self.score_c = float(score_c)
        self.score_d = float(score_d)
        self.use_js = bool(use_js)
        self.use_epi = bool(use_epi)
        self.reorder_leaves = bool(reorder_leaves)
        self.use_mc_alea_epi = bool(use_mc_alea_epi)
        self.mc_samples = int(mc_samples)
        self.mc_noise_std = float(mc_noise_std)
        self.mc_temperature = float(mc_temperature)
        self.mc_kind = str(mc_kind)
        
        # 双重不确定性参数
        self.epi_threshold = float(epi_threshold)
        self.alea_threshold = float(alea_threshold)
        self.epi_center = float(epi_center)
        self.alea_center = float(alea_center)
        self.exploit_bonus = float(exploit_bonus)
        self.explore_penalty = float(explore_penalty)
        self.balance_factor = float(balance_factor)
        self.uncertain_penalty = float(uncertain_penalty)
        
        # 添加参数日志
        # print(f"[TALON] Model initialized with dual uncertainty params:")
        # print(f"  use_mc_alea_epi: {self.use_mc_alea_epi}")
        # print(f"  epi_threshold: {self.epi_threshold}, alea_threshold: {self.alea_threshold}")
        # print(f"  epi_center: {self.epi_center}, alea_center: {self.alea_center}")
        # print(f"  exploit_bonus: {self.exploit_bonus}, explore_penalty: {self.explore_penalty}")
        # print(f"  balance_factor: {self.balance_factor}, uncertain_penalty: {self.uncertain_penalty}")
        # print(f"  score_a: {self.score_a}, score_b: {self.score_b}, score_c: {self.score_c}, score_d: {self.score_d}")

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
            std=None
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

        if use_cache:
            return hidden_states, next_decoder_cache

        return hidden_states

    def reset_kv(self):
        self.stable_kv = None

    @torch.no_grad()
    def topK_genrate(self, hidden_states, input_ids, head, logits_processor, inputs_embeds=None):

        input_ids = input_ids.to(hidden_states.device)
        total_tokens = self.total_tokens
        depth = self.depth
        top_k = self.top_k

        sample_token = input_ids[:, -1]

        if inputs_embeds != None:
            new_embed = self.embed_tokens(sample_token).unsqueeze(dim=0).to(inputs_embeds.device)
            inputs_embeds = torch.cat((inputs_embeds[:,1:],new_embed),dim=1)
        from eagle.model.utils import temp_cache

        scores_list = []
        parents_list = []
        ss_token = []

        input_ids = input_ids[:, 1:]
        input_ids = input_ids.to(hidden_states.device)

        len_posi = input_ids.shape[1]

        if -200 in input_ids:  # Indicates that an image exists, the actual length is 575
            len_posi += 575

        self.reset()

        if hasattr(self, "stable_kv") and self.stable_kv is not None:
            kv_len = self.stable_kv[0][0].shape[2]
            if -200 in input_ids:
                kv_len -= 575
            out_hidden, past_key_values = self(hidden_states, input_ids=input_ids[:, kv_len:],
                                               past_key_values=self.stable_kv, use_cache=True, inputs_embeds=inputs_embeds)
        else:
            image_tokens_num = []
            if -200 in input_ids:
                image_tokens_num = [576]
            elif 151652 in input_ids:
                image_tokens_num = [(input_ids == 151655).sum().item()]
            out_hidden, past_key_values = self(hidden_states, input_ids=input_ids, use_cache=True, inputs_embeds=inputs_embeds, image_tokens_num=image_tokens_num)

        self.stable_kv = past_key_values
        last_hidden = out_hidden[:, -1]

        last_headout = head(last_hidden)

        last_p = self.logsoftmax(last_headout)
        top = torch.topk(last_p, top_k, dim=-1)
        topk_index, topk_logp = top.indices, top.values  # log-probs for K tokens

        if self.use_uncertainty_scoring:
            if node_scores is not None and self.use_mc_alea_epi:
                # 双重不确定性驱动的动态树策略
                logits = last_headout  # [1, V]
                p_bar, H_bar_n, H_epi_n, H_alea_n, js_n = mc_stats_from_logits(
                    logits=logits,
                    num_samples=self.mc_samples,
                    noise_std=self.mc_noise_std,
                    temperature=self.mc_temperature,
                    kind=self.mc_kind,
                    use_js=self.use_js,
                )  # [1,V], [1], [1], [1], [1]
                
                # 使用双重不确定性计算对齐奖励
                p_align = dual_uncertainty_alignment(
                    topk_logp=topk_logp,
                    H_epi_n=H_epi_n,
                    H_alea_n=H_alea_n,
                    epi_threshold=self.epi_threshold,
                    alea_threshold=self.alea_threshold,
                    epi_center=self.epi_center,
                    alea_center=self.alea_center,
                    exploit_bonus=self.exploit_bonus,
                    explore_penalty=self.explore_penalty,
                    balance_factor=self.balance_factor,
                    uncertain_penalty=self.uncertain_penalty,
                )  # [1, K]
                
                # 全局惩罚项（保持兼容性）
                penalties = (self.score_b * H_bar_n + self.score_d * H_epi_n + self.score_c * js_n)[..., None]
                scores = topk_logp + self.score_a * p_align - penalties
                scores = scores[0]
            else:
                # 当不使用MC或node_scores不可用时，直接使用原始log概率（与baseline一致）
                scores = topk_logp[0]
        else:
            scores = topk_logp[0]
        scores_list.append(scores[None])
        parents_list.append(torch.zeros(1, dtype=torch.long, device=scores.device))
        ss_token.append(topk_index)
        input_ids = topk_index
        input_hidden = last_hidden[None].repeat(1, top_k, 1)
        tree_mask = self.tree_mask_init
        topk_cs_index = torch.arange(top_k, device=self.embed_tokens.weight.device)

        # 4
        for i in range(depth):
            self.tree_mask = tree_mask
            position_ids = len_posi + self.position_ids

            out_hidden, past_key_values = self(input_hidden, input_ids=input_ids, past_key_values=past_key_values,
                                               position_ids=position_ids, use_cache=True)
            len_posi += 1

            # with Timer("sort1"):
            bias1 = top_k if i > 0 else 0
            bias2 = max(0, i - 1)
            bias = 1 + top_k ** 2 * bias2 + bias1
            parents = (topk_cs_index + bias)
            parents_list.append(parents)

            last_headout = head(out_hidden[0])
            last_p = self.logsoftmax(last_headout)

            top = torch.topk(last_p, top_k, dim=-1)
            topk_index, topk_logp = top.indices, top.values

            if self.use_uncertainty_scoring and (i % self.uncertainty_stride == 0):
                if node_scores is not None and self.use_mc_alea_epi:
                    # 对每个父节点行，使用双重不确定性策略
                    logits = last_headout  # [top_k, V]
                    p_bar, H_bar_n, H_epi_n, H_alea_n, js_n = mc_stats_from_logits(
                        logits=logits,
                        num_samples=self.mc_samples,
                        noise_std=self.mc_noise_std,
                        temperature=self.mc_temperature,
                        kind=self.mc_kind,
                        use_js=self.use_js,
                    )  # [top_k,V], [top_k], [top_k], [top_k], [top_k]
                    
                    # 使用双重不确定性计算对齐奖励
                    p_align = dual_uncertainty_alignment(
                        topk_logp=topk_logp,
                        H_epi_n=H_epi_n,
                        H_alea_n=H_alea_n,
                        epi_threshold=self.epi_threshold,
                        alea_threshold=self.alea_threshold,
                        epi_center=self.epi_center,
                        alea_center=self.alea_center,
                        exploit_bonus=self.exploit_bonus,
                        explore_penalty=self.explore_penalty,
                        balance_factor=self.balance_factor,
                        uncertain_penalty=self.uncertain_penalty,
                    )  # [top_k, K]
                    
                    penalties = (self.score_b * H_bar_n + self.score_d * H_epi_n + self.score_c * js_n)[:, None]
                    local_scores = topk_logp + self.score_a * p_align - penalties
                    cu_scores = local_scores + scores[:, None]
                else:
                    # 当不使用MC时，直接使用原始log概率（与baseline一致）
                    cu_scores = topk_logp + scores[:, None]
            else:
                cu_scores = topk_logp + scores[:, None]

            topk_cs = torch.topk(cu_scores.view(-1), top_k, dim=-1)
            topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
            scores = topk_cs_p

            out_ids = topk_cs_index // top_k
            input_hidden = out_hidden[:, out_ids]
            # with Timer("2index"):
            #     in_ids = topk_cs_index % top_k
            #     input_ids = topk_index[out_ids, in_ids][None]
            # with Timer("1index"):
            input_ids = topk_index.view(-1)[topk_cs_index][None]
            # print(input_ids.equal(input_ids0))

            ss_token.append(topk_index)
            scores_list.append(cu_scores)
            tree_mask = torch.cat((tree_mask[:, :, out_ids], self.tree_mask_init), dim=3)

            # if self.threshold < 0 and cu_scores.max() < self.threshold:
            #     break

        # del parents_list,scores_list,ss_token
        # return draft_tokens, mask_index,tree_mask,tree_position_ids

        # with Timer("post"):

        scores_list = torch.cat(scores_list, dim=0).view(-1)
        ss_token_list = torch.cat(ss_token, dim=0).view(-1)
        top_scores = torch.topk(scores_list, total_tokens, dim=-1)
        top_scores_index = top_scores.indices
        top_scores_index = torch.sort(top_scores_index).values

        draft_tokens = ss_token_list[top_scores_index]
        draft_tokens = torch.cat((sample_token, draft_tokens), dim=0)

        draft_parents = torch.cat(parents_list, dim=0)[top_scores_index // top_k].long()
        mask_index = torch.searchsorted(top_scores_index, draft_parents - 1, right=False)
        # mask_index[(top_scores_index[mask_index]!=draft_parents - 1)]=-1
        mask_index[draft_parents == 0] = -1
        mask_index = mask_index + 1
        mask_index_list = mask_index.tolist()
        # with Timer("mask"):
        tree_mask = torch.eye(total_tokens + 1).bool()
        tree_mask[:, 0] = True
        for i in range(total_tokens):
            tree_mask[i + 1].add_(tree_mask[mask_index_list[i]])

        tree_position_ids = torch.sum(tree_mask, dim=1) - 1

        tree_mask = tree_mask.float()[None, None]
        draft_tokens = draft_tokens[None]

        del parents_list, ss_token, ss_token_list, draft_parents

        # with Timer("retrieve"):

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

        # Reorder leaves by cumulative path scores (V_path)
        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
        if self.reorder_leaves:
            selected_values = scores_list[top_scores_index]
            selected_values_cpu = selected_values.detach().to('cpu', dtype=torch.float32)
            node_values = torch.cat([
                torch.zeros(1, dtype=torch.float32),
                selected_values_cpu
            ], dim=0)
            lengths = (retrieve_indices >= 0).sum(dim=1)
            leaf_ids = retrieve_indices[torch.arange(leaf_num, device=retrieve_indices.device), lengths - 1]
            leaf_ids = leaf_ids.clamp_min(0).clamp_max(node_values.shape[0] - 1)
            leaf_scores = node_values[leaf_ids]
            order = torch.argsort(leaf_scores, descending=True)
            retrieve_indices = retrieve_indices[order]
        del mask_index, mask_index_list, noleaf_index, noleaf_num, leaf_num, max_depth, rid
        tree_position_ids = tree_position_ids.to(hidden_states.device)

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
