def topK_genrate(self, hidden_states, input_ids, head, logits_processor, inputs_embeds=None, enable_candidate_calibration=False, base_model=None, context_past_key_values=None, train_calibrator=False, use_calibrator=False, calibrator=None, alpha=0.2):
    // ... existing code ...
    # 稳健性参数（可放在函数顶部统一定义）
    PROB_FLOOR = 1e-3
    MAX_CALIB_LOGIT = 3.0
    SORT_ONLY_BIAS = True
    // ... existing code ...

    # ---------- 第 0 层 ----------
    # 当不使用校准器或校准器为空时，直接回到原始 top_k 选择（不走预选集合）
    if (not use_calibrator) or (calibrator is None) or train_calibrator:
        top = torch.topk(last_p, top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values
        scores = topk_p[0]
        scores_list.append(scores[None])
    else:
        # 保持“先预选、再校准、再重选”的逻辑，但 sort-only 模式下 topk_p 用原始 last_p.gather
        preselect_k = preselect_Scale * top_k
        pre_top = torch.topk(last_p, preselect_k, dim=-1)
        preselect_index, preselect_p = pre_top.indices, pre_top.values

        // ... existing code ...
        if use_calibrator and calibrator is not None and not train_calibrator and (layer_0_precal_data is not None):
            try:
                import pandas as pd, numpy as np
                original_preselect_p = preselect_p.clone()
                features_df = pd.DataFrame(layer_0_precal_data)
                calibrated_probs = calibrator.predict_proba(features_df)
                calibrated_probs = np.clip(calibrated_probs, PROB_FLOOR, 1.0 - PROB_FLOOR)

                alpha_mat = _compute_adaptive_alpha(
                    data_list=layer_0_precal_data,
                    base_alpha=float(alpha),
                    layer_idx=0,
                    shape=(1, preselect_k),
                    device=preselect_p.device,
                    dtype=preselect_p.dtype
                )

                calibrated_logits_tensor = torch.tensor(
                    np.log(calibrated_probs) - np.log(1.0 - calibrated_probs),
                    device=preselect_p.device, dtype=preselect_p.dtype
                ).view(1, preselect_k)
                calibrated_logits_tensor = torch.clamp(calibrated_logits_tensor, -MAX_CALIB_LOGIT, MAX_CALIB_LOGIT)

                if SORT_ONLY_BIAS:
                    fused_score = original_preselect_p + alpha_mat * calibrated_logits_tensor
                    reselect = torch.topk(fused_score, top_k, dim=-1)
                    topk_index = preselect_index.gather(dim=-1, index=reselect.indices)
                    # 关键修复：topk_p 用原始 last_p（概率的 log 值），而非 reselect.values
                    topk_p = last_p.gather(dim=-1, index=topk_index)
                else:
                    last_headout = last_headout.clone()
                    bias_vec = alpha_mat * calibrated_logits_tensor
                    last_headout.scatter_add_(dim=-1, index=preselect_index, src=bias_vec)
                    last_p = self.logsoftmax(last_headout)
                    candidate_scores = last_p.gather(dim=-1, index=preselect_index)
                    reselect = torch.topk(candidate_scores, top_k, dim=-1)
                    topk_index = preselect_index.gather(dim=-1, index=reselect.indices)
                    topk_p = reselect.values

                scores = topk_p[0]
                scores_list.append(scores[None])
            except Exception as e:
                # 回退：预选集合上直接取 top_k（与原始 top_k 等价）
                reselect = torch.topk(preselect_p, top_k, dim=-1)
                topk_index = preselect_index.gather(dim=-1, index=reselect.indices)
                topk_p = last_p.gather(dim=-1, index=topk_index) if SORT_ONLY_BIAS else reselect.values
                scores = topk_p[0]
                scores_list.append(scores[None])
        else:
            # 未收集到校准特征时，预选集合直接重选最终 top_k
            reselect = torch.topk(preselect_p, top_k, dim=-1)
            topk_index = preselect_index.gather(dim=-1, index=reselect.indices)
            # 保持概率语义：topk_p 用 last_p.gather
            topk_p = last_p.gather(dim=-1, index=topk_index)
            scores = topk_p[0]
            scores_list.append(scores[None])

    // ... existing code ...

    # ---------- 后续各层 ----------
    for i in range(depth):
        // ... existing code ...
        # 当不使用校准器或校准器为空时，直接回到原始 top_k 选择（不走预选集合）
        if (not use_calibrator) or (calibrator is None) or train_calibrator:
            top = torch.topk(last_p, top_k, dim=-1)
            topk_index, topk_p = top.indices, top.values
            local_scores_list.append(topk_p)
            token_list.append(topk_index)
        else:
            preselect_k = preselect_Scale * top_k
            pre_top = torch.topk(last_p, preselect_k, dim=-1)
            preselect_index, preselect_p = pre_top.indices, pre_top.values

            // ... existing code ...

            if use_calibrator and calibrator is not None and not train_calibrator and (layer_i_data is not None):
                try:
                    import pandas as pd, np as np
                    original_preselect_p = preselect_p.clone()
                    features_df = pd.DataFrame(layer_i_data)
                    calibrated_probs = calibrator.predict_proba(features_df)
                    calibrated_probs = np.clip(calibrated_probs, PROB_FLOOR, 1.0 - PROB_FLOOR)

                    alpha_mat = _compute_adaptive_alpha(
                        data_list=layer_i_data,
                        base_alpha=alpha,
                        layer_idx=i + 1,
                        shape=(top_k, preselect_k),
                        device=preselect_p.device,
                        dtype=preselect_p.dtype
                    )

                    calibrated_logits_tensor = torch.tensor(
                        np.log(calibrated_probs) - np.log(1.0 - calibrated_probs),
                        device=preselect_p.device, dtype=preselect_p.dtype
                    ).view(top_k, preselect_k)
                    calibrated_logits_tensor = torch.clamp(calibrated_logits_tensor, -MAX_CALIB_LOGIT, MAX_CALIB_LOGIT)

                    if SORT_ONLY_BIAS:
                        fused_score = original_preselect_p + alpha_mat * calibrated_logits_tensor
                        reselect = torch.topk(fused_score, top_k, dim=-1)
                        topk_index = preselect_index.gather(dim=-1, index=reselect.indices)
                        # 关键修复：topk_p 用 last_p.gather
                        topk_p = last_p.gather(dim=-1, index=topk_index)
                    else:
                        last_headout = last_headout.clone()
                        bias_mat = alpha_mat * calibrated_logits_tensor
                        last_headout.scatter_add_(dim=-1, index=preselect_index, src=bias_mat)
                        last_p = self.logsoftmax(last_headout)
                        candidate_scores = last_p.gather(dim=-1, index=preselect_index)
                        reselect = torch.topk(candidate_scores, top_k, dim=-1)
                        topk_index = preselect_index.gather(dim=-1, index=reselect.indices)
                        topk_p = reselect.values

                    local_scores_list.append(topk_p)
                    token_list.append(topk_index)
                except Exception as e:
                    # 回退：预选集合直接重选
                    reselect = torch.topk(preselect_p, top_k, dim=-1)
                    topk_index = preselect_index.gather(dim=-1, index=reselect.indices)
                    topk_p = last_p.gather(dim=-1, index=topk_index) if SORT_ONLY_BIAS else reselect.values
                    local_scores_list.append(topk_p)
                    token_list.append(topk_index)
            else:
                # 未收集到校准特征时，预选集合直接重选
                reselect = torch.topk(preselect_p, top_k, dim=-1)
                topk_index = preselect_index.gather(dim=-1, index=reselect.indices)
                topk_p = last_p.gather(dim=-1, index=topk_index)
                local_scores_list.append(topk_p)
                token_list.append(topk_index)

        # 后续累计分数和树结构逻辑保持不变
        cu_scores = topk_p + scores[:, None]
        topk_cs = torch.topk(cu_scores.view(-1), top_k, dim=-1)
        topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
        scores = topk_cs_p
        // ... existing code ...