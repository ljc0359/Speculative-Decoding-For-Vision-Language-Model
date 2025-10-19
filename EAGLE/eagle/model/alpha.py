def _compute_adaptive_alpha(data_list, base_alpha: float, layer_idx: int, shape=None, device=None, dtype=None):
            """
            data_list: list[dict] from _collect_calibration_data_safely
            返回与候选同形状的 alpha 向量（或矩阵，对 layer_i 是 top_k*top_k）
            规则（可按需调优）：
            - margin 越小 -> 越依赖校准（alpha↑）
            - depth 越深  -> 越依赖校准（alpha↑）
            - 注意力越弱  -> 越依赖校准（alpha↑）
            - token_category == 'number' 小幅增加（经验启发）
            """
            if (shape is None) or (device is None) or (dtype is None):
                raise RuntimeError("adaptive alpha: shape/device/dtype must be provided")

            # 默认：全部用 base_alpha（保证健壮）
            if not data_list:
                return torch.full(shape, base_alpha, device=device, dtype=dtype)

            import numpy as np
            # 提取字段（缺失则回退）
            draft_margin = np.array([d.get('draft_margin', np.nan) for d in data_list], dtype=np.float64)
            depth_arr    = np.array([d.get('tree_depth', d.get('layer', np.nan)) for d in data_list], dtype=np.float64)
            attn_arr     = np.array([d.get('avg_visual_attention_intensity', np.nan) for d in data_list], dtype=np.float64)
            token_cat    = [d.get('token_category', None) for d in data_list]

            n = len(data_list)

            # margin 因子：低 margin -> 高权重（线性缩放 + 分位裁切）
            # 若 margin 缺失，用全体的中位数兜底；再缺失用 0.0
            if np.isnan(draft_margin).all():
                draft_margin[:] = 0.0
            else:
                med = np.nanmedian(draft_margin)
                draft_margin = np.where(np.isnan(draft_margin), med, draft_margin)
            # 用分位做归一化，避免极端值主导
            m_lo, m_hi = np.nanpercentile(draft_margin, 10), np.nanpercentile(draft_margin, 90)
            if m_hi <= m_lo:
                m_lo, m_hi = float(np.min(draft_margin)), float(np.max(draft_margin) + 1e-8)
            margin_norm = np.clip((draft_margin - m_lo) / (m_hi - m_lo + 1e-8), 0.0, 1.0)
            # margin 越小越不确定 => alpha 越大
            margin_factor = 1.0 - margin_norm  # [0,1]

            # depth 因子：深度越深越不确定
            if np.isnan(depth_arr).all():
                depth_arr[:] = 1.0
            else:
                d_med = np.nanmedian(depth_arr)
                depth_arr = np.where(np.isnan(depth_arr), d_med, depth_arr)
            # 假定 1~6 合理范围，>6 截断；可按你的树深度分布调整
            depth_factor = np.clip(depth_arr / 6.0, 0.0, 1.0)

            # 注意力因子：注意力越弱（小）越不确定
            if np.isnan(attn_arr).all():
                attn_arr[:] = 0.5
            else:
                a_med = np.nanmedian(attn_arr)
                attn_arr = np.where(np.isnan(attn_arr), a_med, attn_arr)
            a_lo, a_hi = np.nanpercentile(attn_arr, 10), np.nanpercentile(attn_arr, 90)
            if a_hi <= a_lo:
                a_lo, a_hi = float(np.min(attn_arr)), float(np.max(attn_arr) + 1e-8)
            attn_norm = np.clip((attn_arr - a_lo) / (a_hi - a_lo + 1e-8), 0.0, 1.0)
            attn_factor = 1.0 - attn_norm  # 低注意力 -> 因子大

            # token 类别微调
            tok_boost = np.ones(n, dtype=np.float64)
            for i, cat in enumerate(token_cat):
                if isinstance(cat, str) and cat.lower() == 'number':
                    tok_boost[i] = 1.10  # number 稍微更信任校准
                else:
                    tok_boost[i] = 1.00

            # 组合：加权平均再乘微调；避免过激，最后整体缩放到 [0.2, 1.0]
            # 可调系数：margin 0.5, depth 0.3, attn 0.2
            combo = (0.5 * margin_factor + 0.3 * depth_factor + 0.2 * attn_factor)
            combo = np.clip(combo * tok_boost, 0.0, 1.2)
            combo = np.clip(combo, 0.2, 1.0)  # 保底 0.2，避免 alpha 过小无效

            alpha_vec = base_alpha * combo  # 限制不超过全局 alpha
            alpha_t = torch.tensor(alpha_vec, device=device, dtype=dtype)
            if len(shape) == 2:
                # layer_i: 需要 reshape 成 [top_k, top_k]
                try:
                    alpha_t = alpha_t.view(*shape)
                except Exception:
                    alpha_t = alpha_t[: (shape[0] * shape[1])].view(*shape)
            return alpha_t