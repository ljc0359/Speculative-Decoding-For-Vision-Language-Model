# =========================================================
# Uncertainty + scoring helpers for speculative VLM trees
# =========================================================
import torch
import torch.nn.functional as F
import math
from typing import Optional, Tuple

# ---------- 基础工具 ----------
def to_probs(
    logits: Optional[torch.Tensor] = None,
    logprobs: Optional[torch.Tensor] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    把 logits 或 logprobs 变成概率，自动截断到 [eps, 1] 保证数值稳定。
    形状：[..., V]
    """
    assert (logits is None) ^ (logprobs is None), "pass exactly one of (logits, logprobs)"
    if logits is not None:
        p = torch.softmax(logits, dim=-1)
    else:
        p = (logprobs).exp()
    return p.clamp_min(eps)


def entropy_nd(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    香农熵 H(p)；对最后一维求和，返回形状为 probs[..., :-1] 的张量。
    """
    p = probs.clamp_min(eps)
    return -(p * p.log()).sum(dim=-1)


def kl_divergence_nd(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    KL(p || q)；对最后一维求和，支持任意前置维度广播。
    """
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return (p * (p.log() - q.log())).sum(dim=-1)


def js_divergence_nd(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    JS(p || q) = 0.5 * KL(p || m) + 0.5 * KL(q || m), m = 0.5*(p+q)
    """
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence_nd(p, m, eps) + 0.5 * kl_divergence_nd(q, m, eps)


# ---------- 方案 1: 便宜代理（单次前向即可） ----------
def quick_uncertainty_proxies(
    p_hat: torch.Tensor,
    p: Optional[torch.Tensor] = None,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    输入：
      - p_hat: 草稿分布 p̂，形状[..., V]
      - p:     目标分布 p（可选，当轮没有就传 None）
    输出（逐节点标量，形状为 p_hat[..., :-1]）：
      - H_hat: H(p̂)                      （草稿自身熵）
      - alea_proxy: ½(H(p) + H(p̂)) 或 H(p̂)（若缺 p）
      - epi_proxy:  JS(p || p̂) 或 0      （若缺 p）
    """
    H_hat = entropy_nd(p_hat, eps)
    if p is None:
        return H_hat, H_hat, torch.zeros_like(H_hat)
    H_t = entropy_nd(p, eps)
    alea_proxy = 0.5 * (H_t + H_hat)
    epi_proxy = js_divergence_nd(p, p_hat, eps)
    return H_hat, alea_proxy, epi_proxy


# ---------- 方案 1+: 多样本估计（更稳） ----------
def alea_epi_from_mc_probs(
    probs_mc: torch.Tensor,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Monte Carlo / 集成 / TTA 的不确定性分解（分类/生成的离散分布）。
    输入：
      - probs_mc: 形状 [K, ..., V]，K 是样本数（模型或增强）
    输出（去掉 V 维与 K 维，保留其余维度）：
      - H_total = H(平均分布)
      - H_alea  = 平均的 H(p^{(k)})
      - H_epi   = H_total - H_alea
    """
    # [K, ..., V] -> [K, ...]
    H_each = entropy_nd(probs_mc.clamp_min(eps), eps)            # 按 V 求和，保留 K
    p_bar = probs_mc.mean(dim=0)                                  # [..., V]
    H_total = entropy_nd(p_bar, eps)                              # [...]
    H_alea = H_each.mean(dim=0)                                   # [...]
    H_epi = H_total - H_alea                                      # [...]
    return H_total, H_alea, H_epi


# ---------- 节点/路径打分（扩展 + 重排会用到） ----------
def node_scores(
    p_hat: torch.Tensor,
    candidate_ids: torch.Tensor,
    p: Optional[torch.Tensor] = None,
    a: float = 4.0,
    b: float = 1.0,
    c: float = 1.0,
    d: float = 0.0,
    use_js: bool = True,
    use_epi: bool = False,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    给一组候选 token（例如 top-k）打分，用于“扩展 + 重排”。
    公式：score = sigmoid(a * conf - b * H_hat - c * JS - d * H_epi)
    - conf = p̂(候选token)
    - H_hat = H(p̂)
    - JS/H_epi 需要目标分布 p；若当轮拿不到 p，可先把 use_js/use_epi=False（只用草稿侧信号）

    输入：
      - p_hat:         [..., V]
      - candidate_ids: [..., K]  （每个节点的 K 个候选 id）
      - p:             [..., V]（可选）
      - a,b,c,d:       权重（可调，建议离线标定）
    输出：
      - scores:        [..., K]  （与 candidate_ids 对齐）
    """
    # 置信度：按最后一维 gather
    conf = p_hat.gather(dim=-1, index=candidate_ids).clamp_min(eps)   # [..., K]
    H_hat = entropy_nd(p_hat, eps)[..., None]                         # [..., 1] 方便广播

    js = torch.zeros_like(conf)
    H_epi = torch.zeros_like(conf)

    if (p is not None) and use_js:
        js = js_divergence_nd(p, p_hat, eps)[..., None].expand_as(conf)

    if (p is not None) and use_epi:
        # 用 “平均熵差” 近似：H_total - ½(H(p)+H(p̂))
        H_total = entropy_nd(0.5 * (p + p_hat), eps)[..., None]
        H_mix = 0.5 * (entropy_nd(p, eps) + entropy_nd(p_hat, eps))[..., None]
        H_epi = (H_total - H_mix).expand_as(conf)

    z = a * conf - b * H_hat - c * js - d * H_epi                   # 线性组合
    scores = torch.sigmoid(z)                                       # [..., K]
    return scores


# ---------- 路径价值（可乘积或log-加和） ----------
def path_values(scores_seq: torch.Tensor, use_log: bool = True, eps: float = 1e-12) -> torch.Tensor:
    """
    把一条路径上逐步分数 {c_t} 汇总成 V_path。输入形状 [T] 或 [*, T]。
    - use_log=True 时返回 sum(log c_t)（更稳，排序等价）；否则返回 ∏ c_t
    """
    if use_log:
        return (scores_seq.clamp_min(eps)).log().sum(dim=-1)
    else:
        return scores_seq.clamp_min(eps).prod(dim=-1)


# ---------- 基于 MC 的稳健代理（对 logits 进行轻量扰动） ----------
def _sample_probs_from_logits(
    logits: torch.Tensor,
    num_samples: int = 8,
    noise_std: float = 0.3,
    temperature: float = 1.0,
    kind: str = "gauss",
) -> torch.Tensor:
    """
    给定 logits，构造 K 份带扰动的概率分布，返回形状 [K, ..., V]
    - kind = 'gauss': logits + N(0, noise_std)
    - kind = 'gumbel': logits + Gumbel(0, noise_std)
    """
    assert num_samples >= 1
    
    # Ensure float32 for numerical stability
    original_dtype = logits.dtype
    if logits.dtype == torch.float16:
        logits = logits.float()
    
    shape = (num_samples,) + logits.shape
    if kind == "gauss":
        noise = torch.randn(shape, device=logits.device, dtype=logits.dtype) * noise_std
    elif kind == "gumbel":
        # Gumbel(0, beta) 近似：-log(-log(U)), 再乘以 noise_std 作为尺度
        u = torch.rand(shape, device=logits.device, dtype=logits.dtype).clamp_min(1e-6)
        noise = -torch.log(-torch.log(u)).mul(noise_std)
    else:
        raise ValueError(f"Unknown kind '{kind}' for MC noise")

    perturbed = (logits.unsqueeze(0) + noise) / max(1e-6, float(temperature))
    probs_mc = torch.softmax(perturbed, dim=-1)
    
    # Convert back to original dtype if needed
    if original_dtype == torch.float16:
        probs_mc = probs_mc.half()
    
    return probs_mc


def _normalize_entropy(ent: torch.Tensor, vocab_size: int, eps: float = 1e-12) -> torch.Tensor:
    """
    把熵规范化到 [0,1] 量级：H / log(V)
    """
    denom = math.log(max(vocab_size, 2))
    return (ent / max(denom, eps))


def _normalize_js(js: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    把 JS 规范化到 [0,1]：JS / log(2)
    """
    return js / math.log(2.0)


def mc_node_scores_from_logits(
    logits: torch.Tensor,
    candidate_ids: torch.Tensor,
    num_samples: int = 8,
    noise_std: float = 0.3,
    temperature: float = 1.0,
    kind: str = "gauss",
    a: float = 4.0,
    b: float = 1.0,
    c: float = 1.0,
    d: float = 0.0,
    use_js: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    使用 MC 方式从 logits 生成 K 份分布，计算 p̄、H_total、H_alea、H_epi，并据此为候选 token 打分。

    输入：
      - logits:        [..., V]
      - candidate_ids: [..., K2]
    返回：
      - scores:        [..., K2]
    公式：
      conf = p̄(token)
      H_hat = H(p̄)            （规范化）
      JS    = JS(p̄ || p̂)     （规范化，可选）
      H_epi = H_total - H_alea （规范化）
      score = sigmoid(a*conf - b*H_hat - c*JS - d*H_epi)
    """
    # 基线分布 p̂
    p_hat = torch.softmax(logits / max(1e-6, float(temperature)), dim=-1)

    # MC 采样得到 [S, ..., V]
    probs_mc = _sample_probs_from_logits(
        logits=logits,
        num_samples=int(num_samples),
        noise_std=float(noise_std),
        temperature=float(temperature),
        kind=kind,
    )

    # p̄ 与不确定性分解
    p_bar = probs_mc.mean(dim=0)  # [..., V]
    H_total, H_alea, H_epi = alea_epi_from_mc_probs(probs_mc, eps)

    vocab_size = logits.shape[-1]
    H_hat = entropy_nd(p_bar, eps)
    H_hat = _normalize_entropy(H_hat, vocab_size, eps)[..., None]  # [..., 1]
    H_epi = _normalize_entropy(H_epi, vocab_size, eps)[..., None]

    js = torch.zeros_like(H_hat)
    if use_js:
        js_val = js_divergence_nd(p_bar, p_hat, eps)
        js = _normalize_js(js_val, eps)[..., None]

    # 置信度取 p̄(token)
    conf = p_bar.gather(dim=-1, index=candidate_ids).clamp_min(eps)

    # 广播拼装
    js = js.expand_as(conf)
    H_hat = H_hat.expand_as(conf)
    H_epi = H_epi.expand_as(conf)

    z = a * conf - b * H_hat - c * js - d * H_epi
    scores = torch.sigmoid(z)
    return scores


def mc_stats_from_logits(
    logits: torch.Tensor,
    num_samples: int = 8,
    noise_std: float = 0.3,
    temperature: float = 1.0,
    kind: str = "gauss",
    use_js: bool = True,
    eps: float = 1e-12,
):
    """
    返回基于 MC 的统计量：
      - p_bar:        [..., V]
      - H_bar_n:      [...]   （H(p_bar)/log(V)）
      - H_epi_n:      [...]   （(H_total-H_alea)/log(V)）
      - H_alea_n:     [...]   （H_alea/log(V)）
      - js_n:         [...]   （JS(p_bar||p_hat)/log(2)；若 use_js=False 则为 0）
    """
    # Convert to float32 to avoid half precision issues with softmax
    original_dtype = logits.dtype
    logits_f32 = logits.float()
    p_hat = torch.softmax(logits_f32 / max(1e-6, float(temperature)), dim=-1)
    probs_mc = _sample_probs_from_logits(
        logits=logits_f32,  # Use float32 version
        num_samples=int(num_samples),
        noise_std=float(noise_std),
        temperature=float(temperature),
        kind=kind,
    )
    p_bar = probs_mc.mean(dim=0)
    H_total, H_alea, H_epi = alea_epi_from_mc_probs(probs_mc, eps)
    vocab_size = logits.shape[-1]
    H_bar_n = _normalize_entropy(H_total, vocab_size, eps)
    H_epi_n = _normalize_entropy(H_epi, vocab_size, eps)
    H_alea_n = _normalize_entropy(H_alea, vocab_size, eps)
    if use_js:
        js_val = js_divergence_nd(p_bar, p_hat, eps)
        js_n = _normalize_js(js_val, eps)
    else:
        js_n = torch.zeros_like(H_bar_n)
    
    # Convert results back to original dtype if needed
    if original_dtype == torch.float16:
        p_bar = p_bar.half()
        H_bar_n = H_bar_n.half()
        H_epi_n = H_epi_n.half()
        H_alea_n = H_alea_n.half()
        js_n = js_n.half()
    
    return p_bar, H_bar_n, H_epi_n, H_alea_n, js_n


def dual_uncertainty_alignment(
    topk_logp: torch.Tensor,
    H_epi_n: torch.Tensor,
    H_alea_n: torch.Tensor,
    epi_threshold: float = 5.0,
    alea_threshold: float = 5.0,
    epi_center: float = 0.5,
    alea_center: float = 0.5,
    exploit_bonus: float = 3.0,
    explore_penalty: float = -0.3,
    balance_factor: float = 0.5,
    uncertain_penalty: float = -0.8,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    基于双重不确定性的对齐奖励策略（修复数值问题版本）：
    
    改进：
    1. 使用更陡峭的阈值 (5.0) 增强区分度
    2. 调整 center 到 0.5 使分界更明确  
    3. 减少负奖励强度，增加正奖励
    4. 简化奖励逻辑，直接基于 log 概率排序
    
    输入：
      - topk_logp: [..., K] 候选token的log概率
      - H_epi_n: [...] 规范化的epistemic uncertainty [0,1]
      - H_alea_n: [...] 规范化的aleatoric uncertainty [0,1]
    
    返回：
      - p_align: [..., K] 对齐奖励
    """
    # 使用更陡峭的阈值，增强区分度
    epi_confidence = torch.sigmoid(-epi_threshold * (H_epi_n - epi_center))
    alea_sharpness = torch.sigmoid(-alea_threshold * (H_alea_n - alea_center))
    
    # 四种模式的权重
    exploit_mode = epi_confidence * alea_sharpness                    # 双低不确定性：exploit
    explore_mode = (1 - epi_confidence)                             # 高认知不确定性：explore  
    balance_mode = epi_confidence * (1 - alea_sharpness)            # 低认知高随机：balance
    uncertain_mode = (1 - epi_confidence) * (1 - alea_sharpness)    # 双高不确定性：很保守
    
    # 简化的奖励策略：直接基于相对排序
    # 将 topk_logp 排序，给高概率候选更高奖励
    _, indices = torch.sort(topk_logp, dim=-1, descending=True)
    ranks = torch.zeros_like(topk_logp)
    K = topk_logp.shape[-1]
    for i in range(K):
        ranks.scatter_(-1, indices[..., i:i+1], K-i)  # rank K (best) to 1 (worst)
    normalized_ranks = (ranks - 1) / (K - 1)  # [0, 1], 1 for best
    
    # 不同模式的奖励策略  
    # exploit: 强烈奖励高排序候选
    exploit_reward = exploit_bonus * normalized_ranks
    
    # explore: 轻微惩罚，但保持相对排序
    explore_reward = explore_penalty + 0.5 * normalized_ranks
    
    # balance: 中等奖励，平衡排序和保守性
    balance_reward = balance_factor * normalized_ranks
    
    # uncertain: 保守惩罚，但仍保持一定排序
    uncertain_reward = uncertain_penalty + 0.2 * normalized_ranks
    
    # 加权组合
    p_align = (
        exploit_mode[..., None] * exploit_reward +
        explore_mode[..., None] * explore_reward +
        balance_mode[..., None] * balance_reward +
        uncertain_mode[..., None] * uncertain_reward
    )
    
    return p_align
