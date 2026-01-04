"""
MiniMind 模型实现

本文件实现了 MiniMind 语言模型的核心架构，包括：
1. MiniMindConfig - 模型配置类，定义所有超参数
2. RMSNorm - Root Mean Square Layer Normalization
3. Attention - 多头注意力机制（支持 GQA 和 Flash Attention）
4. FeedForward - 前馈神经网络（SwiGLU 激活）
5. MoEGate - 混合专家门控机制
6. MOEFeedForward - 混合专家前馈网络
7. MiniMindBlock - Transformer 解码器层
8. MiniMindModel - 完整的 Transformer 模型
9. MiniMindForCausalLM - 因果语言模型（用于文本生成）

模型架构特点：
- 采用 Pre-Norm 结构（LayerNorm 在注意力和 FFN 之前）
- 使用 RoPE（旋转位置编码）进行位置编码
- 支持 GQA（分组查询注意力）以减少 KV 缓存
- 支持 MoE（混合专家）架构以增加模型容量
- 支持 Flash Attention 加速推理
- 支持 YaRN 位置编码外推以处理长序列
"""

# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Config
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

from transformers import PretrainedConfig


class MiniMindConfig(PretrainedConfig):
    """
    MiniMind 模型配置类
    
    继承自 HuggingFace 的 PretrainedConfig，用于定义模型的所有超参数。
    这些参数控制模型的架构、大小和行为。
    
    Attributes:
        model_type (str): 模型类型标识符，用于 HuggingFace 自动加载
        
    基础架构参数:
        dropout (float): Dropout 概率，用于正则化防止过拟合
        bos_token_id (int): 序列开始标记的 ID
        eos_token_id (int): 序列结束标记的 ID
        hidden_act (str): 隐藏层激活函数类型（默认 'silu'，即 SwiGLU）
        hidden_size (int): 隐藏层维度，决定模型的宽度
        intermediate_size (int): FFN 中间层维度，通常为 hidden_size 的 8/3 倍
        max_position_embeddings (int): 最大位置编码长度
        num_attention_heads (int): 注意力头数量
        num_hidden_layers (int): Transformer 层数，决定模型的深度
        num_key_value_heads (int): KV 头数量（用于 GQA，减少 KV 缓存）
        vocab_size (int): 词表大小
        rms_norm_eps (float): RMSNorm 的 epsilon 值，防止除零
        rope_theta (float): RoPE 位置编码的基础频率
        inference_rope_scaling (bool): 是否启用推理时的 RoPE 缩放（YaRN）
        flash_attn (bool): 是否使用 Flash Attention 加速
        
    MoE（混合专家）参数:
        use_moe (bool): 是否使用 MoE 架构
        num_experts_per_tok (int): 每个 token 激活的专家数量
        n_routed_experts (int): 可路由专家的总数量
        n_shared_experts (int): 共享专家数量（所有 token 都会使用）
        scoring_func (str): 专家选择的评分函数
        aux_loss_alpha (float): 辅助损失的权重系数
        seq_aux (bool): 是否在序列级别计算辅助损失
        norm_topk_prob (bool): 是否归一化 top-k 概率
    """
    model_type = "minimind"  # HuggingFace 模型类型标识

    def __init__(
            self,
            dropout: float = 0.0,                    # Dropout 概率
            bos_token_id: int = 1,                   # 序列开始标记 ID
            eos_token_id: int = 2,                   # 序列结束标记 ID
            hidden_act: str = 'silu',                # 激活函数类型
            hidden_size: int = 512,                  # 隐藏层维度
            intermediate_size: int = None,          # FFN 中间层维度（自动计算）
            max_position_embeddings: int = 32768,   # 最大位置编码长度
            num_attention_heads: int = 8,            # 注意力头数
            num_hidden_layers: int = 8,              # Transformer 层数
            num_key_value_heads: int = 2,            # KV 头数（GQA）
            vocab_size: int = 6400,                  # 词表大小
            rms_norm_eps: float = 1e-05,            # RMSNorm epsilon
            rope_theta: int = 1000000.0,            # RoPE 基础频率
            inference_rope_scaling: bool = False,   # 是否启用 RoPE 缩放
            flash_attn: bool = True,                 # 是否使用 Flash Attention
            ####################################################
            # 以下是 MoE（混合专家）的特定配置
            # 当 use_moe 为 False 时，以下配置无效
            ####################################################
            use_moe: bool = False,                   # 是否使用 MoE
            num_experts_per_tok: int = 2,            # 每个 token 激活的专家数
            n_routed_experts: int = 4,               # 可路由专家总数
            n_shared_experts: int = 1,               # 共享专家数量
            scoring_func: str = 'softmax',           # 评分函数
            aux_loss_alpha: float = 0.1,             # 辅助损失权重
            seq_aux: bool = True,                    # 序列级辅助损失
            norm_topk_prob: bool = True,             # 归一化 top-k 概率
            **kwargs
    ):
        """
        初始化 MiniMind 配置
        
        Args:
            dropout: Dropout 概率，范围 [0, 1]
            bos_token_id: 序列开始标记的 token ID
            eos_token_id: 序列结束标记的 token ID
            hidden_act: 激活函数名称，支持 'silu', 'gelu', 'relu' 等
            hidden_size: 模型隐藏层维度，常见值：512, 768, 1024
            intermediate_size: FFN 中间层维度，None 时自动计算为 hidden_size * 8/3
            max_position_embeddings: 模型支持的最大序列长度
            num_attention_heads: 多头注意力的头数
            num_hidden_layers: Transformer 解码器层数
            num_key_value_heads: GQA 中的 KV 头数，小于 num_attention_heads 可减少内存
            vocab_size: 词表大小，需与 tokenizer 匹配
            rms_norm_eps: RMSNorm 中防止除零的小常数
            rope_theta: RoPE 位置编码的基础频率，影响位置编码的周期
            inference_rope_scaling: 启用 YaRN 位置编码外推，支持更长序列
            flash_attn: 使用 Flash Attention 2 加速，需要 PyTorch >= 2.0
            use_moe: 启用混合专家架构，增加模型容量但保持计算量
            num_experts_per_tok: 每个 token 路由到的专家数量
            n_routed_experts: 可路由专家的总数量
            n_shared_experts: 共享专家数量，所有 token 都会经过
            scoring_func: 专家选择的评分函数，目前支持 'softmax'
            aux_loss_alpha: 辅助损失权重，用于平衡专家负载
            seq_aux: 是否在序列级别计算辅助损失
            norm_topk_prob: 是否对选中专家的概率进行归一化
            **kwargs: 传递给父类的其他参数
        """
        super().__init__(**kwargs)
        
        # 基础架构参数
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        
        # YaRN 位置编码外推配置
        # 当启用 inference_rope_scaling 时，使用 YaRN 方法扩展位置编码
        # 外推长度 = factor * original_max_position_embeddings
        self.rope_scaling = {
            "beta_fast": 4,                          # 快速衰减因子
            "beta_slow": 1,                          # 慢速衰减因子
            "factor": 4,                             # 外推倍数
            "original_max_position_embeddings": 2048, # 原始最大位置
            "type": "yarn"                           # 外推类型
        } if self.inference_rope_scaling else None
        
        self.flash_attn = flash_attn
        
        ####################################################
        # MoE（混合专家）配置
        # 当 use_moe 为 False 时，以下配置无效
        ####################################################
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个 token 选择的专家数量
        self.n_routed_experts = n_routed_experts        # 总的专家数量
        self.n_shared_experts = n_shared_experts        # 共享专家
        self.scoring_func = scoring_func                # 评分函数，默认为 'softmax'
        self.aux_loss_alpha = aux_loss_alpha            # 辅助损失的 alpha 参数
        self.seq_aux = seq_aux                          # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob            # 是否标准化 top-k 概率


# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Model
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

import math
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


class RMSNorm(torch.nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm)
    
    RMSNorm 是 LayerNorm 的简化版本，只进行缩放而不进行中心化。
    相比 LayerNorm，RMSNorm 计算更高效，且在实践中效果相当。
    
    公式: output = x / sqrt(mean(x^2) + eps) * weight
    
    与 LayerNorm 的区别:
    - LayerNorm: (x - mean) / sqrt(var + eps) * gamma + beta
    - RMSNorm: x / sqrt(mean(x^2) + eps) * weight
    
    优点:
    1. 计算更简单，不需要计算均值
    2. 参数更少，只有 weight 没有 bias
    3. 在 LLM 中效果与 LayerNorm 相当
    
    Attributes:
        eps (float): 防止除零的小常数
        weight (nn.Parameter): 可学习的缩放参数
    """
    
    def __init__(self, dim: int, eps: float = 1e-5):
        """
        初始化 RMSNorm
        
        Args:
            dim: 输入特征维度
            eps: 防止除零的 epsilon 值
        """
        super().__init__()
        self.eps = eps
        # 可学习的缩放参数，初始化为全 1
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        计算 RMS 归一化
        
        Args:
            x: 输入张量，形状为 (..., dim)
            
        Returns:
            归一化后的张量，形状与输入相同
        """
        # 计算 x^2 的均值，然后取倒数平方根
        # rsqrt = 1 / sqrt(x)，比 1 / sqrt(x) 更高效
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, dim)
            
        Returns:
            归一化后的张量，形状与输入相同
        """
        # 先转为 float32 进行归一化计算（数值稳定性）
        # 然后转回原始数据类型，最后乘以可学习的权重
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6,
                         rope_scaling: Optional[dict] = None):
    """
    预计算 RoPE（旋转位置编码）的频率
    
    RoPE 是一种相对位置编码方法，通过旋转向量来编码位置信息。
    它的优点是可以自然地处理相对位置，且支持外推到更长的序列。
    
    RoPE 公式:
    - 对于位置 m 和维度 i，旋转角度 θ_i = m * base^(-2i/d)
    - 应用旋转: (q_2i, q_2i+1) -> (q_2i*cos(θ) - q_2i+1*sin(θ), q_2i*sin(θ) + q_2i+1*cos(θ))
    
    YaRN 外推:
    当 rope_scaling 不为 None 时，使用 YaRN 方法进行位置编码外推。
    YaRN 通过调整不同频率分量的缩放因子，实现更好的长度外推。
    
    Args:
        dim: 每个注意力头的维度
        end: 预计算的最大位置数
        rope_base: RoPE 的基础频率（theta）
        rope_scaling: YaRN 外推配置，包含 factor, beta_fast, beta_slow 等参数
        
    Returns:
        Tuple[Tensor, Tensor]: (freqs_cos, freqs_sin)
            - freqs_cos: 余弦频率，形状为 (end, dim)
            - freqs_sin: 正弦频率，形状为 (end, dim)
    """
    # 计算基础频率: 1 / (base^(2i/d))，其中 i = 0, 1, ..., d/2-1
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    
    # 如果启用 YaRN 外推
    if rope_scaling is not None:
        # 获取 YaRN 参数
        orig_max, factor, beta_fast, beta_slow = (
            rope_scaling.get("original_max_position_embeddings", 2048),  # 原始最大位置
            rope_scaling.get("factor", 4),                               # 外推倍数
            rope_scaling.get("beta_fast", 4.0),                          # 快速衰减因子
            rope_scaling.get("beta_slow", 1.0)                           # 慢速衰减因子
        )
        
        # 只有当目标长度超过原始最大长度时才进行外推
        if end / orig_max > 1.0:
            # 找到需要调整的维度边界
            # 对于周期大于原始最大位置的频率分量，需要进行缩放
            corr_dim = next((i for i in range(dim // 2) if 2 * math.pi / freqs[i] > orig_max), dim // 2)
            
            # 计算每个维度的插值权重
            power = torch.arange(0, dim // 2, device=freqs.device).float() / max(dim // 2 - 1, 1)
            beta = beta_slow + (beta_fast - beta_slow) * power
            
            # YaRN 标准公式: λ = (β·α - β + 1) / (β·α)
            # 其中 α = factor，β 是插值权重
            scale = torch.where(
                torch.arange(dim // 2, device=freqs.device) < corr_dim,
                (beta * factor - beta + 1) / (beta * factor),  # 需要缩放的维度
                1.0 / factor                                    # 不需要缩放的维度
            )
            freqs = freqs * scale

    # 生成位置序列 [0, 1, 2, ..., end-1]
    t = torch.arange(end, device=freqs.device)
    
    # 计算位置和频率的外积: (end,) x (dim/2,) -> (end, dim/2)
    freqs = torch.outer(t, freqs).float()
    
    # 计算 cos 和 sin，并复制以匹配完整维度
    # 形状: (end, dim/2) -> (end, dim)
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    应用旋转位置编码到 Query 和 Key
    
    RoPE 通过将向量分成两半，然后应用旋转变换来编码位置信息。
    旋转变换保持向量的模长不变，只改变方向。
    
    旋转公式:
    对于向量 x = [x_0, x_1, ..., x_{d-1}]，分成两半:
    - 前半部分: [x_0, x_1, ..., x_{d/2-1}]
    - 后半部分: [x_{d/2}, x_{d/2+1}, ..., x_{d-1}]
    
    旋转后:
    - 新前半部分 = 前半部分 * cos - 后半部分 * sin
    - 新后半部分 = 后半部分 * cos + 前半部分 * sin
    
    Args:
        q: Query 张量，形状为 (batch, seq_len, num_heads, head_dim)
        k: Key 张量，形状为 (batch, seq_len, num_kv_heads, head_dim)
        cos: 余弦频率，形状为 (seq_len, head_dim)
        sin: 正弦频率，形状为 (seq_len, head_dim)
        position_ids: 位置 ID（可选，当前未使用）
        unsqueeze_dim: 在哪个维度添加维度以进行广播
        
    Returns:
        Tuple[Tensor, Tensor]: 应用 RoPE 后的 (q, k)
    """
    def rotate_half(x):
        """
        将向量的前后两半交换并取反前半部分
        
        这是 RoPE 旋转的关键操作:
        [x_0, x_1, ..., x_{d/2-1}, x_{d/2}, ..., x_{d-1}]
        -> [-x_{d/2}, ..., -x_{d-1}, x_0, ..., x_{d/2-1}]
        
        Args:
            x: 输入张量，最后一维是 head_dim
            
        Returns:
            旋转后的张量
        """
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    # 应用旋转: x_rotated = x * cos + rotate_half(x) * sin
    # 这等价于复数乘法: (a + bi) * (cos + i*sin) = (a*cos - b*sin) + i*(a*sin + b*cos)
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    重复 Key/Value 头以匹配 Query 头的数量（用于 GQA）
    
    在分组查询注意力（GQA）中，KV 头的数量少于 Query 头。
    为了计算注意力，需要将 KV 头重复以匹配 Query 头的数量。
    
    例如: 如果有 8 个 Query 头和 2 个 KV 头，则每个 KV 头需要重复 4 次。
    
    这个函数等价于 torch.repeat_interleave(x, dim=2, repeats=n_rep)，
    但使用 expand + reshape 实现更高效。
    
    Args:
        x: KV 张量，形状为 (batch, seq_len, num_kv_heads, head_dim)
        n_rep: 重复次数，等于 num_heads // num_kv_heads
        
    Returns:
        重复后的张量，形状为 (batch, seq_len, num_heads, head_dim)
    """
    bs, slen, num_key_value_heads, head_dim = x.shape
    
    # 如果不需要重复，直接返回
    if n_rep == 1:
        return x
    
    # 使用 expand + reshape 实现重复
    # (bs, slen, num_kv_heads, head_dim) 
    # -> (bs, slen, num_kv_heads, 1, head_dim)
    # -> (bs, slen, num_kv_heads, n_rep, head_dim)
    # -> (bs, slen, num_kv_heads * n_rep, head_dim)
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """
    多头注意力机制
    
    实现了标准的多头自注意力，支持以下特性:
    1. 分组查询注意力（GQA）: KV 头数可以少于 Query 头数，减少 KV 缓存
    2. Flash Attention: 使用 PyTorch 2.0+ 的 scaled_dot_product_attention 加速
    3. KV 缓存: 支持增量解码，缓存历史 KV 以加速生成
    4. 因果掩码: 自动应用因果掩码，确保只能看到之前的 token
    
    注意力计算公式:
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    
    其中:
    - Q: Query，形状为 (batch, num_heads, seq_len, head_dim)
    - K: Key，形状为 (batch, num_heads, seq_len, head_dim)
    - V: Value，形状为 (batch, num_heads, seq_len, head_dim)
    - d_k: head_dim，用于缩放防止梯度消失
    
    Attributes:
        num_key_value_heads: KV 头数量
        n_local_heads: Query 头数量
        n_local_kv_heads: 本地 KV 头数量
        n_rep: KV 头重复次数
        head_dim: 每个头的维度
        q_proj, k_proj, v_proj: QKV 投影层
        o_proj: 输出投影层
        attn_dropout, resid_dropout: Dropout 层
        flash: 是否使用 Flash Attention
    """
    
    def __init__(self, args: MiniMindConfig):
        """
        初始化注意力层
        
        Args:
            args: 模型配置对象
        """
        super().__init__()
        
        # 设置 KV 头数量（GQA 配置）
        # 如果未指定 num_key_value_heads，则使用与 num_attention_heads 相同的值（标准 MHA）
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        
        # 确保 Query 头数是 KV 头数的整数倍
        assert args.num_attention_heads % self.num_key_value_heads == 0
        
        self.n_local_heads = args.num_attention_heads      # Query 头数
        self.n_local_kv_heads = self.num_key_value_heads   # KV 头数
        self.n_rep = self.n_local_heads // self.n_local_kv_heads  # KV 重复次数
        self.head_dim = args.hidden_size // args.num_attention_heads  # 每个头的维度
        
        # QKV 投影层（无偏置，遵循 LLaMA 设计）
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        
        # 输出投影层
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
        
        # Dropout 层
        self.attn_dropout = nn.Dropout(args.dropout)  # 注意力权重 Dropout
        self.resid_dropout = nn.Dropout(args.dropout)  # 残差连接 Dropout
        self.dropout = args.dropout
        
        # 检查是否可以使用 Flash Attention
        # 需要 PyTorch >= 2.0 且配置中启用
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn

    def forward(self,
                x: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False,
                attention_mask: Optional[torch.Tensor] = None):
        """
        注意力层前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, hidden_size)
            position_embeddings: RoPE 位置编码 (cos, sin)
            past_key_value: 缓存的 KV，用于增量解码
            use_cache: 是否返回更新后的 KV 缓存
            attention_mask: 注意力掩码，1 表示有效位置，0 表示需要掩码
            
        Returns:
            Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
                - output: 注意力输出，形状为 (batch_size, seq_len, hidden_size)
                - past_kv: 更新后的 KV 缓存（如果 use_cache=True）
        """
        bsz, seq_len, _ = x.shape
        
        # 计算 Q, K, V 投影
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        
        # 重塑为多头格式: (batch, seq_len, num_heads, head_dim)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # 应用 RoPE 位置编码
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])

        # KV 缓存处理（用于增量解码）
        if past_key_value is not None:
            # 将新的 KV 与缓存的 KV 拼接
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        
        # 如果需要缓存，保存当前 KV
        past_kv = (xk, xv) if use_cache else None

        # 转置为注意力计算格式: (batch, num_heads, seq_len, head_dim)
        # 同时对 KV 进行重复以匹配 Query 头数（GQA）
        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )

        # 选择注意力计算方式
        if self.flash and seq_len > 1 and (attention_mask is None or torch.all(attention_mask == 1)):
            # 使用 Flash Attention（更快，更省内存）
            # 条件: 启用 Flash Attention，序列长度 > 1，且没有自定义掩码
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True  # 自动应用因果掩码
            )
        else:
            # 标准注意力计算
            # 计算注意力分数: Q @ K^T / sqrt(d_k)
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            # 应用因果掩码（上三角矩阵设为 -inf）
            scores = scores + torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)

            # 应用自定义注意力掩码（如果提供）
            if attention_mask is not None:
                # 扩展掩码维度以匹配注意力分数
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                # 将 0 位置设为大负数，使 softmax 后接近 0
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            # Softmax 归一化（使用 float32 保证数值稳定性）
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            
            # 应用注意力 Dropout
            scores = self.attn_dropout(scores)
            
            # 计算加权和: Attention @ V
            output = scores @ xv

        # 重塑输出: (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, hidden_size)
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        
        # 输出投影和残差 Dropout
        output = self.resid_dropout(self.o_proj(output))
        
        return output, past_kv


class FeedForward(nn.Module):
    """
    前馈神经网络（FFN）
    
    使用 SwiGLU 激活函数的前馈网络，这是 LLaMA 等现代 LLM 的标准设计。
    
    SwiGLU 公式:
    FFN(x) = down_proj(act(gate_proj(x)) * up_proj(x))
    
    其中:
    - gate_proj: 门控投影，输出维度为 intermediate_size
    - up_proj: 上投影，输出维度为 intermediate_size
    - act: 激活函数（默认 SiLU/Swish）
    - down_proj: 下投影，输出维度为 hidden_size
    
    与标准 FFN 的区别:
    - 标准 FFN: down(act(up(x)))
    - SwiGLU: down(act(gate(x)) * up(x))
    
    SwiGLU 通过门控机制提供更好的梯度流动和表达能力。
    
    Attributes:
        gate_proj: 门控投影层
        down_proj: 下投影层
        up_proj: 上投影层
        dropout: Dropout 层
        act_fn: 激活函数
    """
    
    def __init__(self, config: MiniMindConfig):
        """
        初始化前馈网络
        
        Args:
            config: 模型配置对象
        """
        super().__init__()
        
        # 计算中间层维度
        # 如果未指定，使用 hidden_size * 8/3，并对齐到 64 的倍数
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        
        # 定义投影层（无偏置）
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        
        # Dropout 和激活函数
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]  # 从 HuggingFace 获取激活函数

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, hidden_size)
            
        Returns:
            输出张量，形状与输入相同
        """
        # SwiGLU: down(act(gate(x)) * up(x))
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


class MoEGate(nn.Module):
    """
    混合专家门控机制（MoE Gate）
    
    门控网络负责为每个 token 选择最合适的专家。
    它计算每个 token 对所有专家的亲和度分数，然后选择 top-k 个专家。
    
    工作流程:
    1. 将输入投影到专家空间: scores = x @ weight^T
    2. 应用评分函数（如 softmax）得到概率分布
    3. 选择 top-k 个专家
    4. 可选地归一化选中专家的概率
    5. 计算辅助损失以平衡专家负载
    
    辅助损失:
    为了防止所有 token 都路由到少数专家（负载不均衡），
    引入辅助损失来鼓励均匀的专家使用。
    
    Attributes:
        top_k: 每个 token 选择的专家数量
        n_routed_experts: 可路由专家总数
        scoring_func: 评分函数类型
        alpha: 辅助损失权重
        seq_aux: 是否使用序列级辅助损失
        norm_topk_prob: 是否归一化 top-k 概率
        weight: 门控权重矩阵
    """
    
    def __init__(self, config: MiniMindConfig):
        """
        初始化门控网络
        
        Args:
            config: 模型配置对象
        """
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok      # 每个 token 激活的专家数
        self.n_routed_experts = config.n_routed_experts  # 专家总数

        self.scoring_func = config.scoring_func      # 评分函数
        self.alpha = config.aux_loss_alpha           # 辅助损失权重
        self.seq_aux = config.seq_aux                # 序列级辅助损失

        self.norm_topk_prob = config.norm_topk_prob  # 归一化 top-k 概率
        self.gating_dim = config.hidden_size         # 门控输入维度
        
        # 门控权重矩阵: (n_experts, hidden_size)
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        初始化门控权重
        
        使用 Kaiming 均匀初始化，适合 ReLU 类激活函数
        """
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        """
        门控前向传播
        
        Args:
            hidden_states: 输入张量，形状为 (batch_size, seq_len, hidden_size)
            
        Returns:
            Tuple[Tensor, Tensor, float]:
                - topk_idx: 选中专家的索引，形状为 (batch_size * seq_len, top_k)
                - topk_weight: 选中专家的权重，形状为 (batch_size * seq_len, top_k)
                - aux_loss: 辅助损失值
        """
        bsz, seq_len, h = hidden_states.shape
        
        # 展平为 (batch_size * seq_len, hidden_size)
        hidden_states = hidden_states.view(-1, h)
        
        # 计算门控分数: (batch_size * seq_len, n_experts)
        logits = F.linear(hidden_states, self.weight, None)
        
        # 应用评分函数
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'不支持的评分函数: {self.scoring_func}')

        # 选择 top-k 个专家
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # 归一化 top-k 概率（可选）
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        # 计算辅助损失（仅在训练时）
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            
            if self.seq_aux:
                # 序列级辅助损失
                # 计算每个专家在每个序列中被选中的频率
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(
                    1, topk_idx_for_aux_loss,
                    torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)
                ).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                # Token 级辅助损失
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = 0
            
        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    """
    混合专家前馈网络（MoE FFN）
    
    MoE 通过使用多个专家网络来增加模型容量，同时保持计算量不变。
    每个 token 只激活 top-k 个专家，而不是所有专家。
    
    架构:
    1. 门控网络选择 top-k 个专家
    2. 将 token 路由到选中的专家
    3. 加权合并专家输出
    4. 可选地添加共享专家的输出
    
    优点:
    - 增加模型容量而不增加计算量
    - 不同专家可以专注于不同类型的输入
    - 支持更大规模的模型
    
    Attributes:
        experts: 专家网络列表
        gate: 门控网络
        shared_experts: 共享专家列表（可选）
        aux_loss: 辅助损失（用于训练）
    """
    
    def __init__(self, config: MiniMindConfig):
        """
        初始化 MoE 前馈网络
        
        Args:
            config: 模型配置对象
        """
        super().__init__()
        self.config = config
        
        # 创建专家网络列表
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        
        # 门控网络
        self.gate = MoEGate(config)
        
        # 共享专家（所有 token 都会经过）
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])

    def forward(self, x):
        """
        MoE 前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, hidden_size)
            
        Returns:
            输出张量，形状与输入相同
        """
        identity = x  # 保存原始输入用于共享专家
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        
        # 使用门控机制选择专家
        topk_idx, topk_weight, aux_loss = self.gate(x)
        
        # 展平输入
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        
        if self.training:
            # 训练模式: 为每个 token 复制 top_k 份，分别送入对应专家
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x, dtype=torch.float16)
            
            # 遍历每个专家，处理路由到该专家的 token
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)
            
            # 加权合并专家输出
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            # 推理模式: 使用更高效的实现
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        
        # 添加共享专家的输出
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        
        # 保存辅助损失供训练使用
        self.aux_loss = aux_loss
        
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        """
        MoE 推理优化实现
        
        在推理时，使用更高效的批处理方式:
        1. 按专家索引排序所有 token
        2. 批量处理每个专家的 token
        3. 使用 scatter_add 合并结果
        
        这比训练时的实现更高效，因为避免了重复复制。
        
        Args:
            x: 展平的输入，形状为 (batch_size * seq_len, hidden_size)
            flat_expert_indices: 专家索引，形状为 (batch_size * seq_len * top_k,)
            flat_expert_weights: 专家权重，形状为 (batch_size * seq_len * top_k, 1)
            
        Returns:
            输出张量，形状为 (batch_size * seq_len, hidden_size)
        """
        expert_cache = torch.zeros_like(x)
        
        # 按专家索引排序
        idxs = flat_expert_indices.argsort()
        
        # 计算每个专家处理的 token 数量的累积和
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        
        # 计算原始 token 索引
        token_idxs = idxs // self.config.num_experts_per_tok
        
        # 遍历每个专家
        # 例如: tokens_per_expert = [6, 15, 20, 26] 表示:
        # - 专家 0 处理 token_idxs[:6]
        # - 专家 1 处理 token_idxs[6:15]
        # - 专家 2 处理 token_idxs[15:20]
        # - 专家 3 处理 token_idxs[20:26]
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            
            # 跳过没有 token 的专家
            if start_idx == end_idx:
                continue
            
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            
            # 专家处理并加权
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            
            # 使用 scatter_add 累加到结果
            expert_cache.scatter_add_(
                0,
                exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]),
                expert_out
            )

        return expert_cache


class MiniMindBlock(nn.Module):
    """
    MiniMind Transformer 解码器层
    
    每个 Block 包含:
    1. 自注意力层（带残差连接和 Pre-Norm）
    2. 前馈网络（带残差连接和 Pre-Norm）
    
    Pre-Norm 结构:
    output = x + Attention(Norm(x))
    output = output + FFN(Norm(output))
    
    与 Post-Norm 的区别:
    - Pre-Norm: Norm 在子层之前，训练更稳定
    - Post-Norm: Norm 在子层之后，可能需要更小的学习率
    
    Attributes:
        self_attn: 自注意力层
        mlp: 前馈网络（普通 FFN 或 MoE FFN）
        input_layernorm: 注意力前的 LayerNorm
        post_attention_layernorm: FFN 前的 LayerNorm
    """
    
    def __init__(self, layer_id: int, config: MiniMindConfig):
        """
        初始化 Transformer 层
        
        Args:
            layer_id: 层索引
            config: 模型配置对象
        """
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        # 自注意力层
        self.self_attn = Attention(config)

        self.layer_id = layer_id
        
        # Pre-Norm: 在注意力和 FFN 之前应用 LayerNorm
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # 前馈网络: 根据配置选择普通 FFN 或 MoE FFN
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        """
        Transformer 层前向传播
        
        Args:
            hidden_states: 输入张量，形状为 (batch_size, seq_len, hidden_size)
            position_embeddings: RoPE 位置编码 (cos, sin)
            past_key_value: 缓存的 KV（用于增量解码）
            use_cache: 是否返回更新后的 KV 缓存
            attention_mask: 注意力掩码
            
        Returns:
            Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
                - hidden_states: 输出张量
                - present_key_value: 更新后的 KV 缓存
        """
        # 保存残差
        residual = hidden_states
        
        # 自注意力（Pre-Norm）
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask
        )
        
        # 残差连接
        hidden_states += residual
        
        # 前馈网络（Pre-Norm + 残差连接）
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        
        return hidden_states, present_key_value


class MiniMindModel(nn.Module):
    """
    MiniMind 基础模型
    
    完整的 Transformer 解码器模型，包含:
    1. Token 嵌入层
    2. 多个 Transformer 解码器层
    3. 最终的 LayerNorm
    4. 预计算的 RoPE 位置编码
    
    这是一个纯解码器模型，适用于因果语言建模任务。
    
    Attributes:
        embed_tokens: Token 嵌入层
        dropout: 嵌入 Dropout
        layers: Transformer 层列表
        norm: 最终的 LayerNorm
        freqs_cos, freqs_sin: 预计算的 RoPE 频率
    """
    
    def __init__(self, config: MiniMindConfig):
        """
        初始化基础模型
        
        Args:
            config: 模型配置对象
        """
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        
        # Token 嵌入层
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # 嵌入 Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer 解码器层
        self.layers = nn.ModuleList([
            MiniMindBlock(l, config) for l in range(self.num_hidden_layers)
        ])
        
        # 最终的 LayerNorm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 预计算 RoPE 位置编码
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling
        )
        
        # 注册为 buffer（不参与梯度计算，但会随模型保存/加载）
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **kwargs):
        """
        基础模型前向传播
        
        Args:
            input_ids: 输入 token ID，形状为 (batch_size, seq_len)
            attention_mask: 注意力掩码，1 表示有效位置
            past_key_values: 缓存的 KV 列表（每层一个）
            use_cache: 是否返回更新后的 KV 缓存
            **kwargs: 其他参数（兼容性）
            
        Returns:
            Tuple[Tensor, List, float]:
                - hidden_states: 最终隐藏状态，形状为 (batch_size, seq_len, hidden_size)
                - presents: 更新后的 KV 缓存列表
                - aux_loss: MoE 辅助损失（如果使用 MoE）
        """
        batch_size, seq_length = input_ids.shape
        
        # 处理 KV 缓存
        # 如果 past_key_values 是 HuggingFace 的 DynamicCache，转换为 None
        if hasattr(past_key_values, 'layers'):
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        
        # 计算起始位置（用于增量解码）
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # Token 嵌入 + Dropout
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # 获取当前位置的 RoPE 编码
        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )

        # 遍历所有 Transformer 层
        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        # 最终的 LayerNorm
        hidden_states = self.norm(hidden_states)

        # 计算 MoE 辅助损失（如果使用 MoE）
        aux_loss = sum(
            layer.mlp.aux_loss
            for layer in self.layers
            if isinstance(layer.mlp, MOEFeedForward)
        )

        return hidden_states, presents, aux_loss


class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    """
    MiniMind 因果语言模型
    
    在基础模型之上添加语言模型头（LM Head），用于预测下一个 token。
    继承自 HuggingFace 的 PreTrainedModel 和 GenerationMixin，
    支持 HuggingFace 生态系统的所有功能。
    
    架构:
    1. MiniMindModel: 基础 Transformer 模型
    2. lm_head: 线性层，将隐藏状态映射到词表
    
    权重共享:
    lm_head 的权重与 embed_tokens 共享，减少参数量。
    
    Attributes:
        model: 基础 Transformer 模型
        lm_head: 语言模型头
        OUT: 输出容器
    """
    config_class = MiniMindConfig  # 关联配置类

    def __init__(self, config: MiniMindConfig = None):
        """
        初始化因果语言模型
        
        Args:
            config: 模型配置对象，如果为 None 则使用默认配置
        """
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        
        # 基础 Transformer 模型
        self.model = MiniMindModel(self.config)
        
        # 语言模型头: hidden_size -> vocab_size
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        
        # 权重共享: lm_head 和 embed_tokens 共享权重
        # 这是一种常见的技术，可以减少参数量并提高性能
        self.model.embed_tokens.weight = self.lm_head.weight
        
        # 输出容器（用于存储各种输出）
        self.OUT = CausalLMOutputWithPast()

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **args):
        """
        因果语言模型前向传播
        
        Args:
            input_ids: 输入 token ID，形状为 (batch_size, seq_len)
            attention_mask: 注意力掩码
            past_key_values: 缓存的 KV
            use_cache: 是否使用 KV 缓存
            logits_to_keep: 保留多少个位置的 logits（用于节省内存）
                - 0: 保留所有
                - 正整数 n: 只保留最后 n 个位置
                - Tensor: 自定义索引
            **args: 其他参数
            
        Returns:
            CausalLMOutputWithPast: 包含以下字段:
                - logits: 预测的 logits，形状为 (batch_size, seq_len, vocab_size)
                - past_key_values: 更新后的 KV 缓存
                - last_hidden_state: 最后一层的隐藏状态
                - aux_loss: MoE 辅助损失
        """
        # 基础模型前向传播
        h, past_kvs, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )
        
        # 确定要保留的位置
        # 这是一个优化：在生成时，通常只需要最后一个位置的 logits
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        
        # 计算 logits: hidden_states -> vocab_size
        logits = self.lm_head(h[:, slice_indices, :])
        
        # 填充输出容器
        self.OUT.__setitem__('last_hidden_state', h)
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('past_key_values', past_kvs)
        
        return self.OUT
