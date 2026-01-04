"""
LoRA (Low-Rank Adaptation) 实现

LoRA 是一种参数高效微调方法，通过在预训练模型的权重矩阵旁边添加
低秩分解矩阵来实现微调，而不修改原始权重。

核心思想:
- 原始权重 W 保持冻结
- 添加低秩分解: ΔW = A @ B，其中 A ∈ R^(d×r), B ∈ R^(r×k)
- 前向传播: y = (W + ΔW) @ x = W @ x + A @ B @ x
- 只训练 A 和 B，参数量从 d×k 降低到 (d+k)×r

优点:
1. 参数高效: 只需训练很少的参数（通常 < 1%）
2. 内存高效: 不需要存储完整的梯度
3. 可插拔: 可以轻松切换不同的 LoRA 适配器
4. 无推理延迟: 可以将 LoRA 权重合并到原始权重中

本文件实现:
- LoRALinear: 带 LoRA 的线性层
- apply_lora: 将 LoRA 应用到模型
- load_lora: 加载 LoRA 权重
- save_lora: 保存 LoRA 权重
"""

import torch
from torch import nn


class LoRALinear(nn.Module):
    """
    带 LoRA 适配器的线性层
    
    在原始线性层的基础上添加低秩分解矩阵 A 和 B。
    前向传播时，输出 = 原始输出 + (x @ A @ B) * scaling
    
    LoRA 公式:
    h = W @ x + (A @ B) @ x * (alpha / r)
    
    其中:
    - W: 原始权重矩阵（冻结）
    - A: 低秩矩阵 A，形状为 (in_features, r)
    - B: 低秩矩阵 B，形状为 (r, out_features)
    - r: 秩（rank），控制 LoRA 的容量
    - alpha: 缩放因子，控制 LoRA 的影响程度
    - scaling = alpha / r: 最终缩放系数
    
    初始化:
    - A: 使用 Kaiming 均匀初始化
    - B: 初始化为零，确保训练开始时 LoRA 不影响输出
    
    Attributes:
        original_linear (nn.Linear): 原始线性层（权重冻结）
        lora_A (nn.Parameter): 低秩矩阵 A
        lora_B (nn.Parameter): 低秩矩阵 B
        scaling (float): 缩放系数 alpha / r
    """
    
    def __init__(self, original_linear, r=8, alpha=16):
        """
        初始化 LoRA 线性层
        
        Args:
            original_linear (nn.Linear): 原始线性层
            r (int): LoRA 秩，控制低秩分解的维度
                    较小的 r 意味着更少的参数，但可能限制表达能力
                    常用值: 4, 8, 16, 32
            alpha (int): 缩放因子，控制 LoRA 对输出的影响程度
                        通常设置为 r 的 1-2 倍
                        较大的 alpha 意味着 LoRA 的影响更大
        """
        super().__init__()
        
        # 保存原始线性层
        self.original_linear = original_linear
        
        # 获取原始线性层的维度
        in_features = original_linear.in_features   # 输入维度
        out_features = original_linear.out_features  # 输出维度
        
        # 初始化 LoRA 矩阵
        # A: (in_features, r) - 将输入投影到低秩空间
        # B: (r, out_features) - 将低秩表示投影回输出空间
        self.lora_A = nn.Parameter(torch.zeros(in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        
        # 计算缩放系数
        # scaling = alpha / r，用于控制 LoRA 的影响程度
        self.scaling = alpha / r
        
        # 初始化 LoRA 权重
        self._init_lora_weights()

    def _init_lora_weights(self):
        """
        初始化 LoRA 权重
        
        初始化策略:
        - A: 使用 Kaiming 均匀初始化，适合 ReLU 类激活函数
        - B: 初始化为零
        
        这种初始化确保训练开始时 LoRA 的输出为零，
        即模型行为与原始预训练模型完全相同。
        随着训练进行，LoRA 逐渐学习任务特定的调整。
        """
        # Kaiming 初始化 A
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        # 零初始化 B
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        """
        前向传播
        
        计算: output = original_output + lora_output * scaling
        
        其中:
        - original_output = x @ W^T + b（原始线性层的输出）
        - lora_output = x @ A @ B（LoRA 的增量输出）
        
        Args:
            x: 输入张量，形状为 (..., in_features)
            
        Returns:
            输出张量，形状为 (..., out_features)
        """
        # 原始线性层的输出
        original_output = self.original_linear(x)
        
        # LoRA 的增量输出
        # x @ A: (..., in_features) @ (in_features, r) -> (..., r)
        # (x @ A) @ B: (..., r) @ (r, out_features) -> (..., out_features)
        lora_output = (x @ self.lora_A @ self.lora_B) * self.scaling
        
        # 合并输出
        return original_output + lora_output


def apply_lora(model, r=8, alpha=16, target_modules=None):
    """
    将 LoRA 应用到模型的指定层
    
    遍历模型的所有模块，将目标线性层替换为 LoRALinear。
    默认目标是注意力层的 Q、K、V 投影。
    
    Args:
        model: 要应用 LoRA 的模型
        r (int): LoRA 秩
        alpha (int): 缩放因子
        target_modules (list): 要应用 LoRA 的模块名称列表
                              默认为 ['q_proj', 'k_proj', 'v_proj']
                              
    Returns:
        None（原地修改模型）
        
    Example:
        >>> model = MiniMindForCausalLM(config)
        >>> apply_lora(model, r=8, alpha=16)
        >>> # 现在 model 的 Q、K、V 投影层都带有 LoRA
    """
    # 默认目标模块: 注意力层的 Q、K、V 投影
    if target_modules is None:
        target_modules = ['q_proj', 'k_proj', 'v_proj']
    
    # 遍历模型的所有命名模块
    for name, module in model.named_modules():
        # 检查模块名称是否匹配目标
        for target in target_modules:
            if target in name and isinstance(module, nn.Linear):
                # 获取父模块和属性名
                parent_name = '.'.join(name.split('.')[:-1])
                attr_name = name.split('.')[-1]
                
                # 获取父模块
                parent = model
                if parent_name:
                    for part in parent_name.split('.'):
                        parent = getattr(parent, part)
                
                # 创建 LoRA 线性层并替换原始层
                lora_linear = LoRALinear(module, r=r, alpha=alpha)
                setattr(parent, attr_name, lora_linear)


def save_lora(model, path):
    """
    保存 LoRA 权重
    
    只保存 LoRA 相关的参数（lora_A 和 lora_B），
    不保存原始模型的权重。这使得 LoRA 权重文件非常小。
    
    Args:
        model: 带有 LoRA 的模型
        path (str): 保存路径
        
    Example:
        >>> save_lora(model, 'lora_weights.pth')
        >>> # 文件大小通常只有几 MB
    """
    # 收集所有 LoRA 参数
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if 'lora' in name:
            lora_state_dict[name] = param.data
    
    # 保存到文件
    torch.save(lora_state_dict, path)
    print(f"LoRA 权重已保存到: {path}")
    print(f"保存的参数数量: {len(lora_state_dict)}")


def load_lora(model, path):
    """
    加载 LoRA 权重
    
    从文件加载 LoRA 参数并应用到模型。
    注意: 模型必须已经应用了 LoRA（使用 apply_lora）。
    
    Args:
        model: 带有 LoRA 的模型
        path (str): LoRA 权重文件路径
        
    Example:
        >>> apply_lora(model)  # 先应用 LoRA 结构
        >>> load_lora(model, 'lora_weights.pth')  # 再加载权重
    """
    # 加载 LoRA 权重
    lora_state_dict = torch.load(path, map_location='cpu')
    
    # 获取模型当前的 state_dict
    model_state_dict = model.state_dict()
    
    # 更新 LoRA 参数
    for name, param in lora_state_dict.items():
        if name in model_state_dict:
            model_state_dict[name] = param
        else:
            print(f"警告: 参数 {name} 在模型中不存在")
    
    # 加载更新后的 state_dict
    model.load_state_dict(model_state_dict, strict=False)
    print(f"LoRA 权重已从 {path} 加载")


def merge_lora(model):
    """
    将 LoRA 权重合并到原始权重中
    
    合并后，模型不再需要 LoRA 层，可以像普通模型一样使用。
    这消除了 LoRA 带来的额外计算开销。
    
    合并公式:
    W_merged = W_original + A @ B * scaling
    
    Args:
        model: 带有 LoRA 的模型
        
    Returns:
        None（原地修改模型）
        
    注意:
        合并后无法再单独保存或修改 LoRA 权重。
        如果需要保留 LoRA 的灵活性，请不要合并。
    """
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            # 计算 LoRA 的增量权重
            # delta_W = A @ B * scaling
            delta_weight = (module.lora_A @ module.lora_B * module.scaling).T
            
            # 合并到原始权重
            module.original_linear.weight.data += delta_weight
            
            # 获取父模块
            parent_name = '.'.join(name.split('.')[:-1])
            attr_name = name.split('.')[-1]
            
            parent = model
            if parent_name:
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
            
            # 用原始线性层替换 LoRA 层
            setattr(parent, attr_name, module.original_linear)
    
    print("LoRA 权重已合并到原始模型")


def count_lora_parameters(model):
    """
    统计 LoRA 参数数量
    
    Args:
        model: 带有 LoRA 的模型
        
    Returns:
        Tuple[int, int, float]: (LoRA 参数数, 总参数数, LoRA 占比)
    """
    lora_params = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if 'lora' in name:
            lora_params += param.numel()
    
    ratio = lora_params / total_params * 100 if total_params > 0 else 0
    
    return lora_params, total_params, ratio
