import torch
import torch.nn.functional as F

def compute_cognitive_losses(A_matrix, element_tokens):
    """
    计算保证路由(Router)物理意义的结构化损失
    A_matrix: (B, N, K) - 路由分配矩阵
    element_tokens: (B, K, D) - 提取出的实体特征
    """
    # 1. 稀疏性/明确性损失 (Entropy Loss)
    # 物理意义：强迫每个底层像素网格(Grid)明确归属于某一个元素(Element)，而不是糊涂地平均分配。
    # 熵越低，分配越坚决，物体的物理边界越清晰。
    # 加上 1e-8 防止 log(0) 导致 NaN
    entropy_loss = -torch.sum(A_matrix * torch.log(A_matrix + 1e-8), dim=-1).mean()
    
    # 2. 正交分离损失 (Orthogonality Loss)
    # 物理意义：惩罚 K 个锚点（Anchors）提取出相同的特征。
    # 强迫它们各自负责不同的拓扑域（如肤色、衣物、背景光影），彻底杜绝模式坍塌 (Mode Collapse)。
    normalized_elements = F.normalize(element_tokens, p=2, dim=-1)
    
    # 计算余弦相似度矩阵 (B, K, K)
    cosine_sim = torch.bmm(normalized_elements, normalized_elements.transpose(1, 2))
    
    # 构建目标单位矩阵 (对角线为1，非对角线为0)
    eye = torch.eye(cosine_sim.shape[-1], device=cosine_sim.device).unsqueeze(0)
    eye = eye.expand_as(cosine_sim)
    
    # 惩罚非对角线元素（即惩罚不同元素间的相似度）
    ortho_loss = F.mse_loss(cosine_sim, eye)
    
    return entropy_loss, ortho_loss