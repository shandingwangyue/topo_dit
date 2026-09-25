import torch
import torch.nn as nn

class GatedBroadcastRenderer(nn.Module):
    """全息广播渲染器：将稀疏的拓扑演化信息，残差映射回稠密网格空间"""
    def __init__(self, dim):
        super().__init__()
        self.proj_out = nn.Linear(dim, dim)
        # 初始化为0，确保冷启动时相当于恒等映射(Identity Mapping)，极大地稳定训练
        self.gamma = nn.Parameter(torch.zeros(1)) 

    def forward(self, element_v_pred, A_matrix, grid_tokens_original):
        """
        element_v_pred: (B, K, D) 动力学预测的元素变化
        A_matrix: (B, N, K) Router 分配矩阵
        grid_tokens_original: (B, N, D) 输入的底噪网格
        """
        # 1. 广播 (Broadcast / Scatter)
        # (B, N, K) @ (B, K, D) -> (B, N, D)
        delta_grid = torch.bmm(A_matrix, element_v_pred) 
        
        # 2. 门控残差投射 (Gated Residual)
        grid_v_pred = grid_tokens_original + self.gamma * self.proj_out(delta_grid)
        
        return grid_v_pred