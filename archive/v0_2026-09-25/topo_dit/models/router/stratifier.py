import torch
import torch.nn as nn
from .sinkhorn_ot import sinkhorn_knopp

class SinkhornStratifier(nn.Module):
    """阶层化认知路由器：将网格(Grid)映射为拓扑域实体(Element)"""
    def __init__(self, dim, num_elements, sinkhorn_iters=3):
        super().__init__()
        self.num_elements = num_elements
        self.sinkhorn_iters = sinkhorn_iters
        
        # 认知锚点 (Topological Domain Anchors)
        self.domain_anchors = nn.Parameter(torch.randn(1, num_elements, dim))
        self.proj_q = nn.Linear(dim, dim, bias=False)
        self.proj_k = nn.Linear(dim, dim, bias=False)

    def forward(self, grid_tokens):
        B, N, D = grid_tokens.shape
        
        # 1. 计算路由代价矩阵 (Cost Matrix)
        Q = self.proj_q(self.domain_anchors).expand(B, -1, -1) # (B, K, D)
        K_grid = self.proj_k(grid_tokens)                      # (B, N, D)
        cost_matrix = -torch.bmm(K_grid, Q.transpose(1, 2))    # (B, N, K)
        
        # 2. Sinkhorn 最优传输路由
        A_matrix = sinkhorn_knopp(cost_matrix, iters=self.sinkhorn_iters)
        
        # 3. 特征坍缩 (Grid -> Element)
        A_T = A_matrix.transpose(1, 2) 
        # 规范化：确保分配总和能量守恒，避免某个 Anchor 因吸附过多像素而梯度爆炸
        A_T_norm = A_T / (A_T.sum(dim=-1, keepdim=True) + 1e-6)
        
        element_tokens = torch.bmm(A_T_norm, grid_tokens)      # (B, K, D)
        return element_tokens, A_matrix