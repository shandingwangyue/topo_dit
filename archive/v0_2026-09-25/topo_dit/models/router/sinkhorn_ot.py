import torch

def sinkhorn_knopp(cost_matrix, epsilon=0.1, iters=3):
    """
    Log-space Sinkhorn 算法，计算最优传输分配矩阵 (OT Allocation Matrix)
    确保数值稳定性 (Log-sum-exp trick)
    """
    M = -cost_matrix / epsilon
    u = torch.zeros_like(M[:, :, 0])  # (B, N)
    v = torch.zeros_like(M[:, 0, :])  # (B, K)
    
    for _ in range(iters):
        # Update u
        u = -torch.logsumexp(M - v.unsqueeze(1), dim=-1)
        # Update v
        v = -torch.logsumexp(M - u.unsqueeze(2), dim=1)
        
    # 重构回概率分布矩阵
    A_matrix = torch.exp(M - u.unsqueeze(2) - v.unsqueeze(1))
    return A_matrix