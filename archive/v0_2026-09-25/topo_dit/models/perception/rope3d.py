import torch
import torch.nn as nn
import math

class ApplyPosEmb3D(nn.Module):
    """3D 绝对位置编码 (支持时间、高度、宽度维度的解耦)"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        assert dim % 6 == 0, "Dimension must be divisible by 6 for 3D positional embedding (T, H, W)."
        
    def forward(self, x, T, H, W):
        B, N, D = x.shape
        device = x.device
        
        t_pos = torch.arange(T, device=device).unsqueeze(1).expand(-1, H*W).reshape(-1)
        h_pos = torch.arange(H, device=device).unsqueeze(1).expand(-1, W).reshape(-1).repeat(T)
        w_pos = torch.arange(W, device=device).repeat(T*H)
        
        div_term = torch.exp(torch.arange(0, D//3, 2, device=device) * -(math.log(10000.0) / (D//3)))
        
        pe = torch.zeros(N, D, device=device)
        d3 = D // 3
        # T-axis
        pe[:, 0:d3:2] = torch.sin(t_pos.unsqueeze(1) * div_term)
        pe[:, 1:d3:2] = torch.cos(t_pos.unsqueeze(1) * div_term)
        # H-axis
        pe[:, d3:2*d3:2] = torch.sin(h_pos.unsqueeze(1) * div_term)
        pe[:, d3+1:2*d3:2] = torch.cos(h_pos.unsqueeze(1) * div_term)
        # W-axis
        pe[:, 2*d3::2] = torch.sin(w_pos.unsqueeze(1) * div_term)
        pe[:, 2*d3+1::2] = torch.cos(w_pos.unsqueeze(1) * div_term)
        
        return x + pe.unsqueeze(0)