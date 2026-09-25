import torch

class FlowMatcher:
    """
    最优传输流匹配 (Optimal Transport Flow Matching / Rectified Flow)
    目标是预测一条从纯噪声 x_0 走向真实数据 x_1 的直线速度场。
    """
    def __init__(self, sigma_min=1e-5):
        self.sigma_min = sigma_min

    def sample_time_steps(self, batch_size, device):
        """均匀采样时间步 t ~ U[0, 1]"""
        # 在实际工业训练中，可以使用 Logit-Normal 采样来增加网络对中间时间步的关注
        return torch.rand((batch_size,), device=device)

    def construct_flow_target(self, x_1):
        """
        x_1: (B, N, D) 或 (B, C, T, H, W) 真实的视频 Latent 特征
        返回: 
            x_t: 混合了噪声的中间状态
            v_target: 目标速度场 (网络需要预测的值)
            t: 采样的时间步
        """
        B = x_1.shape[0]
        device = x_1.device
        
        # x_0: 标准高斯纯噪声 (代表无序的起始状态)
        x_0 = torch.randn_like(x_1)
        
        # 采样时间步 t，并对齐维度以便广播
        t = self.sample_time_steps(B, device)
        t_expand = t.view(B, *([1] * (x_1.dim() - 1)))
        
        # 线性插值构建中间状态 (Rectified Flow 直线轨迹)
        # t=0 时完全是噪声 x_0，t=1 时完全是真实数据 x_1
        x_t = t_expand * x_1 + (1.0 - t_expand) * x_0
        
        # 目标速度场恒定为 (x_1 - x_0)
        v_target = x_1 - x_0
        
        return x_t, v_target, t