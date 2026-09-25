import torch
import torch.nn as nn
from einops import rearrange

# ==============================================================================
# 方案 A: NVIDIA Cosmos Tokenizer (修正后的真实工程版)
# ==============================================================================
class CosmosCausal3DVAE(nn.Module):
    """
    封装 NVIDIA Cosmos Continuous 3D Causal VAE。
    采用连续型 (Continuous) 而非离散型 (Discrete)，因为流匹配需要连续的特征空间。
    """
    def __init__(self, model_dir, device='cuda'):
        super().__init__()
        # 【修正 1】: 正确的统一类名是 CausalVideoTokenizer
        from cosmos_tokenizer.video_lib import CausalVideoTokenizer
        
        # 【修正 2】: 加载 JIT 权重，文件名标准为 encoder.jit 和 decoder.jit
        self.tokenizer = CausalVideoTokenizer(
            checkpoint_enc=f"{model_dir}\encoder.jit",
            checkpoint_dec=f"{model_dir}\decoder.jit"
        ).to(device)
        
        # 冻结所有参数
        for param in self.tokenizer.parameters():
            param.requires_grad = False
        self.tokenizer.eval()
        
        # Cosmos CV 模型的默认潜空间维度是 16 通道
        self.latent_dim = 16 
        self.scale_factor = 0.18215 

    @torch.no_grad()
    def encode(self, x):
        """
        输入 x: (B, 3, T, H, W) 像素范围必须是 [-1, 1]
        """
        # 【修正 3】: Cosmos 的 encode 方法返回的是一个元组 (latent, )，我们需要取第 0 项
        out = self.tokenizer.encode(x)
        latents = out[0] 
        
        # 缩放潜空间以稳定 Flow Matching 的训练
        latents = latents * self.scale_factor
        
        # 提取 Latent 的时空维度
        B, C, T_l, H_l, W_l = latents.shape
        
        # 转换并展平为 Topo-DiT 所需的 (B, N, D)
        x_flat = rearrange(latents, 'b c t h w -> b (t h w) c')
        
        return x_flat, (T_l, H_l, W_l)

    @torch.no_grad()
    def decode(self, x_flat, T_l, H_l, W_l):
        """
        供推理解码使用
        输入 x_flat: (B, N, D)
        """
        # 恢复形状并反缩放
        x_flat = x_flat / self.scale_factor
        latents = rearrange(x_flat, 'b (t h w) c -> b c t h w', t=T_l, h=H_l, w=W_l)
        
        # 还原回像素 (B, 3, T, H, W) 范围 [-1, 1]
        video_pixels = self.tokenizer.decode(latents)
        return video_pixels



# ==============================================================================
# 方案 B: MagViT-2 风格的 Causal 3D VAE 
# 安装依赖: pip install magvit2-pytorch (lucidrains 复现版)
# ==============================================================================
class MagViT2Causal3DVAE(nn.Module):
    """
    封装基于 MagViT-2 架构的 Causal 3D VAE。
    如果你想用离散 Token 做自回归，或者提取其连续层特征，可使用此方案。
    """
    def __init__(self, model_path=None, device='cuda'):
        super().__init__()
        from magvit2_pytorch import MagViT2
        
        # 这里仅作配置示例。实际工业中需加载已用海量视频训好的权重。
        self.vae = MagViT2(
            image_size = 256,
            channels = 3,
            temporal_downsample_factor = 4, # 时间轴压缩 4 倍
            spatial_downsample_factor = 8,  # 空间轴压缩 8 倍
            dim = 1024,                     # VAE 内部特征维度
            use_causal_conv = True          # 开启因果卷积，防止未来帧泄露
        ).to(device)

        if model_path:
            self.vae.load_state_dict(torch.load(model_path, map_location=device))
            
        for param in self.vae.parameters():
            param.requires_grad = False
        self.vae.eval()
        
        self.latent_dim = self.vae.dim
        self.scale_factor = 0.23 # 经验缩放因子

    @torch.no_grad()
    def encode(self, x):
        """
        输入 x: (B, 3, T, H, W)
        """
        # 截取 MagViT-2 量化层之前的连续特征图作为 DiT 的 Latent
        # (B, D, T', H', W')
        latents, _, _ = self.vae.encode_to_continuous(x)
        latents = latents * self.scale_factor
        
        B, C, T_l, H_l, W_l = latents.shape
        x_flat = rearrange(latents, 'b c t h w -> b (t h w) c')
        
        return x_flat, (T_l, H_l, W_l)

    @torch.no_grad()
    def decode(self, x_flat, T_l, H_l, W_l):
        x_flat = x_flat / self.scale_factor
        latents = rearrange(x_flat, 'b (t h w) c -> b c t h w', t=T_l, h=H_l, w=W_l)
        # 从连续特征直接解码
        return self.vae.decode_from_continuous(latents)