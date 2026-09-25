import torch
import torch.nn as nn

from .perception.causal_3d_vae import CosmosCausal3DVAE
from .perception.rope3d import ApplyPosEmb3D
from .router.stratifier import SinkhornStratifier
from .dynamics.element_dit import ElementDiT
from .renderer.broadcast import GatedBroadcastRenderer

class TopoDiT_Model(nn.Module):
    """
    Topo-DiT 认知视频生成架构 (顶层串联模型)
    包含：1.时空感知引擎 2.阶层化认知路由器 3.拓扑动力学主干 4.全息广播渲染器
    """
    def __init__(self, dim=1024, cond_dim=512, num_elements=256, depth=12, num_heads=8):
        super().__init__()
        self.dim = dim
        
        # 1. 时空感知引擎 (Perception)
        # 实例化真实的 VAE
        self.vae = CosmosCausal3DVAE(model_dir="D:\\topo_dit_project\\Cosmos-Tokenizer", device='cuda')
        # 注意：这里的 dim 要与 VAE 的输出通道对齐（如 Cosmos 是 16，你可以用个 Linear 升维到 1024）
        self.latent_proj = nn.Linear(self.vae.latent_dim, dim)
        
        # 2. 阶层化认知路由器 (Cognitive Router)
        self.router = SinkhornStratifier(dim=dim, num_elements=num_elements)
        
        # 3. 拓扑动力学主干 (Topological Dynamics)
        self.dynamics = ElementDiT(dim=dim, cond_dim=cond_dim, depth=depth, num_heads=num_heads)
        
        # 4. 全息广播渲染器 (Broadcast Renderer)
        self.renderer = GatedBroadcastRenderer(dim=dim)

    def forward(self, video_x_t, condition):
        # 1. 真实 VAE 编码：提取 Latent 并获取压缩后的真实网格维度
        grid_tokens, (T_l, H_l, W_l) = self.vae.encode(video_x_t)
        
        # 2. 如果 VAE 维度小（如 16），映射到我们认知的模型维度（1024）
        grid_tokens = self.latent_proj(grid_tokens)
        
        # 3. RoPE 3D 必须使用压缩后的 T_l, H_l, W_l
        grid_tokens = self.pos_emb(grid_tokens, T_l, H_l, W_l)
        
        # 4. 进入后续的路由与动力学...
        element_tokens, A_matrix = self.router(grid_tokens)
        element_v_pred = self.dynamics(element_tokens, condition)
        grid_v_pred = self.renderer(element_v_pred, A_matrix, grid_tokens)
        
        return grid_v_pred, A_matrix, element_tokens