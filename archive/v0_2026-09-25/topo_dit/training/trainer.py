import torch
import torch.nn.functional as F
from .regularizers import compute_cognitive_losses
from .flow_matching import FlowMatcher

class TopoDiTTrainer:
    def __init__(self, model, optimizer, config, device='cuda'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.flow_matcher = FlowMatcher()

    def _train_step(self, loss):
        """执行优化与梯度裁剪"""
        self.optimizer.zero_grad()
        loss.backward()
        # 极度重要：防止 Sinkhorn 层梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

    # ==========================================
    # Phase 1: 认知架构冷启动 (路由与重建)
    # ==========================================
    def train_phase_1_routing(self, dataloader, epoch):
        """
        Phase 1: 自编码重构。
        冻结 DiT 主干，专注于训练 Router (Grid->Element) 和 Renderer (Element->Grid)。
        输入无噪真实视频，目标是完美重构，确立物体的物理边界。
        """
        print(f"[Phase 1] Training Routing & Rendering - Epoch {epoch}")
        self.model.train()
        
        # 冻结拓扑动力学主干
        self.model.dynamics.requires_grad_(False)
        self.model.router.requires_grad_(True)
        self.model.renderer.requires_grad_(True)
        
        for step, batch in enumerate(dataloader):
            # 获取无噪视频 (假设预处理已转入 Latent)
            video_x1 = batch['video_latent'].to(self.device) 
            cond = batch['condition'].to(self.device)
            B, C, T, H, W = video_x1.shape
            
            # 因为是重构，时间步固定设为 1.0 (表示真实数据)
            t_dummy = torch.ones(B, device=self.device)
            
            # 前向推演 (无噪重构)
            grid_v_pred, A_matrix, E_tokens = self.model(video_x1, cond, T, H, W)
            
            # 展平 Target 方便算 Loss
            grid_target = video_x1.permute(0, 2, 3, 4, 1).reshape(B, T*H*W, -1)
            
            # 损失 1：重构损失 (MSE)
            loss_recon = F.mse_loss(grid_v_pred, grid_target)
            
            # 损失 2：认知结构正则化 (极其关键)
            loss_ent, loss_ortho = compute_cognitive_losses(A_matrix, E_tokens)
            
            # 联合损失计算
            loss = loss_recon + self.config.lambda_ent * loss_ent + self.config.lambda_ortho * loss_ortho
            self._train_step(loss)
            
            if step % 100 == 0:
                print(f"Step {step} | Recon: {loss_recon.item():.4f} | Ent: {loss_ent.item():.4f} | Ortho: {loss_ortho.item():.4f}")

    # ==========================================
    # Phase 2: 物理流匹配 (拓扑动力学训练)
    # ==========================================
    def train_phase_2_dynamics(self, dataloader, epoch):
        """
        Phase 2: OT-Flow Matching 流匹配。
        冻结 Router 和 Renderer（保留 Phase 1 习得的元素认知），专门训练 DiT。
        让网络在 Element 空间学习物体如何随时间和条件(文本)演变。
        """
        print(f"[Phase 2] Training Topological Dynamics (Flow Matching) - Epoch {epoch}")
        self.model.train()
        
        # 切换梯度锁
        self.model.router.requires_grad_(False)
        self.model.renderer.requires_grad_(False)
        self.model.dynamics.requires_grad_(True)
        
        for step, batch in enumerate(dataloader):
            video_x1 = batch['video_latent'].to(self.device)
            cond = batch['condition'].to(self.device)
            B, C, T, H, W = video_x1.shape
            
            # 构造流匹配目标：x_t (加噪状态), v_target (速度场), t (时间步)
            x_t, v_target, t = self.flow_matcher.construct_flow_target(video_x1)
            
            # 模型预测当前的速度场 v_pred
            v_pred, _, _ = self.model(x_t, cond, T, H, W)
            
            # Target 展平
            v_target_flat = v_target.permute(0, 2, 3, 4, 1).reshape(B, T*H*W, -1)
            
            # 纯流匹配损失 (MSE between predicted vector field and target vector field)
            loss_fm = F.mse_loss(v_pred, v_target_flat)
            
            self._train_step(loss_fm)
            
            if step % 100 == 0:
                print(f"Step {step} | Flow Matching Loss: {loss_fm.item():.4f}")

    # ==========================================
    # Phase 3: 端到端联合对齐微调
    # ==========================================
    def train_phase_3_joint(self, dataloader, epoch):
        """
        Phase 3: 联合微调。
        放开所有参数，使用极低学习率，确保认知路由与动力学完美融合。
        """
        print(f"[Phase 3] Joint Alignment Fine-Tuning - Epoch {epoch}")
        self.model.train()
        self.model.requires_grad_(True) # 全部解冻
        
        for step, batch in enumerate(dataloader):
            video_x1 = batch['video_latent'].to(self.device)
            cond = batch['condition'].to(self.device)
            B, C, T, H, W = video_x1.shape
            
            x_t, v_target, t = self.flow_matcher.construct_flow_target(video_x1)
            v_pred, A_matrix, E_tokens = self.model(x_t, cond, T, H, W)
            
            v_target_flat = v_target.permute(0, 2, 3, 4, 1).reshape(B, T*H*W, -1)
            
            loss_fm = F.mse_loss(v_pred, v_target_flat)
            loss_ent, loss_ortho = compute_cognitive_losses(A_matrix, E_tokens)
            
            # 联合 Loss，正则化权重降低，主打流匹配
            loss = loss_fm + (self.config.lambda_ent * 0.1) * loss_ent + (self.config.lambda_ortho * 0.1) * loss_ortho
            self._train_step(loss)
            
            if step % 100 == 0:
                print(f"Step {step} | Joint Loss: {loss.item():.4f} | FM: {loss_fm.item():.4f}")