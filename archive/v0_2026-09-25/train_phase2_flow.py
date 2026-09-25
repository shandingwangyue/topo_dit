import os
import argparse
import torch
import torch.optim as optim

from configs.config import TopoConfig
from topo_dit.models import TopoDiT_Model
from topo_dit.data.video_dataset import create_dataloader
from topo_dit.training.trainer import TopoDiTTrainer

def main():
    parser = argparse.ArgumentParser(description="Topo-DiT Phase 2: Flow Matching Dynamics")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt_phase1", type=str, required=True, help="Path to Phase 1 checkpoint")
    parser.add_argument("--save_dir", type=str, default="checkpoints/phase2")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    config = TopoConfig(args.config)
    
    model = TopoDiT_Model(
        dim=config.model.dim, cond_dim=config.model.cond_dim,
        num_elements=config.model.num_elements, depth=config.model.dit_depth
    )
    
    # [核心骤] 加载 Phase 1 习得的 Router 认知与 Decoder 残差权重
    print(f"Loading Phase 1 weights from {args.ckpt_phase1}")
    checkpoint = torch.load(args.ckpt_phase1, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    dataloader = create_dataloader(config.data)
    # 此处可能使用比 Phase 1 稍小的学习率
    optimizer = optim.AdamW(model.parameters(), lr=config.training.learning_rate * 0.5)
    
    trainer = TopoDiTTrainer(model, optimizer, config.training, device='cuda')
    
    for epoch in range(1, config.training.max_epochs + 1):
        # 执行第二阶段管线 (冻结 Router，专练 DiT 速度场预测)
        trainer.train_phase_2_dynamics(dataloader, epoch)
        
        if epoch % 5 == 0:
            ckpt_path = os.path.join(args.save_dir, f"topodit_phase2_ep{epoch}.pt")
            torch.save({'model_state_dict': model.state_dict()}, ckpt_path)
            print(f"Saved Phase 2 checkpoint to {ckpt_path}")

if __name__ == "__main__":
    main()