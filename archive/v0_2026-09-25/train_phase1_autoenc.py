import os
import argparse
import torch
import torch.optim as optim

# 引入我们写好的核心模块
from configs.config import TopoConfig
from topo_dit.models import TopoDiT_Model
from topo_dit.data.video_dataset import create_dataloader
from topo_dit.training.trainer import TopoDiTTrainer

def main():
    parser = argparse.ArgumentParser(description="Topo-DiT Phase 1: Routing & Reconstruction")
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config file")
    parser.add_argument("--save_dir", type=str, default="checkpoints/phase1", help="Directory to save ckpts")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    
    # 1. 加载配置
    config = TopoConfig(args.config)
    print(f"Starting Phase 1 Training with config: {args.config}")
    
    # 2. 初始化四大引擎串联的模型
    model = TopoDiT_Model(
        dim=config.model.dim,
        cond_dim=config.model.cond_dim,
        num_elements=config.model.num_elements,
        depth=config.model.dit_depth,
        num_heads=config.model.num_heads
    )
    
    # 3. 数据与优化器
    dataloader = create_dataloader(config.data)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.training.learning_rate, 
        weight_decay=config.training.weight_decay
    )
    
    # 4. 实例化 Trainer
    trainer = TopoDiTTrainer(model, optimizer, config.training, device='cuda')
    
    # 5. 开启 Epoch 训练循环
    for epoch in range(1, config.training.max_epochs + 1):
        # 执行第一阶段管线 (冻结 DiT，只练 Router 和 Renderer)
        trainer.train_phase_1_routing(dataloader, epoch)
        
        # 定期保存 Checkpoint
        if epoch % 5 == 0 or epoch == config.training.max_epochs:
            ckpt_path = os.path.join(args.save_dir, f"topodit_phase1_ep{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

if __name__ == "__main__":
    main()