import os
import json
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 工业界加载视频的标配，比 OpenCV / imageio 快得多，且支持精准帧跳转
try:
    import decord
    decord.bridge.set_bridge('torch')
except ImportError:
    raise ImportError("Please install decord: pip install decord")


class TopoVideoDataset(Dataset):
    """
    Topo-DiT 视频数据集加载器
    支持超长视频的流式 Chunk 采样、时序降采样(Stride)与动态分辨率裁剪。
    """
    def __init__(
        self, 
        meta_path, 
        video_dir="", 
        num_frames=16, 
        frame_stride=1, 
        spatial_size=256,
        is_training=True
    ):
        """
        meta_path: 描述文件路径 (JSONL格式，包含 video_path 和 text_prompt)
        video_dir: 视频根目录
        num_frames: 每次采样的 Token 时长 T (例如 16, 32)
        frame_stride: 抽帧步长，用于控制视频的时间感受野
        spatial_size: 空间裁剪分辨率 (H, W)
        """
        super().__init__()
        self.video_dir = video_dir
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.spatial_size = spatial_size
        self.is_training = is_training
        
        # 1. 加载元数据 (假定是每行一个 JSON 的 JSONL 格式)
        self.metadata = []
        with open(meta_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.metadata.append(json.loads(line))
                    
        print(f"Loaded {len(self.metadata)} video records from {meta_path}.")

        # 2. 空间图像增强与标准化 (映射到 [-1, 1] 供 Flow Matching 使用)
        self.transform = transforms.Compose([
            transforms.Resize(spatial_size, antialias=True),
            transforms.CenterCrop(spatial_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def _get_video_chunk(self, video_path):
        """核心方法：从超长视频中精准切取 T 帧的 Chunk"""
        try:
            # ctx=decord.cpu(0) 将解码放在 CPU 端，防止阻塞 GPU 显存
            vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
            total_frames = len(vr)
            
            # 计算所需的物理帧跨度
            required_frames = (self.num_frames - 1) * self.frame_stride + 1
            
            # 策略 A: 视频足够长，随机截取一个 Chunk (适用于长视频训练)
            if total_frames >= required_frames:
                if self.is_training:
                    # 随机起始帧
                    start_idx = random.randint(0, total_frames - required_frames)
                else:
                    # 推理或验证时，取最中间的 Chunk
                    start_idx = (total_frames - required_frames) // 2
                frame_indices = range(start_idx, start_idx + required_frames, self.frame_stride)
            
            # 策略 B: 视频太短，循环填充 (Padding / Looping)
            else:
                # 均等采样或循环补齐，这里采用简单的首尾循环补齐策略
                frame_indices = [
                    (i * self.frame_stride) % total_frames for i in range(self.num_frames)
                ]
                
            # 一次性批量读取目标帧，底层调用 FFmpeg 高效解码
            # video_tensor shape: (T, H, W, C) - 值域 [0, 255]
            video_tensor = vr.get_batch(list(frame_indices))
            
            # 维度转换: (T, H, W, C) -> (C, T, H, W) 
            video_tensor = video_tensor.permute(3, 0, 1, 2).float() / 255.0
            
            # 逐帧进行空间 Transform (Resize & Normalize to [-1, 1])
            # 将 (C, T, H, W) reshape 为 (C, H, W) 进行 torchvision 处理
            C, T, H, W = video_tensor.shape
            video_tensor = video_tensor.transpose(0, 1) # (T, C, H, W)
            video_tensor = torch.stack([self.transform(frame) for frame in video_tensor]) # (T, C, H, W)
            video_tensor = video_tensor.transpose(0, 1) # (C, T, H, W)
            
            return video_tensor

        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            return None

    def _get_text_condition(self, text_prompt):
        """
        获取文本条件特征。
        在真实的工程中，为节省显存，通常会提前用 T5-XXL 提取好特征保存为 .pt 文件。
        这里使用随机张量占位模拟。
        """
        # TODO: 替换为真实的 Tokenizer & Text Encoder (如 CLIP 或 T5)
        # return torch.load(text_prompt_embedding_path)
        cond_dim = 512
        return torch.randn(cond_dim)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        record = self.metadata[idx]
        
        # 拼接完整路径
        video_path = os.path.join(self.video_dir, record['video_path'])
        text_prompt = record.get('text_prompt', '')
        
        # 1. 抽取视频 Chunk (C, T, H, W)
        video_tensor = self._get_video_chunk(video_path)
        
        # 容错处理：如果视频损坏，随机递归读取下一个
        if video_tensor is None:
            return self.__getitem__(random.randint(0, len(self) - 1))
            
        # 2. 获取文本条件特征 (D,)
        condition_tensor = self._get_text_condition(text_prompt)
        
        # 返回字典，与 topo_dit/training/trainer.py 的 batch 取值键强对齐
        # 注意：此处的 'video_latent' 在实际传给模型前，是 raw RGB 像素，
        # 我们在 trainer.py 中直接送入模型，由 VAE 编码为真实的 Latent。
        return {
            'video_latent': video_tensor,   # (3, T, H, W) 
            'condition': condition_tensor,  # (COND_DIM)
            'text': text_prompt             # 原文本 (供 Debug 或 Log 使用)
        }


# ==============================================================================
# 数据集测试与实例化方法
# ==============================================================================
def create_dataloader(config):
    """供外界（如 Trainer）调用的工厂方法"""
    dataset = TopoVideoDataset(
        meta_path=config.meta_path,
        video_dir=config.video_dir,
        num_frames=config.num_frames,
        frame_stride=config.frame_stride,
        spatial_size=config.spatial_size,
        is_training=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,     # 加速数据向 GPU 显存的转移
        drop_last=True
    )
    return dataloader

if __name__ == "__main__":
    # 本地测试代码
    import tempfile
    
    # 创建一个伪造的 metadata JSONL 文件
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
        f.write('{"video_path": "fake1.mp4", "text_prompt": "A cyberpunk city"}\n')
        f.write('{"video_path": "fake2.mp4", "text_prompt": "A robot dancing"}\n')
        tmp_meta_path = f.name
        
    print("Testing DataLoader instantiation...")
    try:
        dataset = TopoVideoDataset(
            meta_path=tmp_meta_path,
            video_dir="/tmp",
            num_frames=16,
            spatial_size=256
        )
        print(f"Dataset size: {len(dataset)}")
        # 实际取值需要存在真实的 mp4 文件，这里仅验证逻辑是否贯通
    finally:
        os.remove(tmp_meta_path)