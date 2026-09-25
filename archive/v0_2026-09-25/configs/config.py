import os
import yaml
from pprint import pformat

class DotDict(dict):
    """
    将 Python 字典转换为可以通过点运算符访问的对象
    例如: config['model']['dim'] -> config.model.dim
    """
    def __getattr__(self, key):
        value = self.get(key)
        if isinstance(value, dict):
            value = DotDict(value)
        return value

    def __setattr__(self, key, value):
        self[key] = value

def dict_to_dotdict(d):
    if isinstance(d, dict):
        return DotDict({k: dict_to_dotdict(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_dotdict(v) for v in d]
    else:
        return d

class TopoConfig:
    """
    Topo-DiT 统一配置管理器
    """
    def __init__(self, yaml_path=None):
        # 1. 默认配置 (防止 yaml 文件中缺少某些字段)
        self.config_dict = {
            'model': {
                'dim': 768,
                'cond_dim': 512,
                'num_elements': 128,
                'dit_depth': 12,
                'num_heads': 12,
                'sinkhorn_iters': 3
            },
            'data': {
                'meta_path': 'metadata.jsonl',
                'video_dir': 'videos/',
                'num_frames': 16,
                'frame_stride': 1,
                'spatial_size': 256,
                'batch_size': 2,
                'num_workers': 4
            },
            'training': {
                'learning_rate': 1e-4,
                'weight_decay': 0.01,
                'max_epochs': 50,
                'lambda_ent': 0.05,
                'lambda_ortho': 0.05,
                'current_phase': 1
            }
        }
        
        # 2. 从 YAML 文件加载并覆盖默认配置
        if yaml_path and os.path.exists(yaml_path):
            with open(yaml_path, 'r', encoding='utf-8') as f:
                loaded_config = yaml.safe_load(f)
                self._update_dict(self.config_dict, loaded_config)
        elif yaml_path:
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
            
        # 3. 转换为 DotDict
        self.cfg = dict_to_dotdict(self.config_dict)

    def _update_dict(self, base, new):
        """递归更新嵌套字典"""
        for k, v in new.items():
            if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                self._update_dict(base[k], v)
            else:
                base[k] = v
                
    def __getattr__(self, name):
        """直接代理到内部的 cfg 属性，实现 config.model.dim 访问"""
        return getattr(self.cfg, name)

    def __str__(self):
        return f"TopoDiT Configuration:\n{pformat(self.config_dict, indent=2)}"


# ==============================================================================
# 本地验证代码
# ==============================================================================
if __name__ == "__main__":
    import tempfile
    
    # 模拟生成一个 yaml 文件供测试
    yaml_content = """
    model:
      dim: 1024
      num_elements: 256
    training:
      current_phase: 2
      lambda_ent: 0.15
    """
    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.yaml') as f:
        f.write(yaml_content)
        tmp_path = f.name
        
    try:
        # 加载配置
        config = TopoConfig(yaml_path=tmp_path)
        
        # 验证读取与点运算符代理
        print(config)
        print("\n--- Testing Attribute Access ---")
        print(f"Model Dim: {config.model.dim}")                  # 输出 1024 (由 YAML 覆盖)
        print(f"Num Elements (K): {config.model.num_elements}")  # 输出 256
        print(f"Entropy Lambda: {config.training.lambda_ent}")   # 输出 0.15
        print(f"Data Batch Size: {config.data.batch_size}")      # 输出 2 (使用默认值)
        
    finally:
        os.remove(tmp_path)