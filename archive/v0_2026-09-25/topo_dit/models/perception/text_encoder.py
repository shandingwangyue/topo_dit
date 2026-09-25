import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer

class TextConditionEncoder(nn.Module):
    """
    文本提示词编码器：将 String 转化为供 DiT 调制的 Condition Vector
    """
    def __init__(self, model_name="openai/clip-vit-large-patch14", device="cuda"):
        super().__init__()
        self.device = device
        
        # 加载 HuggingFace 的 Tokenizer 和 Text Model
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(model_name).to(device)
        
        # 冻结文本编码器参数（视频生成训练中不更新它）
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.text_encoder.eval()
        
        # CLIP ViT-L 的输出维度是 768
        self.output_dim = self.text_encoder.config.hidden_size 

    @torch.no_grad()
    def forward(self, texts):
        """
        texts: list of strings. 例如 ["A cyberpunk city", "A robot dancing"]
        返回: (Batch_Size, D) 的全局文本池化特征
        """
        # 1. 文本分词 (Tokenization)
        text_inputs = self.tokenizer(
            texts, 
            padding="max_length", 
            max_length=77, 
            truncation=True, 
            return_tensors="pt"
        ).to(self.device)
        
        # 2. 提取特征
        outputs = self.text_encoder(
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask
        )
        
        # 3. 获取池化后的全局特征 (Pooled Output)，代表整句话的全局语义
        # 形状: (B, 768)
        pooled_embeds = outputs.pooler_output 
        
        return pooled_embeds