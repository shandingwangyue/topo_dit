"""Optional offline CLIP cache encoder; not a random-conditioning fallback."""
import torch
from torch import nn


class TextConditionEncoder(nn.Module):
    def __init__(self, model_name, device='cpu', local_files_only=True):
        super().__init__()
        from transformers import CLIPTokenizer, CLIPTextModel
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        self.text_encoder = CLIPTextModel.from_pretrained(model_name, local_files_only=local_files_only).to(device)
        self.requires_grad_(False)
        self.eval()
        self.output_dim = self.text_encoder.config.hidden_size

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, texts):
        device = next(self.text_encoder.parameters()).device
        inputs = self.tokenizer(texts,padding=True,truncation=True,max_length=77,return_tensors='pt').to(device)
        return self.text_encoder(**inputs).pooler_output.float()
