from typing import Sequence

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.fm_ct.vit import ViT


class Encoder(ViT):
    def __init__(
        self,
        weights_path: str,
        in_chans: int,
        img_size: Sequence[int] | int,
        patch_size: Sequence[int] | int,
        **kwargs,
    ):
        super().__init__(
            in_chans=in_chans, img_size=img_size, patch_size=patch_size, **kwargs
        )
        self.weights_path = weights_path
        self._load_weights()
        self._freeze_weights()
        self.eval()

    def _load_weights(self):
        loaded_state_dict = torch.load(
            self.weights_path, map_location=torch.device("cpu"), weights_only=True
        )
        loaded_state_dict = {
            k.replace("module.", "").replace("backbone.", ""): v
            for k, v in loaded_state_dict.items()
        }
        self.load_state_dict(loaded_state_dict, strict=False)

    def _freeze_weights(self):
        for param in self.parameters():
            param.requires_grad = False


class Projector(nn.Module):
    def __init__(self, input_channels, inner_channels, out_channels, dropout=0.0):
        super().__init__()
        self.pre_norm = nn.LayerNorm(input_channels)

        self.proj = nn.Sequential(
            nn.Linear(input_channels, inner_channels),
            nn.GELU(),
            nn.Linear(inner_channels, out_channels),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.pre_norm(x)
        return self.dropout(self.proj(x))


class Decoder(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.gradient_checkpointing_enable()

    def forward(self, input_embeds, attention_mask=None, **kwargs):
        return self.model(
            inputs_embeds=input_embeds, attention_mask=attention_mask, **kwargs
        )
