from typing import Optional, Sequence

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.fm_ct.vit import ViT


class Encoder(ViT):
    def __init__(
        self,
        weights_path: Optional[str] = None,
        in_chans: int = 3,
        img_size: Sequence[int] | int = 96,
        patch_size: Sequence[int] | int = 12,
        use_pretrained_weights: bool = True,
        fine_tune: bool = False,
        fine_tune_blocks: int = 3,
        **kwargs,
    ):
        super().__init__(
            in_chans=in_chans, img_size=img_size, patch_size=patch_size, **kwargs
        )
        self.weights_path = weights_path
        if use_pretrained_weights:
            self._load_weights()
        self._freeze_weights()
        if fine_tune:
            self.unfreeze_last_n_blocks(n=fine_tune_blocks)
        else:
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

    def unfreeze_last_n_blocks(self, n: int):
        """Unfreeze the last n transformer blocks for fine-tuning."""
        total_blocks = len(self.blocks)
        for block in self.blocks[total_blocks - n :]:
            for param in block.parameters():
                param.requires_grad = True

        if hasattr(self, "norm"):
            for param in self.norm.parameters():
                param.requires_grad = True


class Projector(nn.Module):
    def __init__(
        self, input_dim=768, hidden_dim=2048, output_dim=896, num_layers=3, dropout=0.0
    ):
        super().__init__()

        layers = []
        layers.append(nn.LayerNorm(input_dim))

        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.GELU())

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.Dropout(dropout) if dropout > 0 else nn.Identity())
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.proj = nn.Sequential(*layers)

    def forward(self, x):
        return self.proj(x)


class Decoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        hidden_dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.gradient_checkpointing_enable()
        self.model.config.hidden_dropout = hidden_dropout
        self.model.config.attention_dropout = attention_dropout

    def forward(self, input_embeds, attention_mask=None, **kwargs):
        return self.model(
            inputs_embeds=input_embeds, attention_mask=attention_mask, **kwargs
        )

    def generate(self, input_embeds, attention_mask=None, **generate_kwargs):
        """
        Generate text for inference.

        Args:
            input_embeds: Input embeddings
            attention_mask: Attention mask
            **generate_kwargs: Additional arguments for generation (max_new_tokens, temperature, etc.)

        Returns:
            Generated token IDs
        """
        return self.model.generate(
            inputs_embeds=input_embeds, attention_mask=attention_mask, **generate_kwargs
        )
