import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from headCT_foundation.models.vit import ViT
except ModuleNotFoundError:
    from models.vit import ViT

class Encoder(ViT):
    def __init__(self, weights_path: str, **kwargs):
        super().__init__(**kwargs)
        self.weights_path = weights_path
        self._load_weights()
        self.eval()

    def _load_weights(self):
        loaded_state_dict = torch.load(self.weights_path, map_location=torch.device('cpu'), weights_only=True)
        loaded_state_dict = {k.replace("module.", "").replace("backbone.", ""): v for k, v in loaded_state_dict.items()}
        self.load_state_dict(loaded_state_dict, strict=False)

class Projector(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(in_channels)

        self.proj = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.GELU(),
            nn.Linear(in_channels, out_channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return self.proj(x)


class Decoder(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)
