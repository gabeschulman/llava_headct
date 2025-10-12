import torch
from models.vit import ViT

class HeadCTEncoder(ViT):
    def __init__(self, weights_path: str, **kwargs):
        super().__init__(**kwargs)
        self.weights_path = weights_path
        self.eval()

    def load_weights(self):
        loaded_state_dict = torch.load(self.weights_path, map_location=torch.device('cpu'), weights_only=True)
        loaded_state_dict = {k.replace("module.", "").replace("backbone.", ""): v for k, v in loaded_state_dict.items()}
        self.load_state_dict(loaded_state_dict, strict=False)
