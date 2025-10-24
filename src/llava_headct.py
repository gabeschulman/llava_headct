from src.components import Encoder, Projector, Decoder
import torch

class LLaVAHeadCT(torch.nn.Module):
    def __init__(self, 
                 vision_encoder_weights: str,
                 vision_encoder_kwargs: dict,
                 projector_in_channels: int,
                 projector_out_channels: int,
                 decoder_model_name: str):
        super().__init__()
        self.encoder = Encoder(weights_path=vision_encoder_weights, **vision_encoder_kwargs)
        self.projector = Projector(in_channels=projector_in_channels, out_channels=projector_out_channels)
        self.decoder = Decoder(model_name=decoder_model_name)

    def forward(self, images, input_ids, attention_mask=None):
        features = self.encoder(images)
        projected_features = self.projector(features) 
        outputs = self.decoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs
