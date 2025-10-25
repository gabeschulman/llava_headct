from typing import Optional

from src.components import Encoder, Projector, Decoder
import torch
from torch import nn


class LLaVAHeadCT(nn.Module):
    def __init__(
        self,
        vision_encoder_weights: str,
        vision_encoder_kwargs: dict,
        projector_in_channels: int,
        projector_inner_channels: int,
        projector_out_channels: int,
        decoder_model_name: str,
    ):
        super().__init__()
        self.encoder = Encoder(
            weights_path=vision_encoder_weights, **vision_encoder_kwargs
        )
        self.projector = Projector(
            in_channels=projector_in_channels,
            inner_channels=projector_inner_channels,
            out_channels=projector_out_channels,
        )
        self.decoder = Decoder(model_name=decoder_model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, image, text: Optional[str] = None, attention_mask=None):
        image_features = self.encoder(image)
        projected_image_features = self.projector(image_features)

        if text is not None:
            text_tokens = self.decoder.tokenizer(
                text, return_tensors="pt", padding=True, truncation=True
            )
            text_input_ids = text_tokens["input_ids"].to(image.device)
            text_embeds = self.decoder.model.get_input_embeddings()(text_input_ids)

            combined_embeds = torch.cat([projected_image_features, text_embeds], dim=1)

            if attention_mask is None:
                img_mask = torch.ones(
                    projected_image_features.shape[:2], device=image.device
                )
                text_mask = text_tokens["attention_mask"].to(image.device)
                attention_mask = torch.cat([img_mask, text_mask], dim=1)
        else:
            combined_embeds = projected_image_features
            if attention_mask is None:
                attention_mask = torch.ones(
                    combined_embeds.shape[:2], device=image.device
                )

        outputs = self.decoder(
            inputs_embeds=combined_embeds, attention_mask=attention_mask
        )
        return outputs
