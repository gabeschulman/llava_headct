from typing import Optional, Sequence

from src.components import Encoder, Projector, Decoder
import torch
from torch import nn


class LLaVAHeadCT(nn.Module):
    def __init__(
        self,
        vision_encoder_weights: str,
        vision_encoder_in_chans: int,
        vision_encoder_img_size: Sequence[int] | int,
        vision_encoder_patch_size: Sequence[int] | int,
        projector_input_channels: int,
        projector_inner_channels: int,
        projector_out_channels: int,
        decoder_model_name: str,
        projector_dropout: float = 0.0,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
    ):
        super().__init__()
        self.encoder = Encoder(
            weights_path=vision_encoder_weights,
            in_chans=vision_encoder_in_chans,
            img_size=vision_encoder_img_size,
            patch_size=vision_encoder_patch_size,
        )
        self.projector = Projector(
            input_channels=projector_input_channels,
            inner_channels=projector_inner_channels,
            out_channels=projector_out_channels,
            dropout=projector_dropout,
        )
        self.decoder = Decoder(model_name=decoder_model_name)

    def forward(
        self,
        image,
        text: Optional[str] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        image_features, _ = self.encoder(image)
        batch_size, num_tokens, feature_dim = image_features.shape
        image_features_flat = image_features.view(-1, feature_dim)

        projected_features_flat = self.projector(image_features_flat)
        projected_image_features = projected_features_flat.view(
            batch_size, num_tokens, -1
        )

        if text is not None or input_ids is not None:
            if input_ids is not None:
                text_input_ids = input_ids
                text_attention_mask = (
                    attention_mask
                    if attention_mask is not None
                    else torch.ones_like(input_ids)
                )
            else:
                text_tokens = self.decoder.tokenizer(
                    text, return_tensors="pt", padding=True, truncation=True
                )
                text_input_ids = text_tokens["input_ids"].to(image.device)
                text_attention_mask = text_tokens["attention_mask"].to(image.device)

            text_embeds = self.decoder.model.get_input_embeddings()(text_input_ids)

            combined_embeds = torch.cat([projected_image_features, text_embeds], dim=1)

            img_mask = torch.ones(
                projected_image_features.shape[:2],
                device=image.device,
                dtype=text_attention_mask.dtype,
            )
            combined_attention_mask = torch.cat([img_mask, text_attention_mask], dim=1)
        else:
            combined_embeds = projected_image_features
            combined_attention_mask = torch.ones(
                combined_embeds.shape[:2], device=image.device
            )

        outputs = self.decoder(
            input_embeds=combined_embeds, attention_mask=combined_attention_mask
        )
        return outputs
