from typing import Optional, Sequence

from src.components import Encoder, Projector, Decoder
import torch
from torch import nn


class LLaVAHeadCT(nn.Module):
    def __init__(
        self,
        vision_encoder_in_chans: int,
        vision_encoder_img_size: Sequence[int] | int,
        vision_encoder_patch_size: Sequence[int] | int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        decoder_model_name: str,
        vision_encoder_weights: Optional[str] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        state_dict_path: Optional[str] = None,
        fine_tune_encoder: bool = False,
        fine_tune_encoder_blocks: int = 3,
    ):
        super().__init__()
        use_pretrained_encoder_weights: bool = (
            True if state_dict_path is None else False
        )

        if state_dict_path is not None and vision_encoder_weights is not None:
            raise ValueError(
                "Cannot specify both state_dict_path and vision_encoder_weights."
            )

        self.encoder = Encoder(
            weights_path=vision_encoder_weights,
            in_chans=vision_encoder_in_chans,
            img_size=vision_encoder_img_size,
            patch_size=vision_encoder_patch_size,
            fine_tune=fine_tune_encoder,
            fine_tune_blocks=fine_tune_encoder_blocks,
            use_pretrained_weights=use_pretrained_encoder_weights,
        )

        self.projector = Projector(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )

        self.decoder = Decoder(model_name=decoder_model_name)

        if state_dict_path is not None:
            self.state_dict_path = state_dict_path
            self.load_pretrained_weights(strict=False)

    def load_pretrained_weights(self, strict: bool = True) -> None:
        """Load weights from a checkpoint file."""
        checkpoint = torch.load(self.state_dict_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            train_loss = checkpoint.get("train_loss", "N/A")
            if isinstance(train_loss, float):
                print(f"  Train loss: {train_loss:.4f}")
            val_loss = checkpoint.get("val_loss", "N/A")
            if isinstance(val_loss, float):
                print(f"  Val loss: {val_loss:.4f}")
        else:
            state_dict = checkpoint

        super().load_state_dict(state_dict, strict=strict)

    def get_text_embeddings(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Get text embeddings from the decoder's embedding layer."""
        return self.decoder.model.get_input_embeddings()(input_ids)

    def get_image_embeddings(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """Get image embeddings from the encoder and projector."""
        full_image_features, _ = self.encoder(image)
        cls_token = full_image_features[:, 0:1, :]  # Use only the CLS token
        spatial_tokens = nn.functional.adaptive_avg_pool1d(
            full_image_features[:, 1:, :].transpose(1, 2),
            output_size=128,  # 128 spatial tokens
        ).transpose(1, 2)

        image_features = torch.cat([cls_token, spatial_tokens], dim=1)

        batch_size, num_tokens, feature_dim = image_features.shape
        image_features_flat = image_features.view(-1, feature_dim)
        projected_features_flat = self.projector(image_features_flat)
        projected_image_features = projected_features_flat.view(
            batch_size, num_tokens, -1
        )
        return projected_image_features

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        image_embeddings: Optional[torch.Tensor] = None,
        text_embeddings: Optional[torch.Tensor] = None,
        text: Optional[str] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        if image_embeddings is not None:
            projected_image_features = image_embeddings
            image_device = projected_image_features.device
        elif image is not None:
            projected_image_features = self.get_image_embeddings(image)
            image_device = image.device
        else:
            raise ValueError("Either image or image_embeddings must be provided.")

        if text_embeddings is not None:
            combined_embeds = torch.cat(
                [projected_image_features, text_embeddings], dim=1
            )
            img_mask = torch.ones(
                projected_image_features.shape[:2],
                device=image_device,
                dtype=torch.float32,
            )
            combined_attention_mask = torch.cat(
                [
                    img_mask,
                    attention_mask
                    if attention_mask is not None
                    else torch.ones(text_embeddings.shape[:2], device=image_device),
                ],
                dim=1,
            )
        elif text is not None or input_ids is not None:
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
                text_input_ids = text_tokens["input_ids"].to(image_device)
                text_attention_mask = text_tokens["attention_mask"].to(image_device)

            text_embeds = self.get_text_embeddings(text_input_ids)

            combined_embeds = torch.cat([projected_image_features, text_embeds], dim=1)

            img_mask = torch.ones(
                projected_image_features.shape[:2],
                device=image_device,
                dtype=text_attention_mask.dtype,
            )
            combined_attention_mask = torch.cat([img_mask, text_attention_mask], dim=1)
        else:
            combined_embeds = projected_image_features
            combined_attention_mask = torch.ones(
                combined_embeds.shape[:2], device=image_device
            )

        outputs = self.decoder(
            input_embeds=combined_embeds, attention_mask=combined_attention_mask
        )
        return outputs

    def generate(
        self,
        image,
        prompt: Optional[str] = None,
        min_new_tokens: int = 1,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **generate_kwargs,
    ):
        """
        Generate text autoregressively for inference.

        Args:
            image: Input image tensor
            prompt: Optional text prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling (vs greedy)
            **generate_kwargs: Additional generation arguments

        Returns:
            Generated token IDs
        """

        self.eval()
        with torch.no_grad():
            # image_features, _ = self.encoder(image)
            # batch_size, num_tokens, feature_dim = image_features.shape
            # image_features_flat = image_features.view(-1, feature_dim)
            # projected_features_flat = self.projector(image_features_flat)
            # projected_image_features = projected_features_flat.view(
            #     batch_size, num_tokens, -1
            # )
            projected_image_features = self.get_image_embeddings(image)

            if prompt is not None:
                if isinstance(prompt, str):
                    prompt = [prompt] * image.shape[0]

                text_tokens = self.decoder.tokenizer(
                    prompt, return_tensors="pt", padding=True, truncation=True
                )
                text_input_ids = text_tokens["input_ids"].to(image.device)
                text_attention_mask = text_tokens["attention_mask"].to(image.device)
                text_embeds = self.decoder.model.get_input_embeddings()(text_input_ids)

                combined_embeds = torch.cat(
                    [projected_image_features, text_embeds], dim=1
                )
                img_mask = torch.ones(
                    projected_image_features.shape[:2],
                    device=image.device,
                    dtype=text_attention_mask.dtype,
                )
                combined_attention_mask = torch.cat(
                    [img_mask, text_attention_mask], dim=1
                )
            else:
                combined_embeds = projected_image_features
                combined_attention_mask = torch.ones(
                    combined_embeds.shape[:2], device=image.device
                )

            generated_ids = self.decoder.generate(
                input_embeds=combined_embeds,
                attention_mask=combined_attention_mask,
                min_new_tokens=min_new_tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.decoder.tokenizer.pad_token_id,
                eos_token_id=self.decoder.tokenizer.eos_token_id,
                **generate_kwargs,
            )

        return generated_ids
