# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from src.utils.patch_embedding import PatchEmbeddingBlock
from src.models.attentionblock import AttentionBlock

__all__ = ["ViT"]


class ViT(nn.Module):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    """

    def __init__(
        self,
        in_chans: int,
        img_size: Sequence[int] | int,
        patch_size: Sequence[int] | int,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        patch_embed: str = "conv",
        pos_embed: str = "learnable",
        classification: bool = False,
        num_classes: int = 2,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        num_register_tokens: int = 0,
        post_activation: str = "Tanh",
        qkv_bias: bool = False,
        lora: bool = False,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        """
        Args:
            in_channels (int): dimension of input channels.
            img_size (Union[Sequence[int], int]): dimension of input image.
            patch_size (Union[Sequence[int], int]): dimension of patch size.
            hidden_size (int, optional): dimension of hidden layer. Defaults to 768.
            mlp_dim (int, optional): dimension of feedforward layer. Defaults to 3072.
            num_layers (int, optional): number of transformer blocks. Defaults to 12.
            num_heads (int, optional): number of attention heads. Defaults to 12.
            patch_embed (str, optional): patch embedding layer type. Defaults to "conv".
            pos_embed (str, optional): position embedding type. Defaults to "learnable".
            classification (bool, optional): bool argument to determine if classification is used. Defaults to False.
            num_classes (int, optional): number of classes if classification is used. Defaults to 2.
            dropout_rate (float, optional): faction of the input units to drop. Defaults to 0.0.
            spatial_dims (int, optional): number of spatial dimensions. Defaults to 3.
            num_register_tokens (int, optional): number of register tokens. Defaults to 0.
            post_activation (str, optional): add a final acivation function to the classification head
                when `classification` is True. Default to "Tanh" for `nn.Tanh()`.
                Set to other values to remove this function.
            qkv_bias (bool, optional): apply bias to the qkv linear layer in self attention block. Defaults to False.
            lora (bool, optional): use LoRA (Learned Relative Attention) in self attention block. Defaults to False.
            norm_layer (nn.Module, optional): normalization layer. Defaults to nn.LayerNorm.

        Examples::

            # for single channel input with image size of (96,96,96), conv position embedding and segmentation backbone
            >>> net = ViT(in_channels=1, img_size=(96,96,96), patch_embed='conv', pos_embed='sincos')

            # for 3-channel with image size of (128,128,128), 24 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(128,128,128), patch_embed='conv', pos_embed='sincos', classification=True)

            # for 3-channel with image size of (224,224), 12 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(224,224), patch_embed='conv', pos_embed='sincos', classification=True,
            >>>           spatial_dims=2)

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.classification = classification
        
        # patch embedding
        self.patch_embedding = PatchEmbeddingBlock(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_chans,
            hidden_size=hidden_size,
            num_heads=num_heads,
            patch_embed=patch_embed,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        
        # transformer encoder
        self.blocks = nn.ModuleList(
            [
                AttentionBlock(hidden_size, mlp_dim, num_heads, dropout_rate, \
                    qkv_bias=qkv_bias, save_attn=False, lora=lora, \
                    norm_layer=norm_layer)
                for _ in range(num_layers)
            ]
        )
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.norm = norm_layer(hidden_size, eps=1e-6)
        
        # register tokens proposed by https://openreview.net/pdf?id=2dnO3LLiJ1
        self.num_register_tokens = num_register_tokens
        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, hidden_size)) if num_register_tokens else None
        )
        
        if self.classification:
            if post_activation == "Tanh":
                self.classification_head = nn.Sequential(nn.Linear(hidden_size, num_classes), nn.Tanh())
            else:
                self.classification_head = nn.Linear(hidden_size, num_classes)  # type: ignore

    def init_weights(self):
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)

    def forward(self, x):
        x = self.patch_embedding(x)
            
        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        
        # add register tokens
        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )
        
        hidden_states_out = []
        # apply Transformer blocks
        residual = None
        for blk in self.blocks:
            x, residual = blk(x, residual)
            hidden_states_out.append(x)
            
        x = self.norm(x)
        if hasattr(self, "classification_head"):
            x = self.classification_head(x[:, 0])
            
        return x, hidden_states_out
    