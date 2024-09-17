import numpy as np
import cv2
import torch
from torch import nn

from .vision_encoders.rgb_pixel import RGBPixel
from .vision_encoders.diffusers_vae import DiffusersVAE
from .vision_encoders.hf_autobackbone import HFAutoBackbone
from .vision_encoders.timm_backbone import TimmBackbone

from timm.layers.attention_pool2d import RotAttentionPool2d, apply_rot_embed
import torch.nn as nn


class VisualTokenEmbedding(torch.nn.Module):

    def __init__(self, config):
        super(VisualTokenEmbedding, self).__init__()

        self.config = config

        type_to_class = {
            'hf_autobacbone': HFAutoBackbone,
            'diffusers_vae': DiffusersVAE,
            'rgb_pixel': RGBPixel,
            'timm_backbone': TimmBackbone
        }

        if config.vision_encoder_type in type_to_class:
            self.vision_encoder = type_to_class[config.vision_encoder_type](
                model_name=config.vision_encoder_name, 
                image_resolution=config.image_resolution
                ).eval()
        else:
            raise NotImplementedError
        
        self.attn_pool = RotAttentionPool2dWithNorm(
            in_features=self.vision_encoder.feature_channels,
            out_features=config.embedding_dim,
            num_heads=config.num_heads
        )


    def forward(self, batch_images, batch_masks):

        with torch.no_grad():
            batch_features = self.vision_encoder(batch_images)

        feat_res = batch_features.shape[-1]
        num_masks = batch_masks.shape[1]

        # batch_masks: (N, num_masks, image_resolution, image_resolution)
        # downsample masks to feature resolution
        batch_masks = nn.functional.interpolate(
            torch.tensor(batch_masks).float(),  
            size=(feat_res, feat_res),
            mode='bilinear',
            align_corners=False
        ).to(batch_features.device).to(batch_features.dtype)
        
        # batch_features:   (N, C,              feat_res, feat_res)
        # batch_masks:      (N, num_masks,      feat_res, feat_res)
        # Apply mask for each feature map
        # masked_features:  (N, num_masks, C,   feat_res, feat_res)
        masked_features = batch_features.unsqueeze(1) * batch_masks.unsqueeze(2)

        # (N, num_masks, C, feat_res, feat_res) -> (N * num_masks, C, feat_res, feat_res)
        masked_features = masked_features.view(-1, *masked_features.shape[2:])
        embeddings = self.attn_pool(masked_features) 
        
        # (N * num_masks, embedding_dim) -> (N, num_masks, embedding_dim)
        embeddings = embeddings.view(-1, num_masks, self.config.embedding_dim)
        return embeddings

    
    @property
    def dtype(self):
        return self.vision_encoder.dtype
    
    @property
    def device(self):
        return self.vision_encoder.device
    

class RotAttentionPool2dWithNorm(RotAttentionPool2d):

    # https://github.com/huggingface/pytorch-image-models/blob/ee5b1e8217134e9f016a0086b793c34abb721216/timm/layers/attention_pool2d.py#L22

    def forward(self, x, pre_logits: bool = False):

        # x: (bs, feature_channels, feat_res, feat_res)
        B, _, H, W = x.shape
        N = H * W
        x = x.flatten(2).transpose(1, 2)
        if self.cls_token is None:
            avg_pooled = x.mean(1, keepdim=True)
            # Need to normalize the avg_pooled query since mask size varies
            avg_pooled = nn.functional.normalize(avg_pooled, p=2, dim=1)
            x = torch.cat([avg_pooled, x], dim=1)
        else:
            x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        if self.qkv is None:
            q = self.q(x).reshape(B, N + 1, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k(x).reshape(B, N + 1, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v(x).reshape(B, N + 1, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            x = self.qkv(x).reshape(B, N + 1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = x.unbind(0)

        rse, rce = self.pos_embed.get_embed((H, W))
        q = torch.cat([q[:, :, :1, :], apply_rot_embed(q[:, :, 1:, :], rse, rce)], dim=2).type_as(v)
        k = torch.cat([k[:, :, :1, :], apply_rot_embed(k[:, :, 1:, :], rse, rce)], dim=2).type_as(v)

        if self.fused_attn:
            x = nn.functional.scaled_dot_product_attention(q, k, v)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, N + 1, -1)
        x = self.drop(x)
        if pre_logits:
            x = self._pool(x, H, W)
            return x
        x = self.proj(x)
        x = self._pool(x, H, W)
        return x

