import numpy as np
import cv2
import torch
from torch import nn

from .vision_encoders.rgb_pixel import RGBPixel
from .vision_encoders.diffusers_vae import DiffusersVAE
from .vision_encoders.hf_autobackbone import HFAutoBackbone
from .vision_encoders.timm_backbone import TimmBackbone


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
        
        self.embedding_dim = self.vision_encoder.feature_channels * self.config.token_resolution ** 2
    
    @property
    def dtype(self):
        return self.vision_encoder.dtype
    
    @property
    def device(self):
        return self.vision_encoder.device
    