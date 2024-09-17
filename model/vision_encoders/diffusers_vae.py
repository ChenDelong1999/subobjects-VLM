import torch
import torch.nn as nn
from diffusers.models import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor

class DiffusersVAE(nn.Module):
    def __init__(self, model_name="chendelong/stable-diffusion-3-medium-vae", image_resolution=224):
        super(DiffusersVAE, self).__init__()
        self.processor = VaeImageProcessor(do_normalize=False)
        self.image_resolution = image_resolution
        
        self.model = AutoencoderKL.from_pretrained(model_name).eval()
        self.model.decoder = None

        if 'stable-diffusion-3-medium-vae' in model_name:
            self.feature_channels = 16
            # self.feature_resolution = image_resolution // 8
    
    @torch.no_grad()
    def forward(self, images):
        inputs = self.processor.preprocess(images, height=self.image_resolution, width=self.image_resolution).to(self.model.device).to(self.model.dtype)
        feature_maps = self.model.encode(inputs).latent_dist.sample()
        return feature_maps

    @property
    def dtype(self):
        return self.model.dtype
    
    @property
    def device(self):
        return self.model.device
