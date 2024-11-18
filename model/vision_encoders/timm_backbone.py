import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import re

class TimmBackbone(nn.Module):
    def __init__(self, model_name, image_resolution=224):
        super().__init__()

        # Parse model name and stages
        if '/' in model_name:
            base_model, stages_str = model_name.split('/')
        else:
            base_model, stages_str = model_name, None

        self.model = timm.create_model(base_model, features_only=True, pretrained=True)

        # Determine stages
        if not stages_str or stages_str == 'all':
            self.stages = list(range(len(self.model.feature_info.channels())))
        else:
            stages = re.findall(r'\d+', stages_str)
            self.stages = sorted(map(int, stages))

        # Adjust image resolution for specific models
        if model_name == 'vit_large_patch14_clip_336.openai':
            image_resolution = 336

        self.config = resolve_data_config({}, model=self.model)
        self.config['input_size'] = (3, image_resolution, image_resolution)
        self.transform = create_transform(**self.config)
        self.feature_channels = sum(self.model.feature_info[stage]['num_chs'] for stage in self.stages)
    
    @torch.no_grad()
    def forward(self, images):
        inputs = torch.stack([self.transform(img) for img in images]).to(self.device, self.dtype)
        features = self.model(inputs)

        if len(self.stages) == 1:
            return features[self.stages[0]]
        else:
            # Normalize and upsample features
            normalized_features = [F.normalize(features[stage], dim=1) for stage in self.stages]
            target_resolution = features[self.stages[0]].shape[-1]
            upsampled_features = [
                F.interpolate(feature, size=target_resolution, mode='bilinear') 
                for feature in normalized_features
            ]
            return torch.cat(upsampled_features, dim=1)
    
    @property
    def dtype(self):
        return next(self.model.parameters()).dtype
    
    @property
    def device(self):
        return next(self.model.parameters()).device