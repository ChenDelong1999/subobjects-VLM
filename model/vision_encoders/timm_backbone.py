import torch
import torch.nn as nn
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

class TimmBackbone(nn.Module):
    def __init__(self, model_name, image_resolution=224):
        super(TimmBackbone, self).__init__()

        if '/' in model_name:
            model_name, stage = model_name.split('/')
            self.stage = int(stage)
        else:
            self.stage = -1
        
        self.model = timm.create_model(model_name, features_only=True, pretrained=True)
        self.model.eval()
        
        self.config = resolve_data_config({}, model=self.model)
        self.config['input_size'] = (3, image_resolution, image_resolution)
        
        self.transform = create_transform(**self.config)
        self.feature_channels = self.model.feature_info[self.stage]['num_chs']
    
    @torch.no_grad()
    def forward(self, images):
        inputs = torch.stack(
            [self.transform(img) for img in images]
            ).to(self.device).to(self.dtype)
        
        return self.model(inputs)[self.stage]

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype
    
    @property
    def device(self):
        return next(self.model.parameters()).device