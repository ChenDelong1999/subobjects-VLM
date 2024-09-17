import torch
import torch.nn as nn
import transformers


class HFAutoBackbone(nn.Module):
    def __init__(self, model_name, image_resolution=224):
        super(HFAutoBackbone, self).__init__()

        # check if the model_name contains two '/'
        if model_name.count('/') > 1:
            out_features = model_name.split('/')[-1]
            model_name = model_name.replace(f'/{out_features}', '')
            out_features = [out_features]
        else:
            out_features = None

        self.processor = transformers.AutoImageProcessor.from_pretrained(
            model_name, 
            do_center_crop=False)

        if 'height' in self.processor.size:
            self.processor.size['height'] = image_resolution
            self.processor.size['width'] = image_resolution
        elif 'shortest_edge' in self.processor.size:
            self.processor.size['shortest_edge'] = image_resolution
        else:
            raise ValueError("Cannot set image resolution")

        self.model = transformers.AutoBackbone.from_pretrained(
            model_name, out_features=out_features).eval()
        
        if hasattr(self.model.config, 'hidden_size'):
            self.feature_channels = self.model.config.hidden_size
        elif hasattr(self.model.config, 'hidden_sizes'):
            if out_features is None:
                stage_idx = -1
            else:
                stage_idx = self.model.config.stage_names.index(out_features[0]) - len(self.model.config.stage_names)
            self.feature_channels = self.model.config.hidden_sizes[stage_idx]
    
    @torch.no_grad()
    def forward(self, images):
        inputs = self.processor(images, return_tensors="pt").to(self.model.device).to(self.model.dtype)
        feature_maps = self.model(**inputs)['feature_maps'][0]
        return feature_maps

    
    @property
    def dtype(self):
        return self.model.dtype
    
    @property
    def device(self):
        return self.model.device
