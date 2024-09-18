import torch
import torchvision

class RGBPixel():
    def __init__(self, model_name=None, image_resolution=224):
        self.preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Resize((image_resolution, image_resolution)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.feature_channels = 3
        # self.feature_resolution = image_resolution

        self.dtype = torch.float32
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def __call__(self, images):
        pixel_values = [self.preprocess(image) for image in images]
        pixel_values = torch.stack(pixel_values).to(self.device).to(self.dtype)
        return pixel_values
    
    def eval(self):
        return self