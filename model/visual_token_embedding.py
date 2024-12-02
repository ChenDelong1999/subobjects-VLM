import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch import nn
import torchvision.ops as ops

from .vision_encoders.rgb_pixel import RGBPixel
from .vision_encoders.diffusers_vae import DiffusersVAE
from .vision_encoders.hf_autobackbone import HFAutoBackbone
from .vision_encoders.timm_backbone import TimmBackbone

vision_encoder_registry = {
    'hf_autobacbone': HFAutoBackbone,
    'diffusers_vae': DiffusersVAE,
    'rgb_pixel': RGBPixel,
    'timm_backbone': TimmBackbone
}

class VisualTokenEmbedding(torch.nn.Module):

    def __init__(self, config):
        super(VisualTokenEmbedding, self).__init__()

        self.config = config
        if config.vision_encoder_type in vision_encoder_registry:
            self.vision_encoder = vision_encoder_registry[config.vision_encoder_type](
                model_name=config.vision_encoder_name, 
                image_resolution=config.image_resolution
                ).eval()
        else:
            raise NotImplementedError

        self.output_resolution = config.output_resolution


    @property
    def dtype(self):
        return self.vision_encoder.dtype
    

    @property
    def device(self):
        return self.vision_encoder.device
    
    @torch.no_grad()
    def forward(self, images, batch_masks):
        """
        Forward pass of the visual token embedding model.
        Args:
            images (list): A list of PIL images.
            batch_masks (np.ndarray): A numpy array of shape (N, M, H, W) containing binary masks.

        Returns:
            roi_boxes  (torch.Tensor): A tensor of shape (N, M, 4) containing the bounding boxes of each mask.
            roi_masks  (torch.Tensor): A tensor of shape (N, M, token_roi_resolution, token_roi_resolution) containing the cropped masks.
            embeddings (torch.Tensor): A tensor of shape (N, M, channels * token_roi_resolution * token_roi_resolution) containing the visual token embeddings.
        """
        with torch.no_grad():
            batch_features = self.vision_encoder(images)

        # upsample batch_features to output_resolution: 
        # N, C, H, W -> N, C, output_resolution, output_resolution
        batch_features = F.interpolate(
            batch_features, 
            size=(self.output_resolution, self.output_resolution),
            mode='bilinear'
        )

        batch_masks = batch_masks.to(batch_features.device).to(batch_features.dtype)

        roi_boxes, roi_masks, embeddings = self.mask_roi_pooling(batch_features, batch_masks)
        return roi_boxes, roi_masks, embeddings
    

    def mask_roi_pooling(self, batch_features, batch_masks):

        N, C, _mask_resolution, _mask_resolution = batch_features.shape
        _N, M, mask_resolution, _mask_resolution = batch_masks.shape
        dtype = batch_features.dtype

        # Get ROI boxes for each mask
        roi_boxes = self.get_roi_boxes_from_masks(batch_masks)

        # Perform ROIAlign for features
        roi_features = ops.roi_align(
            batch_features.float(), 
            roi_boxes, 
            output_size=(self.config.token_roi_resolution, self.config.token_roi_resolution),
            sampling_ratio=1
            ).view(N, M, C, self.config.token_roi_resolution, self.config.token_roi_resolution)

        # Perform ROIAlign for masks
        roi_masks = self.crop_roi_masks(
            batch_masks,
            roi_boxes,
            self.config.token_roi_resolution
        ).to(roi_features.device, dtype=roi_features.dtype) 
        # roi_masks Shape: (N, M, 1, token_roi_resolution, token_roi_resolution)

        # Apply mask to the features, and average pool
        roi_features = roi_features * roi_masks
        mask_sum = roi_masks.sum(dim=(-2, -1)).clamp(min=1e-6)  # Shape: (N, M, C)
        feature_sum = roi_features.sum(dim=(-2, -1))  # Shape: (N, M, C)
        embeddings = feature_sum / mask_sum  # Shape: (N, M, C)

        return torch.stack(roi_boxes) / mask_resolution, roi_masks[:, :, 0], embeddings.to(dtype)
    

    def get_roi_boxes_from_masks(self, batch_masks):
        N, M, H, W = batch_masks.shape
        
        y_coords = torch.arange(H, device=batch_masks.device).view(1, 1, H, 1).expand(N, M, H, W)
        x_coords = torch.arange(W, device=batch_masks.device).view(1, 1, 1, W).expand(N, M, H, W)
        
        mask = batch_masks > 0
        
        max_int = torch.iinfo(torch.int64).max
        min_int = torch.iinfo(torch.int64).min
        
        y_min = torch.where(mask, y_coords, torch.full_like(y_coords, max_int)).view(N, M, -1).min(dim=-1).values
        y_max = torch.where(mask, y_coords, torch.full_like(y_coords, min_int)).view(N, M, -1).max(dim=-1).values
        x_min = torch.where(mask, x_coords, torch.full_like(x_coords, max_int)).view(N, M, -1).min(dim=-1).values
        x_max = torch.where(mask, x_coords, torch.full_like(x_coords, min_int)).view(N, M, -1).max(dim=-1).values
        
        # Handle empty masks
        mask_sums = batch_masks.view(N, M, -1).sum(dim=-1)
        empty_masks = (mask_sums == 0)
        
        # Expand bounding boxes by 1 pixel and clip to image boundaries
        x_min = torch.clamp(x_min, min=0)
        y_min = torch.clamp(y_min, min=0)
        x_max = torch.clamp(x_max + 1, max=W-1)
        y_max = torch.clamp(y_max + 1, max=H-1)
        
        # Combine into bounding boxes
        roi_boxes = torch.stack([x_min, y_min, x_max, y_max], dim=-1)
        
        # Set empty mask boxes to [0, 0, 0, 0]
        roi_boxes[empty_masks] = 0
        
        return [box.float() for box in roi_boxes]
    
    
    def crop_roi_masks(self, batch_masks, roi_boxes, token_roi_resolution):
        N, M, H, W = batch_masks.shape
        device = batch_masks.device
        dtype = batch_masks.dtype

        # Flatten the batch and mask dimensions
        batch_masks_flat = batch_masks.reshape(N * M, H, W).unsqueeze(1)  # Shape: (N*M, 1, H, W)
        
        # Prepare the boxes tensor with correct batch indices
        # roi_boxes is a list of length N, each with shape (M, 4)
        # Stack roi_boxes into a single tensor of shape (N*M, 4)
        roi_boxes_tensor = torch.cat(roi_boxes, dim=0).to(device=device, dtype=torch.long)  # Shape: (N*M, 4)
        batch_indices = torch.arange(N*M, device=device).unsqueeze(1).type(dtype)
        boxes = torch.cat([batch_indices, roi_boxes_tensor], dim=1)  # Shape: (N*M, 5)

        # Perform roi_align on the masks
        cropped_masks = ops.roi_align(
            batch_masks_flat.float(),  # Ensure the masks are in float
            boxes.float(),
            output_size=token_roi_resolution,
            spatial_scale=1.0,          # Masks are in the same scale
            sampling_ratio=0,
            aligned=True
        )  # Output shape: (N*M, C, token_roi_resolution, token_roi_resolution)
        cropped_masks = cropped_masks.reshape(N, M, 1, token_roi_resolution, token_roi_resolution) 

        return cropped_masks > 0