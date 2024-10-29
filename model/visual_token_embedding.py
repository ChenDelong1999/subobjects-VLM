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


    @property
    def dtype(self):
        return self.vision_encoder.dtype
    

    @property
    def device(self):
        return self.vision_encoder.device
    

    def forward(self, batch_images, batch_masks):
        """
        Forward pass of the visual token embedding model.
        Args:
            batch_images (list): A list of PIL images.
            batch_masks (np.ndarray): A numpy array of shape (N, M, H, W) containing binary masks.

        Returns:
            roi_boxes  (torch.Tensor): A tensor of shape (N, M, 4) containing the bounding boxes of each mask.
            roi_masks  (torch.Tensor): A tensor of shape (N, M, token_mask_resolution, token_mask_resolution) containing the cropped masks.
            embeddings (torch.Tensor): A tensor of shape (N, M, channels * token_roi_resolution * token_roi_resolution) containing the visual token embeddings.
        """
        with torch.no_grad():
            batch_features = self.vision_encoder(batch_images)

        batch_masks = batch_masks.to(batch_features.device).to(batch_features.dtype)

        roi_boxes, roi_masks, embeddings = self.mask_roi_pooling(batch_features, batch_masks)
        return roi_boxes, roi_masks, embeddings
    

    def dialate_masks(self, batch_masks, ratio=100):
        N, M, H_mask, W_mask = batch_masks.shape

        kernel_size = batch_masks.shape[-1] // ratio
        kernel_size = int(kernel_size) if int(kernel_size) % 2 == 1 else int(kernel_size) + 1

        if not hasattr(self, 'kernel'):
            radius = kernel_size // 2
            y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
            mask = x**2 + y**2 <= radius**2
            self.kernel = torch.tensor(mask, dtype=batch_masks.dtype, device=batch_masks.device).unsqueeze(0).unsqueeze(0)
            self.kernel.requires_grad = False

        padding = kernel_size // 2
        batch_masks_reshaped = batch_masks.view(N*M, 1, H_mask, W_mask)
    
        dilated_masks = F.conv2d(batch_masks_reshaped, self.kernel, padding=padding, stride=1)
        dilated_masks = dilated_masks.view(N, M, H_mask, W_mask)
        dilated_masks = (dilated_masks > 0).float()

        return dilated_masks
    

    def mask_roi_pooling(self, batch_features, batch_masks):

        N, C, H_feature, W_feature = batch_features.shape
        H_image, _ = batch_masks.shape[-2:]
        M = batch_masks.shape[1]
        dtype = batch_features.dtype

        # Get ROI boxes for each mask
        roi_boxes_image_scale = self.get_roi_boxes_from_masks(batch_masks)
        roi_boxes_feat_scale = [box / H_image * H_feature for box in roi_boxes_image_scale]

        # Perform ROIAlign for features
        roi_features = ops.roi_align(
            batch_features.float(), 
            roi_boxes_feat_scale, 
            output_size=(self.config.token_roi_resolution, self.config.token_roi_resolution),
            sampling_ratio=1
            ).view(N, M, C, self.config.token_roi_resolution, self.config.token_roi_resolution)

        # Downsample masks to feature map resolution -> (N, M, H, W)
        batch_masks_feat_scale = F.interpolate(
            batch_masks, size=(H_feature, W_feature),
            mode='nearest'
        )
        # Dilate masks
        # batch_masks_feat_scale = self.dialate_masks(batch_masks_feat_scale)

        # Perform ROIAlign for masks
        roi_masks = self.crop_roi_masks(
            batch_masks_feat_scale,
            roi_boxes_feat_scale,
            self.config.token_mask_resolution
        ).to(roi_features.device, dtype=roi_features.dtype)

        roi_masks_in_token_resolution = F.interpolate(
            roi_masks.squeeze(2),
            size=(self.config.token_roi_resolution, self.config.token_roi_resolution),
            mode='nearest'
        ).unsqueeze(2)

        # Apply mask to the features and flatten to embeddings
        roi_features = roi_features * roi_masks_in_token_resolution.repeat(1, 1, C, 1, 1)
        embeddings = roi_features.view(N, M, -1)

        return torch.stack(roi_boxes_image_scale) / H_image, roi_masks[:, :, 0], embeddings.to(dtype)
    

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
        x_min = torch.clamp(x_min - 1, min=0)
        y_min = torch.clamp(y_min - 1, min=0)
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
            boxes,
            output_size=token_roi_resolution,
            spatial_scale=1.0,          # Masks are in the same scale
            sampling_ratio=0,
            aligned=True
        )  # Output shape: (N*M, C, token_roi_resolution, token_roi_resolution)
        cropped_masks = cropped_masks.reshape(N, M, 1, token_roi_resolution, token_roi_resolution) 

        return cropped_masks > 0