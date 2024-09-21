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
            roi_masks  (torch.Tensor): A tensor of shape (N, M, token_resolution, token_resolution) containing the cropped masks.
            embeddings (torch.Tensor): A tensor of shape (N, M, channels * token_resolution * token_resolution) containing the visual token embeddings.
        """

        with torch.no_grad():
            batch_features = self.vision_encoder(batch_images)

        batch_masks = torch.tensor(batch_masks).float().to(batch_features.device).to(batch_features.dtype)

        roi_boxes, roi_masks, embeddings = self.mask_roi_pooling(batch_features, batch_masks)
        return roi_boxes, roi_masks, embeddings
    

    def dialate_masks(self, batch_masks, ratio=50):
        N, M, H_mask, W_mask = batch_masks.shape

        kernel_size = batch_masks.shape[-1] // ratio
        kernel_size = int(kernel_size) if int(kernel_size) % 2 == 1 else int(kernel_size) + 1

        if not hasattr(self, 'kernel'):
            self.kernel = torch.ones(
                (1, 1, kernel_size, kernel_size), 
                device=batch_masks.device, dtype=batch_masks.dtype
                )
            self.kernel.requires_grad = False

        padding = kernel_size // 2
        batch_masks_reshaped = batch_masks.view(N*M, 1, H_mask, W_mask)
    
        dilated_masks = F.conv2d(batch_masks_reshaped, self.kernel, padding=padding, stride=1)
        dilated_masks = dilated_masks.view(N, M, H_mask, W_mask)
        dilated_masks = (dilated_masks > 0).float()

        return dilated_masks
    

    def mask_roi_pooling(self, batch_features, batch_masks):
        N, C, H_feature, W_feature = batch_features.shape
        H_image, W_image = batch_masks.shape[-2:]
        M = batch_masks.shape[1]
        dtype = batch_features.dtype

        # Downsample masks to feature map resolution -> (N, M, H, W)
        # do dialation first, to avoid small mask disappear
        batch_masks = self.dialate_masks(batch_masks)

        # Get ROI boxes for each mask
        roi_boxes = self.get_roi_boxes_from_masks(batch_masks)
        scaling_factor = H_feature / H_image
        roi_boxes = [box * scaling_factor for box in roi_boxes]

        batch_masks = F.interpolate(
            batch_masks, size=(H_feature, W_feature),
            mode='nearest',
            # mode='bilinear', align_corners=False
        )

        # Perform ROIAlign for features
        roi_features = ops.roi_align(
            batch_features.float(), 
            roi_boxes, 
            output_size=(self.config.token_resolution, self.config.token_resolution),
            sampling_ratio=1
            ).view(N, M, C, self.config.token_resolution, self.config.token_resolution)

        
        # Perform ROIAlign for masks
        roi_masks = self.crop_roi_masks(
            batch_masks.unsqueeze(2).repeat(1, 1, C, 1, 1), 
            roi_boxes, self.config.token_resolution
        )

        # Apply mask to the features and flatten to embeddings
        roi_features = roi_features * roi_masks
        embeddings = roi_features.view(N, M, -1)

        return torch.stack(roi_boxes) / H_feature, roi_masks[:, :, 0], embeddings.to(dtype)
    

    def get_roi_boxes_from_masks(self, batch_masks):
        binary_masks = (batch_masks > 0).float()
        
        N, M, H, W = batch_masks.shape
        roi_boxes = []
        
        # Compute bounding boxes for each mask in the batch
        for n in range(N):
            sample_roi_boxes = []
            for m in range(M):
                mask = binary_masks[n, m]
                if mask.sum() > 0:
                    y_indices, x_indices = torch.where(mask)

                    # make the bounding box a bit larger
                    box = torch.tensor([
                        max(0, x_indices.min()-1), max(0, y_indices.min()-1),
                        min(W-1, x_indices.max()+1), min(H-1, y_indices.max()+1)
                    ], dtype=torch.int32, device=batch_masks.device)
                    
                    sample_roi_boxes.append([box[0], box[1], box[2], box[3]])
                else:
                    sample_roi_boxes.append([0, 0, 0, 0]) # padded visual tokens

            sample_roi_boxes= torch.tensor(sample_roi_boxes, dtype=torch.float32, device=batch_masks.device)
            roi_boxes.append(sample_roi_boxes)
        
        return roi_boxes
        

    def crop_roi_masks(self, batch_masks, roi_boxes, token_resolution):
        N, M, C, H, W = batch_masks.shape

        # Prepare the output tensor
        cropped_masks = torch.zeros(
            N, M, C, token_resolution, token_resolution, 
            device=batch_masks.device, dtype=batch_masks.dtype
            )
        
        for n in range(N):
            for m in range(M):
                x1, y1, x2, y2 = roi_boxes[n][m].long()
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W - 1, x2), min(H - 1, y2)

                cropped = batch_masks[n, m, :, y1:y2+1, x1:x2+1]
                resized = F.interpolate(
                    cropped.unsqueeze(0),  # Add batch dimension for interpolate
                    size=(token_resolution, token_resolution),
                    mode='nearest',
                    # mode='bilinear', align_corners=False
                ).squeeze(0)
                # if cropped mask is empty (too small), set it to ones
                cropped_masks[n, m] = resized if resized.sum() > 0 else torch.ones_like(resized)
        return cropped_masks
    