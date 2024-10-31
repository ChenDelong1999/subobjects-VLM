import os
import numpy as np
import torch
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

class SAMTokenizer:

    def __init__(self, image_resolution, max_tokens, device="cuda", weights_cache_dir="", **kwargs):
        self.image_resolution = image_resolution
        self.max_tokens = max_tokens
        self.device = device

        # Load SAM model
        sam_checkpoint = os.path.join(weights_cache_dir, "sam/sam_vit_h_4b8939.pth")
        sam_model_type = "vit_h"
        self.sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint).to(device=self.device)
        
        # Initialize mask generator with SAM model
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,
        )

    def load_images(self, images):
        if not isinstance(images, list):
            images = [images]
        processed_images = []
        for img in images:
            if isinstance(img, str):
                img = Image.open(img)
            elif isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            img = img.convert("RGB").resize((self.image_resolution, self.image_resolution))
            processed_images.append(img)
        return processed_images

    @torch.inference_mode()
    def __call__(self, images):
        images = self.load_images(images)
        batch_masks = np.zeros(
            (len(images), self.max_tokens, self.image_resolution, self.image_resolution), 
            dtype=bool
        )

        for i, img in enumerate(images):
            anns = self.mask_generator.generate(np.array(img))
            masks = np.array([ann['segmentation'] for ann in anns]) * 1.0
            batch_masks[i, :len(masks)] = masks[:self.max_tokens]

        return torch.tensor(batch_masks)