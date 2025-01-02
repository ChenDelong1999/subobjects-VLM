import os
import sys
import numpy as np
import torch
from PIL import Image

from .sam import sam_post_processing  # Assuming sam_post_processing is defined in sam.py within the same package

import sys
sys.path.append(f'/private/home/delong/workspace/subobjects-VLM/visual_tokenizer/efficientvit')
# Import necessary modules from EfficientViT
from efficientvit.sam_model_zoo import create_sam_model
from demo.sam.helpers.auto_mask_generator import DemoEfficientViTSamAutomaticMaskGenerator
from demo.sam.helpers.utils import (
    BOX_NMS_THRESH,
    POINTS_PER_BATCH,
    PRED_IOU_THRESH,
    STABILITY_SCORE_THRESH
)

class EfficientViTTokenizer:
    def __init__(
        self,
        model_variant,
        image_resolution,
        max_tokens,
        device="cuda",
        weights_cache_dir="/private/home/delong/workspace/subobjects-VLM/visual_tokenizer/weights_cache",
        mask_generator_kwargs=None,
        **kwargs
    ):
        self.image_resolution = image_resolution
        self.max_tokens = max_tokens
        self.device = device

        # Ensure weights_cache_dir is in sys.path for imports
        efficientvit_path = os.path.join(weights_cache_dir, 'efficientvit')
        sys.path.append(efficientvit_path)


        # Create the EfficientViT-SAM model
        weight_path = os.path.join(weights_cache_dir, f"efficientvit-sam/{model_variant}.pt")
        self.efficientvit_sam = create_sam_model(
            model_variant, 
            pretrained=True, 
            weight_url=weight_path
        ).to(self.device).eval()

        # Initialize the mask generator with default or provided parameters
        if mask_generator_kwargs is None:
            mask_generator_kwargs = {}
        self.effvit_mask_gen = DemoEfficientViTSamAutomaticMaskGenerator(self.efficientvit_sam)
        self.effvit_mask_gen.set_box_nms_thresh(mask_generator_kwargs.get('box_nms_thresh', BOX_NMS_THRESH))
        self.effvit_mask_gen.set_points_per_batch(mask_generator_kwargs.get('points_per_batch', POINTS_PER_BATCH))
        self.effvit_mask_gen.set_pred_iou_thresh(mask_generator_kwargs.get('pred_iou_thresh', PRED_IOU_THRESH))
        self.effvit_mask_gen.set_stability_score_thresh(mask_generator_kwargs.get('stability_score_thresh', STABILITY_SCORE_THRESH))

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
            anns = self.effvit_mask_gen.generate(np.array(img))
            print(anns)
            if len(anns) > 0:
                masks = np.array([ann['segmentation'] for ann in anns])
                masks = sam_post_processing(masks)
            else:
                # If no masks are found, create an empty mask
                masks = np.zeros((1, self.image_resolution, self.image_resolution), dtype=bool)
            # Limit the number of masks to max_tokens
            batch_masks[i, :min(len(masks), self.max_tokens)] = masks[:self.max_tokens]
        return torch.tensor(batch_masks)