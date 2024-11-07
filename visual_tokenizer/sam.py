import os
import numpy as np
import torch
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import cv2

class SAMTokenizer:

    def __init__(self, checkpoint, image_resolution, max_tokens, device="cuda", weights_cache_dir="", AMG_kwargs=None, **kwargs):
        self.image_resolution = image_resolution
        self.max_tokens = max_tokens
        self.device = device

        # Load SAM model
        sam_checkpoint = os.path.join(weights_cache_dir, f"sam/{checkpoint}.pth")
        if "vit_h" in checkpoint:
            sam_model_type = "vit_h"
        elif "vit_l" in checkpoint:
            sam_model_type = "vit_l"
        elif "vit_b" in checkpoint:
            sam_model_type = "vit_b"
        else:
            raise ValueError("Invalid SAM model checkpoint. Checkpoint should contain 'vit_h', 'vit_l', or 'vit_b'.")

        self.sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint).to(device=self.device)
        
        # Initialize mask generator with SAM model
        if AMG_kwargs is None:
            AMG_kwargs = {}
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            **AMG_kwargs,
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
            if len(anns) > 0:
                masks = np.array([ann['segmentation'] for ann in anns])
                masks = sam_post_processing(masks)
            else:
                masks = np.zeros((1, self.image_resolution, self.image_resolution), dtype=bool)

            batch_masks[i, :len(masks)] = masks[:self.max_tokens]

        return torch.tensor(batch_masks)

def sam_post_processing(masks):
    # Ensure masks are boolean
    masks = np.array(masks).astype(bool)
    # Find the background mask that is not covered by any instance mask
    background = ~np.any(masks, axis=0)
    # masks = np.concatenate([masks, background[None, ...]], axis=0)
    
    # generate masks by connected component labelling on the background mask
    background = background.astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(background)
    bg_masks = []
    for i in range(1, num_labels):
        bg_mask = labels == i
        bg_masks.append(bg_mask)
    masks = np.concatenate([masks, bg_masks], axis=0)
    
    # Sort masks by area
    areas = np.sum(masks, axis=(1, 2))
    sorted_indices = np.argsort(areas)[::-1]
    masks = masks[sorted_indices]
    return masks