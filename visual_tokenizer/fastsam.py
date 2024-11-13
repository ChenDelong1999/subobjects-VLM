import os
import numpy as np
import torch
from PIL import Image
from .sam import sam_post_processing

from .FastSAM.fastsam import FastSAM, FastSAMPrompt

class FastSAMTokenizer:

    def __init__(self, checkpoint, image_resolution, max_tokens, device="cuda", weights_cache_dir="", FastSAM_kwargs=None, **kwargs):


        self.image_resolution = image_resolution
        self.max_tokens = max_tokens
        self.device = device

        # Load FastSAM model
        fastsam_checkpoint = os.path.join(weights_cache_dir, f"fastsam/{checkpoint}.pt")
        self.fastsam = FastSAM(fastsam_checkpoint)
        self.fastsam.to(self.device)

        # Set default FastSAM parameters if not provided
        if FastSAM_kwargs is None:
            FastSAM_kwargs = {}
        # Ensure 'retina_masks' is True and 'imgsz' matches image_resolution
        FastSAM_kwargs.setdefault('retina_masks', True)
        FastSAM_kwargs.setdefault('imgsz', self.image_resolution)
        FastSAM_kwargs.setdefault('conf', 0.4)
        FastSAM_kwargs.setdefault('iou', 0.9)
        FastSAM_kwargs.setdefault('verbose', False)
        self.FastSAM_kwargs = FastSAM_kwargs

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
            everything_results = self.fastsam(
                img, 
                device=self.device, 
                **self.FastSAM_kwargs
            )
            prompt_process = FastSAMPrompt(img, everything_results, device=self.device)
            mask = prompt_process.everything_prompt()
            if isinstance(mask, list) or mask.numel() == 0:
                # No masks found, create an empty mask
                mask = np.zeros((1, self.image_resolution, self.image_resolution), dtype=bool)
            else:
                mask = mask.cpu().numpy()
            # Process masks
            masks = sam_post_processing(mask)
            # Limit the number of masks to max_tokens
            batch_masks[i, :min(len(masks), self.max_tokens)] = masks[:self.max_tokens]

        return torch.tensor(batch_masks)