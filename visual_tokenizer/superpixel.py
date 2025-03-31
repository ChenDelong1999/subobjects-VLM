# superpixel.py
import numpy as np
import torch
from skimage.segmentation import felzenszwalb, slic, quickshift
from skimage.util import img_as_float
from PIL import Image

class SuperpixelTokenizer:

    def __init__(self, name='slic', image_resolution=224, max_tokens=256, **kwargs):
        self.name = name
        self.image_resolution = image_resolution
        self.max_tokens = max_tokens
        assert self.name in ['felzenszwalb', 'slic', 'quickshift'], (
            f"'{self.name}' is not supported. Supported superpixel methods: ['felzenszwalb', 'slic', 'quickshift']"
        )
        self.kwargs = kwargs

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
            processed_images.append(img_as_float(np.array(img)))
        return processed_images

    def __call__(self, images):
        images = self.load_images(images)
        batch_masks = np.zeros(
            (len(images), self.max_tokens, self.image_resolution, self.image_resolution),
            dtype=bool
        )
        for i, image in enumerate(images):
            segments = self._segment_image(image)
            masks = self._create_masks_from_segments(segments)
            batch_masks[i, :len(masks)] = masks
        return torch.tensor(batch_masks)

    def _segment_image(self, image):
        if self.name == 'felzenszwalb':
            scale = self.kwargs.get('scale', 2000)
            min_size = int((image.shape[0] * 0.02)**2)
            return felzenszwalb(image, scale=scale, sigma=1, min_size=min_size)
        elif self.name == 'slic':
            return slic(image, start_label=0, n_segments=self.max_tokens)
        elif self.name == 'quickshift':
            return quickshift(image, kernel_size=50, max_dist=6, ratio=0.5)

    def _create_masks_from_segments(self, segments):
        masks = []
        unique_labels = np.unique(segments)
        
        # Create masks and calculate their areas
        masks_with_areas = []
        for label in unique_labels:
            mask = segments == label
            area = np.sum(mask)  # Calculate the area of the segment
            masks_with_areas.append((mask, area))
        
        # Sort masks by area (descending)
        if self.name != 'slic':
            masks_with_areas.sort(key=lambda x: x[1], reverse=True)
        
        # Extract sorted masks
        sorted_masks = [mask for mask, _ in masks_with_areas]
        masks = np.array(sorted_masks)
        
        # Ensure we don't exceed max_tokens
        if len(masks) > self.max_tokens:
            masks = self._merge_small_masks(masks)
        
        return masks[:self.max_tokens]

    def _merge_small_masks(self, masks):
        merged_mask = np.any(masks[self.max_tokens - 1:], axis=0)
        masks[self.max_tokens - 1] = merged_mask
        return masks[:self.max_tokens]