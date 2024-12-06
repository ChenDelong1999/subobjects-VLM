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
            return felzenszwalb(image)
        elif self.name == 'slic':
            return slic(image, start_label=0, n_segments=self.max_tokens)
        elif self.name == 'quickshift':
            return quickshift(image)

    def _create_masks_from_segments(self, segments):
        masks = []
        unique_labels = np.unique(segments)
        for label in unique_labels:
            mask = segments == label
            masks.append(mask)
        masks = np.array(masks)
        if len(masks) > self.max_tokens:
            masks = self._merge_small_masks(masks)
        return masks[:self.max_tokens]

    def _merge_small_masks(self, masks):
        merged_mask = np.any(masks[self.max_tokens - 1:], axis=0)
        masks[self.max_tokens - 1] = merged_mask
        return masks[:self.max_tokens]