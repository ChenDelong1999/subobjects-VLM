import numpy as np
import torch

class PatchTokenizer:
    def __init__(self, image_resolution, max_tokens, patch_per_side=16, order='raster', **kwargs):
        self.patch_per_side = patch_per_side
        self.order = order
        assert self.order in ['raster', 'random']

        self.image_resolution = image_resolution
        self.max_tokens = max_tokens
        
        self.patch_size = image_resolution // patch_per_side
        self.num_patches = patch_per_side * patch_per_side
        
        self.masks = self._prepare_masks()
        
    def _prepare_masks(self):
        masks = np.zeros((self.max_tokens, self.image_resolution, self.image_resolution), dtype=bool)
        
        for i in range(self.patch_per_side):
            for j in range(self.patch_per_side):
                patch_index = i * self.patch_per_side + j
                masks[patch_index, 
                      i*self.patch_size:(i+1)*self.patch_size, 
                      j*self.patch_size:(j+1)*self.patch_size] = True
        
        return masks[:self.max_tokens]
    
    def __call__(self, images):
        if not isinstance(images, list):
            images = [images]
        
        batch_size = len(images)
        if batch_size > 1:
            batch_masks = np.repeat(self.masks[np.newaxis, :, :, :], batch_size, axis=0)
        else:
            batch_masks = self.masks[np.newaxis, :, :, :]
        
        if self.order == 'random':
            for i in range(batch_size):
                np.random.shuffle(batch_masks[i])
        
        return torch.tensor(batch_masks.astype(bool))