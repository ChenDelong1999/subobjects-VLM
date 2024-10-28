import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from skimage.segmentation import watershed

from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor

class DirectSAMTokenizer:

    def __init__(
        self,
        checkpoint,
        threshold,
        image_resolution,
        max_tokens,
        device="cuda",
        **kwargs
    ):
        self.ckpt = checkpoint
        self.threshold = threshold
        self.image_resolution = image_resolution
        self.max_tokens = max_tokens
        self.device = device

        self.image_processor = AutoImageProcessor.from_pretrained("chendelong/DirectSAM-1800px-0424", reduce_labels=True)
        self.image_processor.size['height'] = image_resolution
        self.image_processor.size['width'] = image_resolution 

        self.model = AutoModelForSemanticSegmentation.from_pretrained(checkpoint)
        self.model = self.model.to(self.device).half().eval()

        print(f'DirectSAM initialized on {device}')


    def load_images(self, images):

        images = images.copy()
        if type(images) != list:
            images = [images]

        for i in range(len(images)):
            if type(images[i]) == str:
                images[i] = Image.open(images[i])
            elif type(images[i]) == np.ndarray:
                images[i] = Image.fromarray(images[i])
            images[i] = images[i].convert("RGB").resize((self.image_resolution, self.image_resolution))
        return images


    @torch.inference_mode()
    def __call__(self, images): 

        images = self.load_images(images)
        pixel_values = self.image_processor(images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device).to(self.model.dtype)

        logits = self.model(pixel_values=pixel_values).logits.float().cpu()
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=(self.image_resolution, self.image_resolution),
            mode="bicubic",
            align_corners=False,
        )

        probabilities = torch.sigmoid(upsampled_logits)[:,0].detach().numpy()
        boundaries = (probabilities > self.threshold)

        batch_masks = np.zeros(
            (len(images), self.max_tokens, self.image_resolution, self.image_resolution), 
            dtype=bool
            ) # N, M, H, W
        
        for i, boundary in enumerate(boundaries):
            masks = self.boundary_to_mask(boundary, probabilities[i])
            batch_masks[i, :len(masks)] = masks

        batch_masks = torch.tensor(batch_masks)
        return batch_masks
    
    def sort_masks(self, batch_masks):
        #  sort by area
        sums = torch.sum(batch_masks, dim=(2, 3))
        sorted_indices = torch.argsort(sums, dim=1, descending=True)
        for i in range(batch_masks.shape[0]):
            batch_masks[i] = batch_masks[i, sorted_indices[i]]
        return batch_masks


    def boundary_to_mask(self, boundary, probability):
        """
        Converts a boundary image to a binary mask.
        Input:      A numpy array (H, W) representing the boundary image, True for boundary pixels and False for non-boundary pixels.
        Returns:    A numpy array of binary masks (n_masks, H, W), where each mask corresponds to a connected component in the boundary image.
        """
        num_objects, labels = cv2.connectedComponents(
            (~boundary).astype(np.uint8), 
            connectivity=4, 
            )
        labels = watershed(probability, markers=labels)

        masks = np.zeros((num_objects-1, *boundary.shape), dtype=bool)
        for i in range(1, num_objects):
            masks[i-1] = labels == i

        # sort by area
        areas = np.sum(masks, axis=(1, 2))
        sorted_indices = np.argsort(areas)[::-1]
        masks = masks[sorted_indices]

        # if there are more than max_tokens masks, merge the small masks
        if num_objects > self.max_tokens + 1:
            remaining_masks = masks[self.max_tokens-1:]
            remaining_masks = np.any(remaining_masks, axis=0)
            masks[self.max_tokens-1] = remaining_masks
        return masks[:self.max_tokens]
    