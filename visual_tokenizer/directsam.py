import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn as nn

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

        self.pad = np.zeros((image_resolution, image_resolution), dtype=bool)

        self.image_processor = AutoImageProcessor.from_pretrained("chendelong/DirectSAM-1800px-0424", reduce_labels=True)
        self.image_processor.size['height'] = image_resolution
        self.image_processor.size['width'] = image_resolution 

        self.model = AutoModelForSemanticSegmentation.from_pretrained(checkpoint)
        self.model = self.model.to(self.device).half().eval()


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
            # align_corners=False,
        )
        probabilities = torch.sigmoid(upsampled_logits)[:,0].detach().numpy()
        boundaries = 1 - (probabilities < self.threshold).astype(np.uint8)
        
        def boundary_to_mask(boundary):
            """
            Converts a boundary image to a binary mask.

            Parameters:
            - boundary: A numpy array (H, W) representing the boundary image, True for boundary pixels and False for non-boundary pixels.

            Returns:
            - masks: A numpy array of binary masks (n_masks, H, W), where each mask corresponds to a connected component in the boundary image.
            """

            num_objects, labels = cv2.connectedComponents(
                (1-boundary).astype(np.uint8), 
                connectivity=4, 
                )

            masks = np.zeros((num_objects-1, *boundary.shape), dtype=bool)
            for i in range(1, num_objects):
                masks[i-1] = labels == i
            return masks
        
        batch_masks = []
        for boundary in boundaries:
            masks = boundary_to_mask(boundary)
            batch_masks.append(masks)

        # sort by area
        for i, masks in enumerate(batch_masks):
            sums = np.array([mask.sum() for mask in masks])
            sorted_indices = np.argsort(sums)[::-1]    
            masks = masks[sorted_indices][:self.max_tokens]

            if len(masks) < self.max_tokens:
                masks = np.concatenate([masks, [self.pad] * (self.max_tokens - len(masks))])

            batch_masks[i] = masks

        return np.array(batch_masks).astype(bool)


