import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

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
            # align_corners=False,
        )

        probabilities = torch.sigmoid(upsampled_logits)[:,0].detach().numpy()
        boundaries = (probabilities < self.threshold)

        batch_masks = np.zeros(
            (len(images), self.max_tokens, self.image_resolution, self.image_resolution), 
            dtype=bool
            ) # N, M, H, W
        
        for i, boundary in enumerate(boundaries):
            masks = self.boundary_to_mask(boundary)
            batch_masks[i, :len(masks)] = masks

        batch_masks = torch.tensor(batch_masks)

        # # there are some pixels that are not covered by any mask
        # remainings = ~torch.any(batch_masks, dim=1)
        # remainings = self.erode_masks(remainings.unsqueeze(1)).squeeze(1)
        # n_effective_masks = torch.sum(torch.any(batch_masks, dim=(2, 3)), dim=1)
        # for i in range(batch_masks.shape[0]):
        #     batch_masks[i, n_effective_masks[i]-1] = remainings[i] | batch_masks[i, n_effective_masks[i]-1]

        # batch_masks = self.sort_masks(batch_masks)
        return batch_masks
    
    def sort_masks(self, batch_masks):
        #  sort by area
        sums = torch.sum(batch_masks, dim=(2, 3))
        sorted_indices = torch.argsort(sums, dim=1, descending=True)
        for i in range(batch_masks.shape[0]):
            batch_masks[i] = batch_masks[i, sorted_indices[i]]
        return batch_masks


    def boundary_to_mask(self, boundary):
        """
        Converts a boundary image to a binary mask.
        Input:      A numpy array (H, W) representing the boundary image, True for boundary pixels and False for non-boundary pixels.
        Returns:    A numpy array of binary masks (n_masks, H, W), where each mask corresponds to a connected component in the boundary image.
        """
        num_objects, labels = cv2.connectedComponents(
            boundary.astype(np.uint8), 
            connectivity=4, 
            )

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

            # erode the remaining masks to make it more compact
            remaining_masks = self.erode_masks_numpy(remaining_masks)
            masks[self.max_tokens-1] = remaining_masks
        return masks[:self.max_tokens]
    

    def erode_masks_numpy(self, mask, ratio=50):
        """
        Erode a 2D binary mask using a circular kernel.
        
        Args:
        mask (numpy.ndarray): A 2D binary numpy array representing the mask.
        ratio (int): The ratio to determine the kernel size. Default is 50.
        
        Returns:
        numpy.ndarray: The eroded mask as a 2D binary numpy array.
        """

        if not hasattr(self, 'erosion_kernel'):
            kernel_size = mask.shape[-1] // ratio
            kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

            # Create a circular kernel
            radius = kernel_size // 2
            y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
            self.kernel = x**2 + y**2 <= radius**2

        # Perform erosion using scipy.ndimage
        eroded_mask = ndimage.binary_erosion(mask, structure=self.kernel)

        return eroded_mask

    
    # def erode_masks(self, batch_masks, ratio=50):
    #     N, M, H_mask, W_mask = batch_masks.shape

    #     kernel_size = batch_masks.shape[-1] // ratio
    #     kernel_size = int(kernel_size) if int(kernel_size) % 2 == 1 else int(kernel_size) + 1

    #     if not hasattr(self, 'erosion_kernel'):
    #         radius = kernel_size // 2
    #         y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    #         mask = x**2 + y**2 <= radius**2
    #         self.erosion_kernel = torch.tensor(mask, dtype=torch.float32, device=batch_masks.device).unsqueeze(0).unsqueeze(0)
    #         self.erosion_kernel.requires_grad = False

    #     padding = kernel_size // 2
    #     batch_masks_reshaped = batch_masks.view(N*M, 1, H_mask, W_mask).float()

    #     # Use erosion by applying dilation to the inverse of the mask
    #     eroded_masks = 1 - F.conv2d(1 - batch_masks_reshaped, self.erosion_kernel, padding=padding, stride=1)
    #     eroded_masks = eroded_masks.view(N, M, H_mask, W_mask)

    #     return (eroded_masks > 0.5).bool().cpu()
    
