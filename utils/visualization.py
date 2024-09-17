
import cv2
from PIL import Image
import numpy as np
import tqdm


def visualize_masks(image, masks, mask_erotion=0, show_progress_bar=False):
    if type(image) == Image.Image:
        image = np.array(image)
    canvas = np.ones_like(image) * 255
    masks = tqdm.tqdm(masks) if show_progress_bar else masks
    for mask in masks:
        if mask_erotion>0:
            kernel = np.ones((mask_erotion, mask_erotion), np.uint8)
            mask = cv2.erode(mask.astype(np.uint8), kernel)
        mask_indices = np.where(mask)
        color = np.mean(image[mask_indices], axis=0)
        canvas[mask_indices] = color
    return canvas
