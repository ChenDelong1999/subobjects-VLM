
import cv2
from PIL import Image
import numpy as np
import tqdm
import matplotlib.pyplot as plt


def visualize_masks(image, masks, mask_erotion=0, show_progress_bar=False):
    if type(image) == Image.Image:
        image = np.array(image)
    canvas = np.ones_like(image) * 255

    # sort masks by area
    masks = sorted(masks, key=lambda x: np.sum(x), reverse=True)

    masks = tqdm.tqdm(masks) if show_progress_bar else masks
    for mask in masks:
        if mask_erotion>0:
            kernel = np.ones((mask_erotion, mask_erotion), np.uint8)
            mask = cv2.erode(mask.astype(np.uint8), kernel)
        mask_indices = np.where(mask)
        color = np.mean(image[mask_indices], axis=0)
        canvas[mask_indices] = color
    return canvas


def visualize_sample(sample, inputs):
    print(sample['text'])

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(inputs['image'][0])

    plt.subplot(1, 2, 2)
    plt.imshow(visualize_masks(inputs['image'][0], inputs['masks'][0]))
    plt.show()