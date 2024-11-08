import cv2
import numpy as np


def contour_recall(gt_contour, pred_contour):
    if np.count_nonzero(gt_contour) == 0:
        return 1.0
    return 1 - np.count_nonzero(np.logical_and(gt_contour, ~pred_contour)) / np.count_nonzero(gt_contour)


def create_circular_kernel(size):
    if size > 3:
        kernel = np.zeros((size, size), np.uint8)
        center = size // 2
        cv2.circle(kernel, (center, center), center, 1, -1)
    else:
        kernel = np.ones((size, size), np.uint8)
    return kernel


def masks_to_contour(masks, CIRCULAR_KERNEL):

    label_map = np.zeros_like(masks[0]).astype(np.int32)
    for i, mask in enumerate(masks):
        if np.sum(mask) == 0:
            continue
        label_map += (i + 1) * mask

    # HEIT datasets are 1024x1024
    label_map = cv2.resize(label_map, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    label_map = label_map.astype(np.uint8)

    dilated = cv2.dilate(label_map, CIRCULAR_KERNEL)
    eroded = cv2.erode(label_map, CIRCULAR_KERNEL)
    boundaries = dilated - eroded

    if len(boundaries.shape) == 2:
        boundaries = boundaries[..., np.newaxis]
    boundaries = np.any(boundaries, axis=-1).astype(bool)

    return boundaries

