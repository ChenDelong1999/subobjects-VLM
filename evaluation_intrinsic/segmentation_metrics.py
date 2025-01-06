#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import random
import cv2
import tqdm

import torch
import torch.nn.functional as F
import pycocotools.mask as mask_util

from datasets import load_dataset


def load_samples(output_dir, split, resolution, model):
    samples_dict = {}
    results_dir = f'{output_dir}/{split}/{resolution}/{model}'
    for file in os.listdir(results_dir):
        # Skip non-JSON or hidden files if needed
        if not file.endswith('.json'):
            continue
        idx = int(file.split('.')[0])
        samples_dict[idx] = os.path.join(results_dir, file)

    # Build a list sorted by index
    samples = []
    for index in range(len(samples_dict)):
        samples.append(samples_dict[index])
    return samples


def decode_masks(mask_rles):
    """
    Decodes a list of RLEs (Run-Length Encoded masks) into a NumPy array (N, H, W).
    Filters out all-zero masks.
    """
    masks = []
    for mask_rle in mask_rles:
        mask = mask_util.decode(mask_rle)  # shape (H, W)
        masks.append(mask)
    masks = np.array(masks)  # (N, H, W)
    mask_sums = masks.sum(axis=(1, 2))
    masks = masks[mask_sums > 0]  # remove empty
    return masks


def masks_to_label_map(masks, device='cuda', output_size=(1024, 1024)):
    """
    Converts multiple binary masks into a single label map, with optional resizing to (H_out, W_out).
    Each non-zero region in mask i is labeled with (i+1).
    """
    if not isinstance(masks, torch.Tensor):
        masks = torch.tensor(masks, device=device)
    else:
        masks = masks.to(device)

    if masks.ndim != 3:
        raise ValueError("masks should be of shape (N, H, W)")

    label_map = torch.zeros_like(masks[0], dtype=torch.int64)
    for i, mask in enumerate(masks):
        if torch.sum(mask) == 0:
            continue
        label_map += (i + 1) * mask

    if output_size is not None:
        label_map = label_map.unsqueeze(0).unsqueeze(0).float()
        label_map = F.interpolate(label_map, size=output_size, mode='nearest')
        label_map = label_map.squeeze(0).squeeze(0).long()

    return label_map


def label_map_to_contour(label_map, tolerance):
    """
    Converts a labeled map into a contour (boundary) map using dilation/erosion.
    """
    label_map_4d = label_map.unsqueeze(0).unsqueeze(0).float()  # (1,1,H,W)
    dilated = F.max_pool2d(
        label_map_4d,
        kernel_size=2 * tolerance + 1,
        stride=1,
        padding=tolerance
    )
    eroded = -F.max_pool2d(
        -label_map_4d,
        kernel_size=2 * tolerance + 1,
        stride=1,
        padding=tolerance
    )
    boundaries = (dilated != eroded).squeeze(0).squeeze(0).bool()
    boundaries &= (label_map != 0)
    return boundaries


def contour_metrics(
    contour_gt: torch.Tensor,
    contour_pred_thin: torch.Tensor,
    contour_pred_dilated: torch.Tensor,
    tolerance: int = 5,
    eps: float = 1e-6
):
    """
    Calculate precision/recall/F1 ignoring the outer `tolerance` boundary.
    - Precision uses the thin predicted contour.
    - Recall uses the dilated predicted contour.
    """
    contour_gt = contour_gt.bool()
    contour_pred_thin = contour_pred_thin.bool()
    contour_pred_dilated = contour_pred_dilated.bool()

    H, W = contour_gt.shape
    if (2 * tolerance >= H) or (2 * tolerance >= W):
        cropped_gt = contour_gt
        cpt = contour_pred_thin
        cpd = contour_pred_dilated
    else:
        cropped_gt = contour_gt[tolerance:-tolerance, tolerance:-tolerance]
        cpt = contour_pred_thin[tolerance:-tolerance, tolerance:-tolerance]
        cpd = contour_pred_dilated[tolerance:-tolerance, tolerance:-tolerance]

    # Precision
    tp = torch.sum(cropped_gt & cpt).float()
    fp = torch.sum(~cropped_gt & cpt).float()
    precision = tp / (tp + fp + eps)

    # Recall
    tp = torch.sum(cropped_gt & cpd).float()
    fn = torch.sum(cropped_gt & ~cpd).float()
    recall = tp / (tp + fn + eps)

    # F1
    f1 = 2 * (precision * recall) / (precision + recall + eps)

    return {
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item()
    }


def compute_monosemanticity(
    contour_gt: torch.Tensor,
    label_map_pred: torch.LongTensor,
    tolerance: int = 5
):
    """
    Morphologically erode each label by `tolerance`, then count how many GT contour pixels
    fall inside that eroded region. Returns one count per label.
    """
    unique_labels = torch.unique(label_map_pred)

    # Build stack of label masks
    label_stack = torch.stack([label_map_pred == lbl for lbl in unique_labels], dim=0)
    # Erode each label
    label_stack_4d = label_stack.unsqueeze(1).float()  # (K,1,H,W)
    eroded_4d = -F.max_pool2d(
        -label_stack_4d,
        kernel_size=2 * tolerance + 1,
        stride=1,
        padding=tolerance
    )
    eroded_stack = eroded_4d.squeeze(1).bool()  # (K,H,W)

    # Count overlap
    crossing_tensor = eroded_stack & contour_gt.unsqueeze(0)
    n_crossing_pixels = crossing_tensor.sum(dim=(1, 2)).cpu().tolist()

    return n_crossing_pixels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--results_dir", type=str, default="outputs/segmentation_results")
    parser.add_argument("--output_dir", type=str, default="outputs/segmentation_metrics")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tolerance_recall", type=int, default=5)
    parser.add_argument("--tolerance_monosemanticity", type=int, default=25)
    parser.add_argument("--max_eval", type=int, default=None, help="Max # of samples to process")
    args = parser.parse_args()

    # Prepare
    dataset = load_dataset("chendelong/HEIT", split=args.split)
    os.makedirs(f"{args.output_dir}/{args.split}/{args.resolution}", exist_ok=True)

    total_area = args.resolution**2

    model_dir = os.path.join(args.results_dir, args.split, str(args.resolution), args.model_name)
    if not os.path.isdir(model_dir):
        print(f"[Error] No directory found at {model_dir}. Exiting.")
        return

    output_file = os.path.join(args.output_dir, args.split, str(args.resolution), f"{args.model_name}.json")
    if os.path.exists(output_file):
        print(f"[Skipping] {args.model_name}, already has metrics => {output_file}")
        return

    print(f"Evaluating model: {args.model_name}")
    samples_pred_paths = load_samples(args.results_dir, args.split, args.resolution, args.model_name)

    all_metrics = []
    n_samples = len(samples_pred_paths) if args.max_eval is None else min(args.max_eval, len(samples_pred_paths))

    for i in tqdm.tqdm(range(n_samples), desc=args.model_name):
        sample = dataset[i]
        contour_gt = torch.tensor(np.array(sample["contour"]), device=args.device).bool()

        sample_pred_path = samples_pred_paths[i]
        with open(sample_pred_path, "r") as f:
            sample_pred = json.load(f)

        # If no predicted masks => single empty mask
        if len(sample_pred.get("rles", [])) == 0:
            masks_pred = torch.zeros((1, args.resolution, args.resolution), device=args.device, dtype=torch.bool)
        else:
            masks = decode_masks(sample_pred["rles"])
            masks_pred = torch.tensor(masks, device=args.device, dtype=torch.bool)

        # Convert to label map
        label_map_pred = masks_to_label_map(
            masks_pred,
            device=args.device,
            output_size=(1024, 1024)
        )

        # Build contours
        contour_pred_thin = label_map_to_contour(label_map_pred, tolerance=2)
        contour_pred_dilated = label_map_to_contour(label_map_pred, tolerance=args.tolerance_recall)

        # Contour metrics
        c_metrics = contour_metrics(
            contour_gt,
            contour_pred_thin,
            contour_pred_dilated,
            tolerance=args.tolerance_recall
        )
        c_metrics["time"] = sample_pred.get("time", 0.0)
        c_metrics["n_tokens"] = masks_pred.shape[0]

        # Areas
        mask_areas = masks_pred.sum(dim=(1, 2)) / total_area
        c_metrics["mask_areas"] = mask_areas.cpu().numpy().tolist()

        # Monosemanticity
        crossing_pixels = compute_monosemanticity(
            contour_gt,
            label_map_pred,
            tolerance=args.tolerance_monosemanticity
        )
        c_metrics["monosemanticity"] = crossing_pixels

        all_metrics.append(c_metrics)

    # Save
    with open(output_file, "w") as f:
        json.dump(all_metrics, f)
    print(f"[Saved] {output_file}")


if __name__ == "__main__":
    main()