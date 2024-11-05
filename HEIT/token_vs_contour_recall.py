import os
import json
import argparse
import cv2
import numpy as np
import tqdm
import json
import time
import datetime
import sys
sys.path.append('..')

from visual_tokenizer import get_visual_tokenizer
from datasets import load_dataset


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


def contour_recall(gt_contour, pred_contour):
    if np.count_nonzero(gt_contour) == 0:
        return 1.0
    return 1 - np.count_nonzero(np.logical_and(gt_contour, ~pred_contour)) / np.count_nonzero(gt_contour)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--split', type=str, required=True, help='Dataset split to use')
    parser.add_argument('--tokenizer_config', type=str, required=True, help='Path to the tokenizer configuration file')
    parser.add_argument('--input_resolution', type=int, default=1024, help='Resolution of the input images')
    parser.add_argument('--output_dir', type=str, default='outputs/token_vs_contour_recall', help='Directory to save the output')
    parser.add_argument('--tolerance', type=int, default=10, help='Tolerance level for processing')
    parser.add_argument('--max_tokens', type=int, default=1024, help='Maximum number of tokens to generate')
    parser.add_argument('--threshold', type=float, default=None, help='For DirectSAM ablation')
    
    args = parser.parse_args()
    args.time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    dataset = load_dataset("chendelong/HEIT", split=args.split)
    print(dataset)

    config = json.load(open(args.tokenizer_config, 'r'))
    if args.threshold is not None:
        config['threshold'] = args.threshold

    print(config)
    visual_tokenizer = get_visual_tokenizer(
        **config, 
        image_resolution=args.input_resolution, 
        max_tokens=args.max_tokens
    )

    CIRCULAR_KERNEL = create_circular_kernel(args.tolerance)

    print(args)

    samples = []
    # for i in tqdm.tqdm(range(3)):
    for i in tqdm.tqdm(range(len(dataset))):
        sample = dataset[i]
        image = sample['image'].resize((args.input_resolution, args.input_resolution))
        
        masks = visual_tokenizer(image).cpu().numpy()[0]

        gt_contour = np.array(sample['contour'])
        gt_contour[:args.tolerance] = gt_contour[-args.tolerance:] = gt_contour[:, :args.tolerance] = gt_contour[:, -args.tolerance:] = 0
        
        pred_contour = masks_to_contour(masks, CIRCULAR_KERNEL)

        start = time.time()
        recall = contour_recall(gt_contour, pred_contour)
        tokens = (np.sum(masks, axis=(1, 2)) > 0).sum().item()

        samples.append([tokens, recall])
        # print(f'Number of tokens: {tokens}, Recall: {recall}')

    results = {
        'mean_tokens': np.mean([sample[0] for sample in samples]).item(),
        'mean_recall': np.mean([sample[1] for sample in samples]).item(),
        'args': vars(args),
        'samples': samples
    }
    output_dir = os.path.join(args.output_dir, f'{args.split}')
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f'{args.tokenizer_config.split("/")[-1].replace(".json", "")}_{args.time}.json')
    json.dump(results, open(file_path, 'w'), indent=4)

    print(f'Mean number of tokens: {results["mean_tokens"]}, Mean recall: {results["mean_recall"]}')
    print(f'Results of {len(samples)} samples saved at {file_path}')

