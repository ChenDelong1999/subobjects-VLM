import os
import json
import argparse
import numpy as np
import tqdm
import json
import pycocotools.mask as mask_util
import datetime
import sys
sys.path.append('..')

from visual_tokenizer import get_visual_tokenizer
from datasets import load_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--split', type=str, required=True, help='Dataset split to use')
    parser.add_argument('--tokenizer_config', type=str, required=True, help='Path to the tokenizer configuration file')
    parser.add_argument('--input_resolution', type=int, default=1024, help='Resolution of the input images')
    parser.add_argument('--output_dir', type=str, default='outputs/token_vs_contour_recall', help='Directory to save the output')
    
    args = parser.parse_args()
    args.time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    dataset = load_dataset("chendelong/HEIT", split=args.split)
    print(dataset)

    config = json.load(open(args.tokenizer_config, 'r'))
    print(config)
    visual_tokenizer = get_visual_tokenizer(
        **config, 
        image_resolution=args.input_resolution, 
        max_tokens=1024
    )

    print(args)
    samples = []
    output_dir = os.path.join(args.output_dir, f'{args.split}/{args.input_resolution}/{args.tokenizer_config.split("/")[-1].replace(".json", "")}')
    os.makedirs(output_dir, exist_ok=True)

    # for i in tqdm.tqdm(range(5)):
    for i in tqdm.tqdm(range(len(dataset))):
        sample = dataset[i]
        image = sample['image'].resize((args.input_resolution, args.input_resolution))
        
        masks = visual_tokenizer(image).cpu().numpy()[0]
        rles = []
        for mask in masks:
            if np.sum(mask) == 0:
                continue
            rle = mask_util.encode(np.asfortranarray(mask))
            rle['counts'] = rle['counts'].decode('utf-8')
            rles.append(rle)

        json.dump(rles, open(os.path.join(output_dir, f'{i}.json'), 'w'), indent=4)
