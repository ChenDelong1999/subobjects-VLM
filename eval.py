# eval.py

import argparse
import os
import random
import json
import tqdm
import torch
import datetime
import yaml

from model.utils import create_vlm
from model.utils import VisualTextualTokenization
from data import get_dataset
from visual_tokenizer import get_visual_tokenizer

def main():
    parser = argparse.ArgumentParser(description='Evaluate model loss on a dataset.')

    # Dataset parameters
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset class to use')
    parser.add_argument('--dataset_root', type=str, required=True, help='Root directory of the dataset')
    parser.add_argument('--split', type=str, required=True, help='Dataset split to use (e.g., train, val)')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to evaluate')

    # Model parameters
    parser.add_argument('--model_checkpoint', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--llm_class', type=str, default='phi', help='Language model class')

    args = parser.parse_args()

    # Import dataset class based on the provided name
    if args.dataset.lower() == 'sharegpt4v':
        from data import ShareGPT4V
        DatasetClass = ShareGPT4V
    elif args.dataset.lower() == 'imagenet':
        from data import ImageNet
        DatasetClass = ImageNet
    elif args.dataset.lower() == 'cambrian':
        from data import Cambrian
        DatasetClass = Cambrian
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Load dataset
    dataset = DatasetClass(
        root=args.dataset_root,
        split=args.split
    )

    # Load model and textual tokenizer
    model, textual_tokenizer = create_vlm(
        model_name_or_checkpoint=args.model_checkpoint,
        llm_class=args.llm_class,
    )
    model = model.to('cuda').half().eval()

    # Load args.yaml from the checkpoint directory
    args_yaml_path = os.path.join(os.path.dirname(args.model_checkpoint), 'args.yaml')
    with open(args_yaml_path, 'r') as f:
        checkpoint_args = yaml.safe_load(f)

    # Get visual tokenizer parameters from args.yaml
    visual_tokenizer_config_path = checkpoint_args['visual_tokenizer_config']
    tokenizer_input_resolution = checkpoint_args.get('tokenizer_input_resolution', 224)
    max_visual_tokens = checkpoint_args.get('max_visual_tokens', 4)

    # Load visual tokenizer configuration
    with open(visual_tokenizer_config_path, 'r') as f:
        config = json.load(f)

    visual_tokenizer = get_visual_tokenizer(
        **config,
        image_resolution=tokenizer_input_resolution,
        max_tokens=max_visual_tokens,
    )

    vl_tokenizer = VisualTextualTokenization(textual_tokenizer, visual_tokenizer)

    # Evaluate loss over num_samples
    num_samples = min(args.num_samples, len(dataset))
    num_na = 0
    total_loss = 0
    total_visual_tokens = 0
    for _ in tqdm.tqdm(range(num_samples)):
        sample = dataset[random.randint(0, len(dataset) - 1)]
        inputs = vl_tokenizer([sample], eval=True)

        mask_sum = inputs['masks'].sum(dim=(2, 3))[0]
        total_visual_tokens += (mask_sum > 0).sum().item()

        with torch.no_grad():
            loss = model(**inputs)['loss'].item()
            # check if loss is NaN
            if loss != loss:
                num_na += 1
            else:
                total_loss += loss

    average_loss = total_loss / (num_samples - num_na)
    average_visual_tokens = total_visual_tokens / num_samples
    print(f"Average Loss: {average_loss}")
    print(f"Average Visual Tokens: {average_visual_tokens}")

    # Save the average loss and evaluation arguments to a JSON file
    timestamp = datetime.datetime.now().strftime('%m%d-%H%M')
    eval_results = {
        'average_loss': average_loss,
        'average_visual_tokens': average_visual_tokens,
        'evaluation_args': vars(args),
        'checkpoint_args': checkpoint_args,
    }

    eval_output_dir = os.path.dirname(args.model_checkpoint)
    eval_output_path = os.path.join(eval_output_dir, f'eval_{args.split.split("_")[0]}_{timestamp}.json')
    with open(eval_output_path, 'w') as f:
        json.dump(eval_results, f, indent=4)

if __name__ == '__main__':
    main()