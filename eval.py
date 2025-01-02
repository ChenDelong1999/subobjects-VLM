# eval.py

import argparse
import os
import random
import json
import tqdm
import torch
import datetime
import yaml
import math

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

    # Generation parameters
    parser.add_argument('--max_new_tokens', type=int, default=512, 
                        help='Number of tokens to generate for each sample (for autoregressive generation)')

    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime('%m%d-%H%M')
    eval_output_dir = os.path.dirname(args.model_checkpoint)
    eval_output_path = os.path.join(eval_output_dir, f'vlm_eval_{args.split.split("_")[0]}_{timestamp}.json')
    
    # Avoid overwriting existing evaluation files
    for file in os.listdir(eval_output_dir):
        if file.startswith(f'vlm_eval_{args.split.split("_")[0]}') and file.endswith('.json'):
            print(f"Output file {file} already exists. Exiting.")
            return

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
    elif args.dataset.lower() == 'pixmo_cap':
        from data import PixmoDataset
        DatasetClass = PixmoDataset
    elif args.dataset.lower() == 'clevr_caption':
        from data import CLEVRCaption
        DatasetClass = CLEVRCaption
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
    num_samples = min(args.num_samples, len(dataset))
    num_na = 0
    total_loss = 0
    total_visual_tokens = 0
    generations = []

    for i in tqdm.tqdm(range(num_samples)):
        # Randomly pick a sample
        sample = dataset[random.randint(0, len(dataset) - 1)]
        
        # We'll do a forward pass with the full text so that the 'loss' is meaningful
        inputs = vl_tokenizer([sample], eval=True)

        # Count how many visual tokens
        mask_sum = inputs['masks'].sum(dim=(2, 3))[0]
        total_visual_tokens += (mask_sum > 0).sum().item()

        with torch.no_grad():
            loss_val = model(**inputs)['loss'].item()

        if math.isnan(loss_val):
            num_na += 1
            sample_loss = 0.0
            sample_ppl = float('inf')
        else:
            sample_loss = loss_val
            sample_ppl = math.exp(sample_loss)  # perplexity = exp(loss)

        total_loss += sample_loss

        raw_text = sample['text']
        ground_truth = ""

        # For many text datasets, there's a pattern like "<|assistant|>" dividing prompt/response.
        # If that doesn't apply to your dataset, adapt accordingly.
        if "<|assistant|>" in raw_text:
            parts = raw_text.split("<|assistant|>")
            # everything up to <|assistant|> is the prompt
            prompt_text = parts[0] + "<|assistant|>"
            # everything after <|assistant|> is the ground truth
            ground_truth = parts[1].strip().replace(textual_tokenizer.eos_token, "")
        else:
            # If no special marker, treat the entire text as prompt or do something else
            prompt_text = raw_text

        gen_sample = dict(sample)  # shallow copy
        gen_sample['text'] = prompt_text

        inputs_gen = vl_tokenizer([gen_sample], eval=True)

        with torch.no_grad():
            inputs_embeds, _ = model.prepare_inputs_embeds(
                inputs_gen['text'],
                inputs_gen['image'],
                inputs_gen['masks']
            )
            outputs = model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=textual_tokenizer.eos_token_id,
                pad_token_id=textual_tokenizer.pad_token_id,
            )

        prediction = textual_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        generations.append({
            "generation": prediction,
            "ground_truth": ground_truth,
            "perplexity": sample_ppl,
        })

        if (i + 1) % 5 == 0 or i == num_samples - 1:
            print(f"Sample {i+1}: Loss: {loss_val:.4f}, PPL: {sample_ppl:.4f}")
            print(f"Running Avg Loss: {total_loss / (i + 1 - num_na):.4f}")

            valid_count = len(generations) - num_na
            average_loss = total_loss / valid_count if valid_count > 0 else float('inf')
            average_visual_tokens = total_visual_tokens / len(generations)

            # Save results
            eval_results = {
                "evaluation_finished": i==num_samples-1,
                "average_loss": average_loss,
                "average_visual_tokens": average_visual_tokens,
                "evaluation_args": vars(args),
                "checkpoint_args": checkpoint_args,
                "generations": generations
            }

            with open(eval_output_path, 'w') as f:
                json.dump(eval_results, f, indent=4)
    
    print(f"Saved evaluation results to {eval_output_path}")

if __name__ == '__main__':
    main()