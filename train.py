import argparse
import os
import logging

import json
import torch
import random
import numpy as np

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

from transformers import TrainingArguments


from model.utils import create_vlm
from model.utils import VisualTextualTokenization
from data import get_dataset
from visual_tokenizer import get_visual_tokenizer

from transformers import Trainer

from training.utils import (
    load_args_from_yaml,
    get_params_count_summary,
    get_run_description,
    save_and_print_args,
    set_random_seed
)

set_random_seed()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s | %(message)s')


if __name__ == '__main__':

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)

    torch.distributed.init_process_group(backend='nccl')

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--dataset_root', type=str, required=True)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--train_samples', type=int, default=None)

    parser.add_argument('--trainer_config', type=str, required=True)
    parser.add_argument('--visual_tokenizer_config', type=str, required=True)
    parser.add_argument('--llm', type=str, required=True)
    parser.add_argument('--lora_config', type=str, default=None)
    parser.add_argument('--visual_embed_config', type=str, required=True)

    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--dataloader_num_workers', type=int, default=8)

    parser.add_argument('--embedding_input_resolution', type=int, default=384)
    parser.add_argument('--tokenizer_input_resolution', type=int, default=384)
    parser.add_argument('--max_visual_tokens', type=int, default=128)

    parser.add_argument('--vm_loss_weight', type=float, default=1.0)
    parser.add_argument('--lm_loss_weight', type=float, default=1.0)
    parser.add_argument('--insert_queries', action='store_true')

    parser.add_argument('--output_dir', type=str)

    args = parser.parse_args()
    args = load_args_from_yaml(args.trainer_config, args)

    # get rank
    args.rank = torch.distributed.get_rank()
    
    # load dataset, tokenizer, model, and image segmenter
    train_dataset = get_dataset(
        args.dataset, args.dataset_root, split=args.split, max_samples=args.train_samples)

    # calculate max length
    if args.insert_queries:
        model_max_length = 2 * args.max_visual_tokens + train_dataset.max_text_tokens
    else:
        model_max_length = args.max_visual_tokens + train_dataset.max_text_tokens 

    # create model and textualn tokenizer
    model, textual_tokenizer = create_vlm(
        llm = args.llm, 
        visual_embed_config = args.visual_embed_config,
        embedding_input_resolution=args.embedding_input_resolution,
        tokenizer_input_resolution=args.tokenizer_input_resolution,
        lora_config = args.lora_config, 
        model_max_length=model_max_length
        )

    model.config.vm_loss_weight = args.vm_loss_weight
    model.config.lm_loss_weight = args.lm_loss_weight
    model.config.insert_queries = args.insert_queries
    
    # avoid CUDA OOM during evaluation
    # model.config.keys_to_ignore_at_inference = ['logits', 'past_key_values', 'hidden_states'] 

    # create visual and VL tokenizer (data_collector)
    visual_tokenizer = get_visual_tokenizer(
        **json.load(open(args.visual_tokenizer_config)), 
        image_resolution=args.tokenizer_input_resolution, 
        max_tokens=args.max_visual_tokens,
        device=f'cuda:{args.rank}'
        )
    vl_tokenizer = VisualTextualTokenization(textual_tokenizer, visual_tokenizer)

    args.training_args['output_dir'] = os.path.join(args.output_dir, get_run_description(args))
    training_args = TrainingArguments(
        **args.training_args,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        num_train_epochs=args.epoch,
        )

    # save and print args
    if torch.distributed.get_rank() == 0:
        print(f'{"-"*100}\n-> model:\n{model}\n{get_params_count_summary(model)}')
        print(f'{"-"*100}\n-> tokenizer:\n{textual_tokenizer}')
        print(f'{"-"*100}\n-> training sample 0:\n{train_dataset[0]}')
        save_and_print_args(args, training_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=vl_tokenizer,
        tokenizer=textual_tokenizer, 
    )

    # https://discuss.huggingface.co/t/no-log-for-validation-loss-during-training-with-trainer/40094/2
    trainer.can_return_loss = True 

    trainer.train(resume_from_checkpoint=False)
    print(args)
    