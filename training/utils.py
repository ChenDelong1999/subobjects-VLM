
import os
os.environ["NCCL_P2P_LEVEL"] = "NVL"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import datetime
import yaml
import torch
import random
import numpy as np

def set_random_seed(random_seed=42):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)


def load_args_from_yaml(file_path, args):
    with open(file_path, 'r') as stream:
        try:
            new_args = yaml.safe_load(stream)
            for key, value in new_args.items():
                args.__setattr__(key, value)
        except yaml.YAMLError as exc:
            print(exc)
    return args


def get_params_count_summary(model, max_name_len: int = 96):
  padding = 64
  
  params = [(name[:max_name_len], p.numel(), str(tuple(p.shape)), p.requires_grad) for name, p in model.named_parameters()]
  total_trainable_params = sum([x[1] for x in params if x[-1]])
  total_nontrainable_params = sum([x[1] for x in params if not x[-1]])
  
  param_counts_text = ''
  param_counts_text += '=' * (max_name_len + padding) + '\n'
  param_counts_text += f'| {"Module":<{max_name_len}} | {"Trainable":<8} | {"Shape":>20} | {"Param Count":>13} |\n'
  param_counts_text += '-' * (max_name_len + padding) + '\n'
  
  for name, param_count, shape, trainable in params:
      truncated_name = name[:max_name_len]  # Truncate the name if it's too long
      param_counts_text += f'| {truncated_name:<{max_name_len}} | {"True" if trainable else "False":<8} | {shape:>20} | {param_count:>13,} |\n'
  param_counts_text += '-' * (max_name_len + padding) + '\n'
  param_counts_text += f'| {"Total trainable params":<{max_name_len}} | {"":<8} | {"":<20} | {total_trainable_params:>13,} |\n'
  param_counts_text += f'| {"Total non-trainable params":<{max_name_len}} | {"":<8} | {"":<20} | {total_nontrainable_params:>13,} |\n'
  param_counts_text += '=' * (max_name_len + padding) + '\n'
  
  return param_counts_text

    
def get_run_description(args):

    description = datetime.datetime.now().strftime("%m%d-%H%M")
    description += '-' +  args.dataset
    description += '-' + args.visual_tokenizer_config.split('/')[-1].split('.')[0] + f'({args.max_visual_tokens})'
    description += '-' + args.visual_embed_config.split('/')[-1].split('.')[0]
    # if args.insert_queries:
    #     description += '-vm' +  str(args.vm_loss_weight)
    # else:
    #     description += '-no-query'
    description += '-' +  args.llm.split('/')[-1].replace('.', '_')
    if args.lora_config is not None:
        description += '-' +  args.lora_config.split('/')[-1].split('.')[0]
    
    return description


def save_and_print_args(args, training_args):

    os.makedirs(training_args.output_dir, exist_ok=True)
    with open(f'{training_args.output_dir}/args.yaml', 'w') as f:
        yaml.dump(vars(args), f)

    with open(f'{training_args.output_dir}/all_training_args.yaml', 'w') as f:
        yaml.dump(vars(training_args), f)

    print(f'{"-"*100}\n-> args:')
    for arg in vars(args):
        print(f'\t{arg:25}\t{getattr(args, arg)}')

    print(f'{"-"*100}\n-> training args:')
    for arg in args.training_args:
        print(f'\t{arg:25}\t{args.training_args[arg]}')

