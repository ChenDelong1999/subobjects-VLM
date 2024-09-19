
import os
os.environ["NCCL_P2P_LEVEL"] = "NVL"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import datetime
import yaml
import torch
from dataclasses import dataclass
from peft import LoraConfig
from transformers import AutoTokenizer, PreTrainedTokenizerBase, PretrainedConfig

from data import ImageNet, CocoCaptionDataset, ImageParagraphCaptioning, CLEVRCaption
from model import VisionLanguageConfig, GPT2forVisionLanguageModeling, PhiforVisionLanguageModeling, LlamaforVisionLanguageModeling


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


@dataclass
class DataCollatorForVisionLanguageModeling:

    tokenizer: PreTrainedTokenizerBase
    image_segmenter: None
    token_resolution: int
    max_segments: int
    
    @torch.no_grad()
    def __call__(self, features, eval=False):

        text = [feature['text'] for feature in features]
        images = [feature['image'].resize((self.image_segmenter.image_resolution, self.image_segmenter.image_resolution)) for feature in features] 

        batch_masks = self.image_segmenter(images)
        batch_boxes = []
        batch_crop_masks = []
        for mask in batch_masks:
            boxes, masks = self.image_segmenter.convert_whole_mask_to_box_mask(mask[:self.max_segments], self.token_resolution)
            batch_boxes.append(boxes)
            batch_crop_masks.append(masks)

        if not eval:
            input_ids = self.tokenizer(text, padding='max_length', truncation=True, return_tensors='pt')['input_ids']
        else:
            input_ids = self.tokenizer(text, return_tensors='pt')['input_ids']

        return {'text': input_ids, 'image': images, 'boxes': batch_boxes, 'masks': batch_crop_masks}


def get_tokenizer(tokenizer_path, model_max_length):

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.padding_side = 'right'
    tokenizer.add_tokens(["<|assistant|>", "<|user|>", "<|system|>", "<|end|>", '<|image|>', '<|startofimage|>', '<|endofimage|>', '<|visual_content|>', '<|visual_position|>', '<unk>'], special_tokens=True)
    
    tokenizer.bos_token = '<s>'
    tokenizer.eos_token = '<|endoftext|>'
    tokenizer.unk_token = '<unk>'
    tokenizer.pad_token = '<unk>'

    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.model_max_length = model_max_length

    return tokenizer


def get_model(llm, vlm_config=None, lora_config=None):

    if 'phi' in llm.lower():
        modeling_class = PhiforVisionLanguageModeling
    elif 'gpt2' in llm.lower():
        modeling_class = GPT2forVisionLanguageModeling
    elif 'smollm' in llm.lower():
        modeling_class = LlamaforVisionLanguageModeling
    else:
        raise NotImplementedError
    
    # load model
    if llm.endswith('.json'):
        llm_config = PretrainedConfig.from_json_file(llm)
        model = modeling_class(llm_config)
        print(f'Biult randomly initialized LLM from config file: {llm_config}')
    else:
        model = modeling_class.from_pretrained(llm, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, device_map=None, cache_dir='/home/dchenbs/workspace/cache')
        llm_config = model.config
        print(f'Loded LLM from pretrained: {llm}')
    
    if vlm_config is not None:
        vlm_config = VisionLanguageConfig.from_json_file(vlm_config)
        print(f'Biulding VLM from config file: {vlm_config}')
        model.init_vlm(vlm_config)

    if lora_config is not None:
        lora_config = LoraConfig.from_json_file(lora_config)
        print(f'Inserting Lora from config file: {lora_config}')
        model.init_lora(lora_config)

    for param in model.embed_segments.feature_extractor.parameters():
        param.requires_grad = False

    return model
    
    
def get_run_description(args):

    description = datetime.datetime.now().strftime("%m%d-%H%M")
    description += '-' +  args.dataset
    description += '-' + args.vlm_config.split('/')[-1].split('.')[0]
    if args.insert_queries:
        description += '-vm' +  str(args.vm_loss_weight)
    else:
        description += '-no-query'
    description += '-' +  args.llm.split('/')[-1].split('.')[0]
    if args.lora_config is not None:
        description += '-' +  args.lora_config.split('/')[-1].split('.')[0]
    # description += '-' +  args.trainer_config.split('/')[-1].split('.')[0] 
    description += '-' +  args.image_segmenter.split('/')[-1].split('.')[0]
    
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




from transformers import Trainer

from transformers.training_args import OptimizerNames
from typing import Dict, Union, Any
import torch
import torch.nn as nn

from transformers.utils import is_sagemaker_mp_enabled
if is_sagemaker_mp_enabled():
    from transformers.trainer_pt_utils import smp_forward_backward

from transformers import is_apex_available
if is_apex_available():
    from apex import amp


class CustomTrainer(Trainer):
    
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:

        """
        Copied from default Trainer (transformers 4.42.3) and add additional logging
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True) # <- modified to return outputs

        if 'loss_terms' in outputs:
            for k in outputs['loss_terms']:
                outputs['loss_terms'][k] = outputs['loss_terms'][k].item() 
            self.log(outputs['loss_terms'])

        del inputs

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learnign rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss, **kwargs)

        return loss.detach() / self.args.gradient_accumulation_steps

@torch.no_grad()
def compute_metrics_for_loss_term_logging(predictions):
    return {k: v.mean() for k, v in predictions.predictions.items()}