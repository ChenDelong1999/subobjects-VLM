

import os
os.environ["NCCL_P2P_LEVEL"] = "NVL"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import torch
from dataclasses import dataclass
from peft import LoraConfig
from transformers import AutoTokenizer, PreTrainedTokenizerBase, PretrainedConfig

from .modeling import GPT2forVisionLanguageModeling, PhiforVisionLanguageModeling, LlamaforVisionLanguageModeling

@dataclass
class VisualTextualTokenization:

    textual_tokenizer: PreTrainedTokenizerBase
    visual_tokenizer: None
    
    @torch.no_grad()
    def __call__(self, features, eval=False):

        text = [feature['text'] for feature in features]
        images = [feature['image'].resize((self.visual_tokenizer.image_resolution, self.visual_tokenizer.image_resolution)) for feature in features] 

        batch_masks = self.visual_tokenizer(images)

        if not eval:
            input_ids = self.textual_tokenizer(text, padding='max_length', truncation=True, return_tensors='pt')['input_ids']
        else:
            input_ids = self.textual_tokenizer(text, return_tensors='pt')['input_ids']

        return {'text': input_ids, 'image': images, 'masks': batch_masks}


def create_textual_tokenizer(tokenizer_path, model_max_length):

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.padding_side = 'right'
    tokenizer.add_tokens(["<|assistant|>", "<|user|>", "<|system|>", "<|end|>", '<|image|>', '<|startofimage|>', '<|endofimage|>', '<|visual_content|>', '<|visual_position|>', '<unk>'], special_tokens=True)
    
    tokenizer.bos_token = '<s>'
    tokenizer.eos_token = '<|endoftext|>'
    tokenizer.unk_token = '<unk>'
    tokenizer.pad_token = '<unk>'

    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    if model_max_length is not None:
        tokenizer.model_max_length = model_max_length

    return tokenizer


def create_vlm(
        llm,
        visual_embed_config=None,
        embedding_input_resolution=None,
        tokenizer_input_resolution=None,
        lora_config=None,
        model_max_length=None):

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
        model = modeling_class.from_pretrained(llm, torch_dtype=torch.bfloat16, device_map=None)
        llm_config = model.config
        print(f'Loded LLM from pretrained: {llm}')
    
    if visual_embed_config is not None:
        if type(visual_embed_config) == str and visual_embed_config.endswith('.json'):
            visual_embed_config = json.load(open(visual_embed_config))
        visual_embed_config['output_resolution'] = tokenizer_input_resolution
        visual_embed_config['image_resolution'] = embedding_input_resolution
        visual_embed_config = PretrainedConfig.from_dict(visual_embed_config)
        print(f'Biulding VLM from config: {visual_embed_config}')
        model.init_visual(visual_embed_config)

    if lora_config is not None:
        lora_config = LoraConfig.from_json_file(lora_config)
        print(f'Inserting Lora from config file: {lora_config}')
        model.init_lora(lora_config)

    for param in model.visual_token_embedding.vision_encoder.parameters():
        param.requires_grad = False

    tokenizer = create_textual_tokenizer(llm, model_max_length)
    model.load_tokenizer_info(tokenizer)

    return model, tokenizer
    