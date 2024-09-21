
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    GPT2LMHeadModel,
    Phi3ForCausalLM,
    LlamaForCausalLM
    )

from peft import LoraConfig, LoraModel

from .visual_token_embedding import VisualTokenEmbedding

import torch._dynamo
import torch._dynamo.config
torch._dynamo.config.suppress_errors = True


class SubobjectVLM(PreTrainedModel):
    
    def __init__(self, config):
        super().__init__(config)

        self.visual_initialized = False
        self.lora_initialized = False

        if hasattr(config, 'visual_embed_config'):
            config.visual_embed_config = PretrainedConfig.from_dict(config.visual_embed_config)
            self.init_visual(config.visual_embed_config)

        if hasattr(config, 'lora_config'):
            self.init_lora(config.lora_config)

    def init_visual(self, visual_embed_config):

        if self.visual_initialized:
            print('VLM already initialized')
            return

        self.visual_token_embedding = VisualTokenEmbedding(visual_embed_config)
        feature_channels = self.visual_token_embedding.vision_encoder.feature_channels

        self.feature_embed = nn.Linear(
            feature_channels * (visual_embed_config.token_resolution ** 2), 
            self.config.hidden_size, bias=False)
        self.box_embed = nn.Linear(
            4, self.config.hidden_size, bias=False)
        self.mask_embed = nn.Linear(
            visual_embed_config.token_resolution * visual_embed_config.token_resolution, 
            self.config.hidden_size, bias=False)

        nn.init.zeros_(self.box_embed.weight)
        nn.init.zeros_(self.feature_embed.weight)
        nn.init.zeros_(self.mask_embed.weight)

        # self.feature_prediction_head = nn.Sequential(
        #     nn.Linear(self.config.hidden_size, self.config.hidden_size*2),
        #     nn.ReLU(),
        #     nn.Linear(self.config.hidden_size*2, self.config.hidden_size*2),
        #     nn.ReLU(),
        #     nn.Linear(self.config.hidden_size*2, feature_channels * (visual_embed_config.token_resolution ** 2))
        # )
        
        if not hasattr(self.config, 'visual_embed_config'):
            self.config.visual_embed_config = visual_embed_config

        self.visual_initialized = True

    
    def init_lora(self, lora_config):    

        if self.lora_initialized:
            print('Lora already initialized')
            return    

        self.model = LoraModel(self.model, LoraConfig(**lora_config), "default")
        if not hasattr(self.config, 'lora_config'):
            self.config.lora_config = lora_config

        self.lora_initialized = True


    def load_tokenizer_info(self, tokenizer):

        self.token_ids = {'pad': tokenizer.pad_token_id}
        for token_id, added_token in tokenizer.added_tokens_decoder.items():
            self.token_ids[added_token.content] = token_id
        
        self.model_max_length = tokenizer.model_max_length
        self.resize_token_embeddings(len(tokenizer))
        
    
    def insert_visual_tokens(self, batch_input_ids, batch_n_visual_tokens):
        
        batch_input_ids = batch_input_ids.to('cpu')
        batch_processed_input_ids, batch_position_idx, batch_content_idx = [], [], []

        start_token = torch.full((1,), self.token_ids['<|startofimage|>'], dtype=torch.long)
        end_token = torch.full((1,), self.token_ids['<|endofimage|>'], dtype=torch.long)
        
        for input_ids, n_visual_tokens in zip(batch_input_ids, batch_n_visual_tokens):

            image_token_idx = (input_ids == self.token_ids['<|image|>']).nonzero()[0].item()

            if self.config.insert_queries:
                visual_tokens = torch.full((2 * n_visual_tokens,), self.token_ids['<|visual_position|>'], dtype=torch.long)
                visual_tokens[1::2] = self.token_ids['<|visual_content|>']
                position_idx = np.arange(image_token_idx+1, image_token_idx+1+2*n_visual_tokens, 2)

                input_ids = torch.cat((
                    input_ids[:image_token_idx], 
                    start_token, visual_tokens, end_token, 
                    input_ids[image_token_idx+1:]
                    ), dim=0)[:self.model_max_length]

                batch_processed_input_ids.append(input_ids)
                batch_position_idx.append(position_idx)
                batch_content_idx.append(position_idx + 1)
            
            else:
                # no position queires, only content tokens
                visual_tokens = torch.full((n_visual_tokens,), self.token_ids['<|visual_content|>'], dtype=torch.long)
                content_idx = np.arange(image_token_idx + 1, image_token_idx + 1 + n_visual_tokens)

                input_ids = torch.cat((
                    input_ids[:image_token_idx],
                    start_token, visual_tokens, end_token,
                    input_ids[image_token_idx+1:]
                    ), dim=0)[:self.model_max_length]
                
                batch_processed_input_ids.append(input_ids)
                batch_position_idx.append([])
                batch_content_idx.append(content_idx)

        batch_processed_input_ids = torch.stack(batch_processed_input_ids, dim=0).to(self.device)
        return batch_processed_input_ids, batch_position_idx, batch_content_idx
    

    def prepare_visual_embeds(self, image, masks):

        boxes, masks, features = self.visual_token_embedding(image, masks)
        # boxes:    (N, M, 4)
        # masks:    (N, M, token_resolution, token_resolution)
        # features: (N, M, C * token_resolution * token_resolution)

        boxes = boxes.to(self.dtype).to(self.device).detach()
        masks = masks.to(self.dtype).to(self.device).detach()
        features = features.to(self.dtype).to(self.device).detach()

        not_padding = (boxes.sum(dim=-1) != 0).unsqueeze(-1)

        box_embeds = self.box_embed(boxes) * not_padding
        mask_embeds = self.mask_embed(masks.view(masks.shape[0], masks.shape[1], -1)) * not_padding
        feature_embeds = self.feature_embed(features) * not_padding

        return box_embeds, mask_embeds, feature_embeds, features, not_padding.squeeze(-1).sum(dim=-1).cpu().numpy()


    def prepare_inputs_embeds(self, input_ids, image, masks):

        assert len(input_ids) == len(image) == len(masks); f"Inputs batch size mismatch: {len(input_ids)}, {len(image)}, {len(masks)}"

        box_embeds, mask_embeds, feature_embeds, features, n_visual_tokens = self.prepare_visual_embeds(image, masks)
        input_ids, position_idx, content_idx = self.insert_visual_tokens(input_ids, n_visual_tokens)
        inputs_embeds = self.get_input_embeddings()(input_ids)

        # add visual embeddings to textual embeddings
        for i in range(len(input_ids)):
            inputs_embeds[i, content_idx[i]] += (
                box_embeds[i, :n_visual_tokens[i]] +
                mask_embeds[i, :n_visual_tokens[i]] + 
                feature_embeds[i, :n_visual_tokens[i]]
                ).to(self.dtype) 
            if self.config.insert_queries:
                inputs_embeds[i, position_idx[i]] += box_embeds[i, :n_visual_tokens[i]].to(self.dtype) 
        inputs_embeds = inputs_embeds.contiguous()

        # labels for textual next token prediction 
        lm_labels = input_ids.clone()
        lm_labels[lm_labels == self.token_ids['pad']] = -100
        for i in range(len(input_ids)): 
            # only predict the assistant's response
            lm_labels[i, : (input_ids[i] == self.token_ids['<|assistant|>']).nonzero()[0].item()+1] = -100

        return inputs_embeds, {
            'lm_labels': lm_labels,
            'features': features,
            'position_idx': position_idx,
        }
    

    def get_vision_modeling_loss(self, last_hidden_states, features, position_idx):
        # vm_loss = 0
        vm_loss = torch.tensor(0.0, device=self.device)
        if self.config.insert_queries:
            for i in range(len(last_hidden_states)):
                segment_prediction = self.feature_prediction_head(last_hidden_states[i][position_idx[i]])
                vm_loss += nn.functional.mse_loss(segment_prediction, features[i])
        return vm_loss
    

    def forward(self, text=None, image=None, masks=None, labels=None, inputs_embeds=None, **kwargs):

        if inputs_embeds is None and 'past_key_values' not in kwargs: # for training
            inputs_embeds, labels = self.prepare_inputs_embeds(text, image, masks)

        kwargs['output_hidden_states'] = True
        outputs = super().forward(
            inputs_embeds=inputs_embeds, 
            labels=labels['lm_labels'] if labels else None,
            **kwargs
            )
        
        if labels: # for training
            outputs = OrderedDict(outputs)
            loss_vm = self.get_vision_modeling_loss(
                outputs['hidden_states'][-1], labels['features'], labels['position_idx']
                )
            outputs['loss_terms'] = {'loss_lm': outputs['loss'].detach().clone(), 'loss_vm': loss_vm}
            outputs['loss'] = outputs['loss'] * self.config.lm_loss_weight + loss_vm * self.config.vm_loss_weight
        return outputs
    

class GPT2forVisionLanguageModeling(SubobjectVLM, GPT2LMHeadModel):

    def __init__(self, config):
        super().__init__(config)

    def forward(self, text=None, image=None, masks=None, labels=None, inputs_embeds=None, **kwargs):
        return super().forward(text=text, image=image, masks=masks, labels=labels, inputs_embeds=inputs_embeds, **kwargs)
    

class LlamaforVisionLanguageModeling(SubobjectVLM, LlamaForCausalLM):

    def __init__(self, config):
        super().__init__(config)

    def forward(self, text=None, image=None, masks=None, labels=None, inputs_embeds=None, **kwargs):
        return super().forward(text=text, image=image, masks=masks, labels=labels, inputs_embeds=inputs_embeds, **kwargs)
    

class PhiforVisionLanguageModeling(SubobjectVLM, Phi3ForCausalLM):

    def __init__(self, config):
        super().__init__(config)

    def forward(self, text=None, image=None, masks=None, labels=None, inputs_embeds=None, **kwargs):
        return super().forward(text=text, image=image, masks=masks, labels=labels, inputs_embeds=inputs_embeds, **kwargs)