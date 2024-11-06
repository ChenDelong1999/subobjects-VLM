import os
import torch
import json
import tqdm
from PIL import Image
import random
# set random seed
random.seed(42)

train_val_split={'train': 0.95, 'val': 0.05}

class ShareGPT4V(torch.utils.data.Dataset):

    def __init__(self, root, annotation, split='train', max_samples=None):
        self.root = root
        self.max_text_tokens = 300
        self.max_samples = max_samples
        samples = json.load(open(os.path.join(root, 'sharegpt4v', annotation), 'r'))
        random.shuffle(samples)

        if split == 'train':
            start_idx = 0
            end_idx = int(len(samples) * train_val_split['train'])
        elif split == 'val':
            start_idx = int(len(samples) * train_val_split['train'])
            end_idx = len(samples)
        else:
            raise ValueError(f'split should be one of [train, val], but got {split}')
        
        self.samples = samples[start_idx:end_idx]
        print(f'Total samples: {len(samples)}, using {split} split: {len(self.samples)} (from {start_idx} to {end_idx})')

        self.samples = [s for s in self.samples if 'image' in s]
        print(f'After removing text-only samples: {len(self.samples)}')
        

    def process_sharegpt4v_sample(self, img_path, sample, image_tag='<|image|>', human_turn='<|user|>\n', gpt_turn='<|assistant|>\n', eos_token='<|endoftext|>\n'):
    # def __init__(self, root, split='train', user_tag='<|user|>\n', assistant_tag='<|assistant|>\n', image_tag='<|image|>', end_tag='<|endoftext|>\n', max_samples=None):
        text = ''
        for utterance in sample['conversations']:
            text += human_turn if utterance['from']=='human' else gpt_turn
            text += utterance['value'].replace('<image>', image_tag) + eos_token
        
        image = Image.open(img_path).convert('RGB')
        return {"text": text, "image": image.convert('RGB')}


    def __getitem__(self, index, only_return_img_path=False):

        try:
            sample = self.samples[index]
            img_path = sample['image']
            if 'sam/images' in img_path:
                img_path = img_path.replace('sam/images', '/datasets01/segment_anything/032023_anonymized_resized')
            else:
                img_path = os.path.join(self.root, sample['image'])
                
            assert os.path.exists(img_path), f'Image not found: {img_path}'

            if only_return_img_path:
                return img_path
            else:
                return self.process_sharegpt4v_sample(img_path, sample)
        except Exception as e:
            print(f'Error loading sample {index}: {e}')
            return self.__getitem__((index + 1) % len(self.samples))
    
    def __len__(self):
        if self.max_samples is not None:
            return min(len(self.samples), self.max_samples)
        else:
            return len(self.samples)