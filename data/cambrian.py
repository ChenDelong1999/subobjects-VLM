import os
import torch
import json
import tqdm
from PIL import Image
import random
# set random seed
random.seed(42)

class Cambrian(torch.utils.data.Dataset):

    def __init__(self, root, annotation='Cambrian7M_withsystemprompt.jsonl', split=None, max_samples=None):
        self.root = root
        self.max_text_tokens = 300

        samples = []
        with open(os.path.join(root, 'annotations', annotation), 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                if max_samples and max_samples>0 and len(samples) >= max_samples:
                    break
                samples.append(json.loads(line))
                if len(samples) % 500000 == 0:
                    print(f'Loaded {len(samples)/1000000}M samples')
        print(f'Loaded {len(samples)} samples')
        random.shuffle(samples)

        self.samples = [s for s in samples if 'image' in s]
        print(f'After removing text-only samples: {len(self.samples)}')
        

    def process_sharegpt4v_sample(self, img_path, sample, image_tag='<|image|>', human_turn='<|user|>\n', gpt_turn='<|assistant|>\n', eos_token='<|endoftext|>\n'):
    # def __init__(self, root, split='train', user_tag='<|user|>\n', assistant_tag='<|assistant|>\n', image_tag='<|image|>', end_tag='<|endoftext|>\n', max_samples=None):
        text = ''
        for utterance in sample['conversations']:
            text += human_turn if utterance['from']=='human' else gpt_turn
            text += utterance['value'].replace('<image>', image_tag) + eos_token
        
        image = Image.open(img_path).convert('RGB')
        return {"text": text, "image": image.convert('RGB')}


    def __getitem__(self, index):

        try:
            sample = self.samples[index]
            img_path = os.path.join(self.root, sample['image'])
                
            assert os.path.exists(img_path), f'Image not found: {img_path}'
            return self.process_sharegpt4v_sample(img_path, sample)
        
        except Exception as e:
            print(f'Error loading sample {index}: {e}')
            return self.__getitem__((index + 1) % len(self.samples))
    
    def __len__(self):
        return len(self.samples)