import json
import os
import torch
from PIL import Image

class CLEVRCaption(torch.utils.data.Dataset):

    def __init__(self, root, split, user_tag='<|user|>\n', assistant_tag='<|assistant|>\n', image_tag='<|image|>', end_tag='<|endoftext|>\n', max_samples=None):
        self.dataset_name = 'CLEVRCaption'
        self.root = root
        self.split = split

        self.user_tag = user_tag
        self.assistant_tag = assistant_tag
        self.image_tag = image_tag
        self.end_tag = end_tag
        self.max_samples = max_samples
        self.max_text_tokens = 100
        
        annotation_file = os.path.join(root, 'captions', f'{split}.json')
        self.samples = json.load(open(annotation_file, 'r'))

    def __len__(self):
        if self.max_samples is not None and self.max_samples > 0:
            return self.max_samples
        else:
            return len(self.samples)
    
    def __getitem__(self, idx):
        sample =  self.samples[idx]
        
        return {
            # 'text': f'{self.user_tag} Describe this image: {self.image_tag}{self.end_tag}{self.assistant_tag}{sample["caption"]}{self.end_tag}',
            'text': f'{self.user_tag} Enumerate all objects and their size, color, material, and shape, from left to right: {self.image_tag}{self.end_tag}{self.assistant_tag}{sample["caption"]}{self.end_tag}',
            'image': Image.open(os.path.join(self.root, sample['img_path'])).convert('RGB')
        }