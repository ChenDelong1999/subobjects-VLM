import json
import os
import torch
from PIL import Image

class ImageParagraphCaptioning(torch.utils.data.Dataset):

    def __init__(self, root, split, user_tag='<|user|>\n', assistant_tag='<|assistant|>\n', image_tag='<|image|>', end_tag='<|endoftext|>\n', max_samples=None):
        self.dataset_name = 'VisualGenome'
        self.root = root
        self.split = split

        self.user_tag = user_tag
        self.assistant_tag = assistant_tag
        self.image_tag = image_tag
        self.end_tag = end_tag
        self.max_samples = max_samples
        self.max_text_tokens = 250
        
        with open(f'{root}/image_paragraph_captioning/paragraphs_v1.json', 'r') as pf:
            paragraphs = json.load(pf)

        with open(f'{root}/image_paragraph_captioning/{split}_split.json', 'r') as sf:
            splits = json.load(sf)

        self.samples = []
        for p in paragraphs:
            if p['image_id'] in splits:
                dir, img = p['url'].split('/')[-2:]
                img_path = os.path.join(dir, img)
                paragraph = p['paragraph'].strip().replace('  ', ' ')
                self.samples.append({
                    "image_path": f"{root}/{img_path}",
                    "text":f'{self.user_tag} Describe this image in detail. {self.image_tag}{self.end_tag}{self.assistant_tag}{paragraph}{self.end_tag}'
                })

    def __len__(self):
        if self.max_samples is not None and self.max_samples > 0:
            return self.max_samples
        else:
            return len(self.samples)
    
    def __getitem__(self, idx):
        sample =  self.samples[idx].copy()
        sample['image'] = Image.open(sample['image_path']).convert('RGB')
        sample.pop('image_path')
        return sample