import torch
import random
import os
from torchvision.datasets import CocoCaptions

class CocoCaptionDataset(torch.utils.data.Dataset):

    def __init__(self, root, split, user_tag='<|user|>\n', assistant_tag='<|assistant|>\n', image_tag='<|image|>', end_tag='<|endoftext|>\n', max_samples=None):
        self.dataset = CocoCaptions(os.path.join(root, f'images/{split}2017'), os.path.join(root, f'annotations/captions_{split}2017.json'))
        self.user_tag = user_tag
        self.assistant_tag = assistant_tag
        self.image_tag = image_tag
        self.end_tag = end_tag
        self.max_samples = max_samples
        self.max_text_tokens = 60

    def __len__(self):
        if self.max_samples is not None and self.max_samples > 0:
            return self.max_samples
        else:
            return len(self.dataset)

    def __getitem__(self, index):
        image, captions = self.dataset[index]
        text = f'{self.user_tag} Provide a one-sentence caption for the provided image {self.image_tag}{self.end_tag}{self.assistant_tag}{random.choice(captions)}{self.end_tag}'

        return {"text": text, "image": image}
    