import torch
import torchvision
import numpy as np


class ImageNet(torch.utils.data.Dataset):

    def __init__(self, root, split='train', user_tag='<|user|>\n', assistant_tag='<|assistant|>\n', image_tag='<|image|>', end_tag='<|endoftext|>\n', max_samples=None):
        self.dataset = torchvision.datasets.ImageNet(root, split=split)
        self.class_names = [names[0] for names in self.dataset.classes]

        self.user_tag = user_tag
        self.assistant_tag = assistant_tag
        self.image_tag = image_tag
        self.end_tag = end_tag
        self.max_samples = max_samples
        self.max_text_tokens = 40

        self.index_mapping = np.random.permutation(len(self.dataset))

    def __len__(self):
        if self.max_samples is not None and self.max_samples > 0:
            return self.max_samples
        else:
            return len(self.dataset)
    
    def __getitem__(self, index):

        index = self.index_mapping[index]
        image, class_idx = self.dataset[index]
        # text = f'{self.user_tag} What is this: {self.image_tag}{self.end_tag}{self.assistant_tag} It\'s a {self.class_names[class_idx]}.{self.end_tag}'
        text = f'{self.user_tag}{self.image_tag}{self.end_tag}{self.assistant_tag}{self.class_names[class_idx]}{self.end_tag}'

        return {"text": text, "image": image.convert('RGB')}
