import os
import torch
from PIL import Image
import random
import datasets

class PixmoDataset(torch.utils.data.Dataset):
    def __init__(self, root, split='train', user_tag='<|user|>\n', assistant_tag='<|assistant|>\n',
                 image_tag='<|image|>', end_tag='<|endoftext|>\n', max_samples=None):
        # Load the dataset annotations
        self.root = root
        self.annotations = datasets.load_dataset("allenai/pixmo-cap", split="train")
        total_samples = len(self.annotations)
        val_size = 10000  # Last 10k images for validation
        train_size = total_samples - val_size

        # Determine indices based on split
        if split == 'train':
            self.indices = list(range(train_size))
        elif split == 'val':
            self.indices = list(range(train_size, total_samples))
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'.")

        self.user_tag = user_tag
        self.assistant_tag = assistant_tag
        self.image_tag = image_tag
        self.end_tag = end_tag
        
        if max_samples is not None and max_samples > 0:
            self.indices = self.indices[:max_samples]
        
        self.total_samples = len(self.indices)
        self.max_text_tokens = 300
        
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        index = self.indices[idx]
        image_path = os.path.join(self.root, f"{index}.jpg")
        if not os.path.exists(image_path):
            # Skip if image does not exist
            return self.__getitem__((idx + 1) % self.total_samples)
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image at index {index}: {e}")
            return self.__getitem__((idx + 1) % self.total_samples)
        
        # Build the text
        sample = self.annotations[index]
        caption = sample['caption']
        text = f"{self.user_tag}{self.image_tag}{self.end_tag}{self.assistant_tag}{caption}{self.end_tag}"
        return {"text": text, "image": image}