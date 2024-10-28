import os
import torch
from PIL import Image
import numpy as np
import tqdm


class ImageNet(torch.utils.data.Dataset):
    def __init__(self, root, split='train', user_tag='<|user|>\n', assistant_tag='<|assistant|>\n', image_tag='<|image|>', end_tag='<|endoftext|>\n', max_samples=None):
        # Define root paths
        self.root = root
        self.split = split

        # Load labels from labels.txt
        if os.path.exists(os.path.join(root, 'labels.txt')):
            # ImageNet-1K
            self.data_path = os.path.join(root, split)
            self.class_names = self._load_class_names(os.path.join(root, 'labels.txt'), sep=',')
        else:
            # ImageNet-22K
            if split != 'train':
                print('Warning: ImageNet-22K does not have a validation set. Using training set instead')
            self.data_path = root
            self.class_names = self._load_class_names(os.path.join(root, 'words.txt'), sep='\t')

        # Find all image paths and corresponding labels
        self.image_paths, self.labels = self._load_images(self.data_path)

        # Shuffle indices for random sampling
        self.index_mapping = np.random.permutation(len(self.image_paths))

        # Tags and other configurations
        self.user_tag = user_tag
        self.assistant_tag = assistant_tag
        self.image_tag = image_tag
        self.end_tag = end_tag
        self.max_samples = max_samples
        self.max_text_tokens = 40

    def _load_class_names(self, label_file, sep):
        class_names = {}
        with open(label_file, 'r') as f:
            for line in f:
                folder_name, name = line.strip().split(sep)
                class_names[folder_name] = name
        return class_names

    def _load_images(self, data_path):
        image_paths = []
        labels = []
        for class_folder in tqdm.tqdm(os.listdir(data_path)):
            class_path = os.path.join(data_path, class_folder)
            if os.path.isdir(class_path) and class_folder in self.class_names:
                for img_file in os.listdir(class_path):
                    image_paths.append(os.path.join(class_path, img_file))
                    labels.append(class_folder)
        return image_paths, labels

    def __len__(self):
        if self.max_samples is not None and self.max_samples > 0:
            return min(self.max_samples, len(self.image_paths))
        return len(self.image_paths)

    def __getitem__(self, index):
        # Retrieve image and label by shuffled index
        index = self.index_mapping[index]
        image_path = self.image_paths[index]
        class_folder = self.labels[index]
        image = Image.open(image_path).convert('RGB')
        class_name = self.class_names[class_folder]

        # Construct text prompt
        text = f'{self.user_tag}{self.image_tag}{self.end_tag}{self.assistant_tag}{class_name}{self.end_tag}'

        return {"text": text, "image": image}