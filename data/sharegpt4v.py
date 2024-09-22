import os
import torch
import json
import tqdm
from PIL import Image


dir_mapping = {
    'sam/images': '/home/dchenbs/workspace/datasets/sa1b',
    'coco/train2017': '/home/dchenbs/workspace/datasets/coco2017/images/train2017',
    'llava/llava_pretrain/images': '/home/dchenbs/workspace/datasets/sharegpt4v/LLaVA-Pretrain',
    'wikiart/images': '/home/dchenbs/workspace/datasets/sharegpt4v/WebData/wikiart/images',
    'share_textvqa/images': '/home/dchenbs/workspace/datasets/sharegpt4v/WebData/share_textvqa/images',
    'web-celebrity/images': '/home/dchenbs/workspace/datasets/sharegpt4v/WebData/web-celebrity/images',
        'Choi_Min-sik': 'Choi_Min_sik', 
        'Lee_Byung-hun': 'Lee_Byung_hun',
    'web-landmark/images': '/home/dchenbs/workspace/datasets/sharegpt4v/WebData/web-landmark/images',
    'vg/VG_100K': '/home/dchenbs/workspace/datasets/VisualGenome/VG_100K',
    'vg/VG_100K_2': '/home/dchenbs/workspace/datasets/VisualGenome/VG_100K_2',
    'gqa/images': '/home/dchenbs/workspace/datasets/gqa/images',
    'ocr_vqa/images': '/home/dchenbs/workspace/datasets/sharegpt4v/ocr_vqa/images',
    'textvqa/train_images': '/home/dchenbs/workspace/datasets/sharegpt4v/textvqa/train_images',
}
train_val_split={'train': 0.9, 'val': 0.1}

class ShareGPT4V(torch.utils.data.Dataset):

    def __init__(self, root, split='train', max_samples=None):
        
        self.max_text_tokens = 300
        self.max_samples = max_samples
        samples = json.load(open(root, 'r'))

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
        print(f'Total samples: {len(samples)}, after removing text-only samples: {len(self.samples)}')

        self.sam_dir_mapping = {}
        for i in range(51):
            files = os.listdir(os.path.join(dir_mapping['sam/images'], f"sa_{i:06}"))
            for file in files:
                if file.endswith('.jpg'):
                    self.sam_dir_mapping[file] = f"sa_{i:06}/{file}"

    def validate_exist(self, valid_img_paths=None):
        validated_samples = []
        not_exist = []
        not_exist_in_provided_list = []
        for i in tqdm.tqdm(range(len(self.samples))):
            try:
                self.__getitem__(i)
            except:
                not_exist.append(self.samples[i])
                continue
            
            if valid_img_paths is not None:
                if self.__getitem__(i, only_return_img_path=True) not in valid_img_paths:
                    not_exist_in_provided_list.append(self.samples[i])
                else:
                    validated_samples.append(self.samples[i])
            else:
                validated_samples.append(self.samples[i])

        self.samples = validated_samples
            
        print(f'Found {len(not_exist)} samples failed to load due to file not exist.')
        if len(not_exist_in_provided_list) > 0:
            print(f'Found {len(not_exist_in_provided_list)} samples failed to load due to file not exist in provided list.')
        print(f'After validation, {len(self.samples)} samples left.')

        return not_exist


    def process_sharegpt4v_sample(self, img_path, sample, image_tag='<|image|>', human_turn='<|user|>\n', gpt_turn='<|assistant|>\n', eos_token='<|endoftext|>\n'):
    # def __init__(self, root, split='train', user_tag='<|user|>\n', assistant_tag='<|assistant|>\n', image_tag='<|image|>', end_tag='<|endoftext|>\n', max_samples=None):
        text = ''
        for utterance in sample['conversations']:
            text += human_turn if utterance['from']=='human' else gpt_turn
            text += utterance['value'].replace('<image>', image_tag) + eos_token
        
        image = Image.open(img_path).convert('RGB')
        return {"text": text, "image": image.convert('RGB')}


    def __getitem__(self, index, only_return_img_path=False):
        sample = self.samples[index]
        img_path = sample['image']

        if 'sam/images' in img_path:
            relative_img_path = img_path.split('sam/images/')[-1]
            img_path = img_path.replace(relative_img_path, self.sam_dir_mapping[relative_img_path])

        for org, new in dir_mapping.items():
            img_path = img_path.replace(org, new)

        # img = Image.open(img_path)
        assert os.path.exists(img_path), f'Image not found: {img_path}'

        if only_return_img_path:
            return img_path
        else:
            return self.process_sharegpt4v_sample(img_path, sample)
    
    def __len__(self):
        if self.max_samples is not None:
            return min(len(self.samples), self.max_samples)
        else:
            return len(self.samples)