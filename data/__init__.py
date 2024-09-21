from .imagenet import ImageNet
from .coco import CocoCaptionDataset
from .image_paragraph_captioning import ImageParagraphCaptioning
from .clevr_caption import CLEVRCaption
from .sharegpt4v import ShareGPT4V


def get_dataset(dataset_name, dataset_root, split='train', max_samples=None):

    if dataset_name == 'imagenet':
        dataset = ImageNet(root=dataset_root, split=split, max_samples=max_samples)
    
    elif dataset_name == 'coco':
        dataset = CocoCaptionDataset(root=dataset_root, split=split, max_samples=max_samples)
    
    elif dataset_name == 'image_paragraph_captioning':
        dataset = ImageParagraphCaptioning(root=dataset_root, split=split, max_samples=max_samples)
    
    elif dataset_name == 'clevr_caption':
        dataset = CLEVRCaption(root=dataset_root, split=split, max_samples=max_samples)

    elif dataset_name == 'sharegpt4v':
        dataset = ShareGPT4V(root=dataset_root, split=split, max_samples=max_samples)
    
    else:
        raise NotImplementedError

    return dataset

