from .directsam import DirectSAMTokenizer
from .patch import PatchTokenizer
from .superpixel import SuperpixelTokenizer
from .panoptic import PanopticTokenizer
from .sam import SAMTokenizer

def get_visual_tokenizer(**kwargs):
    if kwargs['type'] == 'directsam':
        return DirectSAMTokenizer(**kwargs)
    elif kwargs['type'] == 'patch':
        return PatchTokenizer(**kwargs)
    elif kwargs['type'] == 'superpixel':
        return SuperpixelTokenizer(**kwargs)
    elif kwargs['type'] == 'panoptic':
        return PanopticTokenizer(**kwargs)
    elif kwargs['type'] == 'sam':
        return SAMTokenizer(**kwargs)
    else:
        raise NotImplementedError