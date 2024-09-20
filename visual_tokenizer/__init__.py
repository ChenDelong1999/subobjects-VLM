from .directsam import DirectSAMTokenizer
from .patch import PatchTokenizer


def get_visual_tokenizer(**kwargs):
    if kwargs['type'] == 'directsam':
        return DirectSAMTokenizer(**kwargs)
    elif kwargs['type'] == 'patch':
        return PatchTokenizer(**kwargs)
    else:
        raise NotImplementedError