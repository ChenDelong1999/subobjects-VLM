

def get_visual_tokenizer(**kwargs):
    if kwargs['type'] == 'directsam':
        from .directsam import DirectSAMTokenizer
        return DirectSAMTokenizer(**kwargs)
    
    elif kwargs['type'] == 'patch':
        from .patch import PatchTokenizer
        return PatchTokenizer(**kwargs)
    
    elif kwargs['type'] == 'superpixel':
        from .superpixel import SuperpixelTokenizer
        return SuperpixelTokenizer(**kwargs)
    
    elif kwargs['type'] == 'panoptic':
        from .panoptic import PanopticTokenizer
        return PanopticTokenizer(**kwargs)
    
    elif kwargs['type'] == 'sam':
        from .sam import SAMTokenizer
        return SAMTokenizer(**kwargs)
    
    elif kwargs['type'] == 'fastsam':
        from .fastsam import FastSAMTokenizer 
        return FastSAMTokenizer(**kwargs)
    
    elif kwargs['type'] == 'mobilesamv2':
        from .mobilesamv2 import MobileSAMv2Tokenizer
        return MobileSAMv2Tokenizer(**kwargs)

    elif kwargs['type'] == 'efficientvit':
        from .efficientvit import EfficientViTTokenizer
        return EfficientViTTokenizer(**kwargs)
    
    else:
        raise NotImplementedError