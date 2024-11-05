import os
import sys
import numpy as np
import torch
from PIL import Image

from .sam import sam_post_processing

import sys
sys.path.insert(0, 'visual_tokenizer/MobileSAM/MobileSAMv2')

from mobilesamv2.promt_mobilesamv2 import ObjectAwareModel
from mobilesamv2 import sam_model_registry, SamPredictor

class MobileSAMv2Tokenizer:
    def __init__(
        self,
        checkpoint,
        image_resolution,
        max_tokens,
        device="cuda",
        weights_cache_dir="",
        object_conf=0.4,
        object_iou=0.9,
        num_box_prompts=320,
        **kwargs
    ):

        self.image_resolution = image_resolution
        self.max_tokens = max_tokens
        self.device = device
        self.object_conf = object_conf
        self.object_iou = object_iou
        self.num_box_prompts = num_box_prompts

        # Add MobileSAMv2 to the Python path
        sys.path.append(os.path.join(weights_cache_dir, 'mobilesam', 'MobileSAMv2'))

        # Set up paths to model weights
        encoder_paths = {
            'efficientvit_l2': os.path.join(weights_cache_dir, 'mobilesam', 'l2.pt'),
            'tiny_vit': os.path.join(weights_cache_dir, 'mobilesam', 'mobile_sam.pt'),
            'sam_vit_h': os.path.join(weights_cache_dir, 'mobilesam', 'sam_vit_h.pt')
        }

        prompt_guided_path = os.path.join(weights_cache_dir, 'mobilesam', 'Prompt_guided_Mask_Decoder.pt')
        obj_model_path = os.path.join(weights_cache_dir, 'mobilesam', 'ObjectAwareModel.pt')

        # Load the ObjectAwareModel
        self.mobilesam_objawaremodel = ObjectAwareModel(obj_model_path)

        # Load the PromptGuidedDecoder and image encoder
        PromptGuidedDecoder = sam_model_registry['PromptGuidedDecoder'](prompt_guided_path)
        image_encoder = sam_model_registry['sam_vit_h'](encoder_paths['sam_vit_h'])

        # Initialize MobileSAMv2 model
        self.mobilesamv2 = sam_model_registry['vit_h']()
        self.mobilesamv2.prompt_encoder = PromptGuidedDecoder['PromtEncoder']
        self.mobilesamv2.mask_decoder = PromptGuidedDecoder['MaskDecoder']
        self.mobilesamv2.image_encoder = image_encoder
        self.mobilesamv2.to(device=self.device).eval()

        # Initialize the predictor
        self.mobilesamv2_predictor = SamPredictor(self.mobilesamv2)

        print(f"MobileSAMv2 initialized on {self.device}")

    def load_images(self, images):
        if not isinstance(images, list):
            images = [images]
        processed_images = []
        for img in images:
            if isinstance(img, str):
                img = Image.open(img)
            elif isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            img = img.convert("RGB").resize((self.image_resolution, self.image_resolution))
            processed_images.append(img)
        return processed_images

    @torch.inference_mode()
    def __call__(self, images):
        images = self.load_images(images)
        batch_masks = np.zeros(
            (len(images), self.max_tokens, self.image_resolution, self.image_resolution),
            dtype=bool
        )

        for i, img in enumerate(images):
            # Obtain object detection results
            obj_results = self.mobilesam_objawaremodel(
                img,
                device=self.mobilesamv2.device,
                retina_masks=True,
                imgsz=self.image_resolution,
                conf=self.object_conf,
                iou=self.object_iou,
                verbose=False
            )

            # Set the image for the predictor
            self.mobilesamv2_predictor.set_image(np.array(img))

            if obj_results is None or len(obj_results) == 0:
                # No detections, create an empty mask
                mask = torch.zeros((1, self.image_resolution, self.image_resolution))
            else:
                input_boxes = obj_results[0].boxes.xyxy.cpu().numpy()
                input_boxes = self.mobilesamv2_predictor.transform.apply_boxes(
                    input_boxes, self.mobilesamv2_predictor.original_size
                )
                input_boxes = torch.from_numpy(input_boxes).to(self.device)

                num_boxes = input_boxes.shape[0]
                image_embedding = self.mobilesamv2_predictor.features.repeat(num_boxes, 1, 1, 1)
                prompt_embedding = self.mobilesamv2.prompt_encoder.get_dense_pe().repeat(num_boxes, 1, 1, 1)

                # Function to iterate over batches
                def batch_iterator(batch_size, *args):
                    n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
                    for b in range(n_batches):
                        yield [arg[b * batch_size : (b + 1) * batch_size] for arg in args]

                sam_masks = []
                batch_size = 64  # Adjust as necessary
                for (boxes_batch,) in batch_iterator(batch_size, input_boxes):
                    batch_size_current = boxes_batch.shape[0]
                    img_emb_batch = image_embedding[:batch_size_current]
                    prompt_emb_batch = prompt_embedding[:batch_size_current]

                    sparse_embeddings, dense_embeddings = self.mobilesamv2.prompt_encoder(
                        points=None,
                        boxes=boxes_batch,
                        masks=None,
                    )
                    low_res_masks, _ = self.mobilesamv2.mask_decoder(
                        image_embeddings=img_emb_batch,
                        image_pe=prompt_emb_batch,
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                        simple_type=True,
                    )
                    low_res_masks = self.mobilesamv2_predictor.model.postprocess_masks(
                        low_res_masks, self.mobilesamv2_predictor.input_size, self.mobilesamv2_predictor.original_size
                    )
                    masks_batch = (low_res_masks > self.mobilesamv2.mask_threshold).float().squeeze(1)
                    sam_masks.append(masks_batch.cpu())

                if len(sam_masks) == 0:
                    mask = torch.zeros((1, self.image_resolution, self.image_resolution))
                else:
                    mask = torch.cat(sam_masks, dim=0)
            # Process masks
            masks = sam_post_processing(mask.numpy())
            # Limit the number of masks to max_tokens
            batch_masks[i, :min(len(masks), self.max_tokens)] = masks[:self.max_tokens]

        return torch.tensor(batch_masks)