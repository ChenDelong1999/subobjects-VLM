

### VLM Training

```bash
# ImageNet
cd /home/dchenbs/workspace/subobjects-dev/VLM
conda activate subobject
CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node 1 --master_port 29500 main.py \
    --epoch 1 --batch_size 8 --gradient_accumulation_steps 4 \
    --dataset imagenet --dataset_root /share/datasets/imagenet \
    --llm HuggingFaceTB/SmolLM-1.7B-Instruct \
    --image_segmenter configs/segmenter/patch-16-shuffle.json --max_segments 256 --dataloader_num_workers 4 \
    --trainer_config  configs/training/default.yaml \
    --vlm_config      configs/vision/dino_small_768.json \
    --lm_loss_weight 1 --vm_loss_weight 0.5 --insert_queries


    --image_segmenter configs/segmenter/patch-16-shuffle.json --max_segments 256 --dataloader_num_workers 4 \
    --image_segmenter configs/segmenter/directsam.json --max_segments 256 --dataloader_num_workers 4 \

    --image_segmenter configs/segmenter/patch-10-shuffle.json --max_segments 100 --dataloader_num_workers 16 \
    --image_segmenter configs/segmenter/slic.json --max_segments 100 --dataloader_num_workers 16 \

# CLEVR
cd /home/dchenbs/workspace/subobjects-dev/VLM
conda activate subobject
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port 29504 main.py \
    --epoch 5 --batch_size 32 --gradient_accumulation_steps 1 \
    --dataset clevr_caption --dataset_root /home/dchenbs/workspace/datasets/CLEVR_v1.0 \
    --llm gpt2 \
    --image_segmenter configs/segmenter/patch-6-shuffle.json --max_segments 36 --dataloader_num_workers 16 \
    --trainer_config  configs/training/clevr.yaml \
    --vlm_config      configs/vision/vae_img384px_6x6token.json \
    --lm_loss_weight 1 --vm_loss_weight 1 --insert_queries


    --image_segmenter configs/segmenter/patch-6-shuffle.json --max_segments 36 --dataloader_num_workers 16 \
    --image_segmenter configs/segmenter/directsam.json --max_segments 36 --dataloader_num_workers 4 \

    --image_segmenter configs/segmenter/patch-10-shuffle.json --max_segments 100 --dataloader_num_workers 16 \
    --image_segmenter configs/segmenter/slic.json --max_segments 100 --dataloader_num_workers 16 \

# cd /home/dchenbs/workspace/subobjects-dev/VLM
# conda activate subobject
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port 29502 main.py \
#     --epoch 15 --batch_size 32 --gradient_accumulation_steps 1 \
#     --dataset imagenet --dataset_root /share/datasets/imagenet \
#     --max_segments 256 \
#     --llm gpt2-large \
#     --image_segmenter configs/segmenter/patch-8-16-shuffle.json \
#     --trainer_config  configs/training/default.yaml \
#     --vlm_config      configs/vision/vae_img384px_6x6token.json \
    # --vm_loss_weight 0.5 --insert_queries


# options    

    --llm gpt2 \
    --llm gpt2-medium \
    --llm gpt2-large \
    --llm gpt2-xl \
    # --llm configs/llm/phi3_small.json \
    # --llm microsoft/Phi-3-mini-128k-instruct \


    --llm HuggingFaceTB/SmolLM-135M \


    # total effective batch size = 128
        --batch_size 128  
            # gpt2, no queries
        --batch_size 64
            # gpt2, insert_queries

    --dataset imagenet --dataset_root /share/datasets/imagenet \
    --dataset clevr_caption --dataset_root /home/dchenbs/workspace/datasets/CLEVR_v1.0 \
    --dataset coco --dataset_root /share/datasets/coco2017 \
    --dataset image_paragraph_captioning --dataset_root /home/dchenbs/workspace/datasets/VisualGenome \

    --vlm_config configs/vision/vae_384px.json
    # --vlm_config configs/vision/dino_small_224.json
    # --vlm_config configs/vision/resnet50_layer2_384px.json
    # --vlm_config configs/vision/resnet50_layer3_768px.json
    # --vlm_config configs/vision/resnet50_layer4_512px.json

```


### VLM Evaluation


```bash
cd /home/dchenbs/workspace/subobjects-dev/VLM
conda activate subobject
CUDA_VISIBLE_DEVICES=1 python eval_imagenet.py \
    --checkpoint '/home/dchenbs/workspace/subobjects-dev/VLM/runs/0808-1534-imagenet-vae_img448px_6x6token-no-query-SmolLM-360M-patch-14-shuffle/checkpoint-6000' \
    --split 'val' --imagenet_root '/share/datasets/imagenet' --n_samples -1 \
    --image_segmenter configs/segmenter/patch-14-shuffle.json --max_segments 196
    # --image_segmenter configs/segmenter/directsam.json --max_segments 196

```


---

<div align="center">

## [Subobject-level Image Tokenization](https://arxiv.org/abs/2402.14327)

[Delong Chen (陈德龙)](https://chendelong.world/)
<img src="assets/hkust_logo.png" alt="Logo" width="8">, &nbsp; 
[Samuel Cahyawijaya](https://samuelcahyawijaya.github.io/)
<img src="assets/hkust_logo.png" alt="Logo" width="8">, &nbsp; 
[Jianfeng Liu (刘剑锋)](https://www.linkedin.com/in/jianfeng-liu-9539897b/) 
<img src="assets/xiaobing_logo.jpg" alt="Logo" width="10">, &nbsp; 

[Baoyuan Wang (王宝元)](https://sites.google.com/site/zjuwby/)
<img src="assets/xiaobing_logo.jpg" alt="Logo" width="10">, &nbsp; 
[Pascale Fung](https://pascale.home.ece.ust.hk/)
<img src="assets/hkust_logo.png" alt="Logo" width="8"> &nbsp; 

<img src="assets/hkust_logo.png" alt="Logo" width="10"> Hong Kong University of Science and Technology &nbsp; &nbsp; 
<img src="assets/xiaobing_logo.jpg" alt="Logo" width="15"> Xiaobing.AI


<!-- [[arXiv]](https://arxiv.org/abs/2402.14327)&nbsp;|&nbsp;
[[Github]](https://github.com/ChenDelong1999/subobjects) -->

</div>

![teaser](assets/teaser.png)


## Updates
- **2024/04/24**: We updated our paper with the Direct Segment Anything Model (DirectSAM), which efficiently generates comprehensive subobject segmentations with a single forward pass! Checkout our latest arXiv ([2402.14327v2](https://arxiv.org/abs/2402.14327v2)) and 🎬Demo Video on [YouTube](https://www.youtube.com/watch?v=tlNs7xUQ0x4) or [bilibili](https://www.bilibili.com/video/BV1yH4y11A7V3/). The pretrained DirectSAM model is released on HuggingFace: 🤗[DirectSAM-1800px-0424](https://huggingface.co/chendelong/DirectSAM-1800px-0424), and the training code is also available in this repo!


- **2024/02/22**: The first version of our paper is released on arXiv ([2402.14327](https://arxiv.org/abs/2402.14327)). Codes and models will be open-sourced at this repository.


## Direct Segment Anything Model (DirectSAM)

<div align="center">

🎬[Demo Video (YouTube)](https://www.youtube.com/watch?v=tlNs7xUQ0x4) | 🎬[Demo Video (bilibili)](https://www.bilibili.com/video/BV1yH4y1A7V3) | 🤗[DirectSAM-1800px-0424](https://huggingface.co/chendelong/DirectSAM-1800px-0424)

</div>


![DirectSAM visualizations](assets/DirectSAM_visualizations.jpg)

![DirectSAM qingming](assets/DirectSAM_qingming.jpg)


### Using DirectSAM

- Clone the repository 

    ```bash
    git clone https://github.com/ChenDelong1999/subobjects.git
    cd subobjects
    ```

- Install dependencies

    ```bash
    conda create -n subobject python=3.9 -y
    conda activate subobject
    pip install -r requirements.txt
    ```

- Run DirectSAM on an example image

    ```python
    import requests
    from PIL import Image
    from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor
    from utils import inference_single_image, visualize_direct_sam_result

    checkpoint = "chendelong/DirectSAM-1800px-0424"

    image_processor = AutoImageProcessor.from_pretrained(checkpoint, reduce_labels=True)
    model = AutoModelForSemanticSegmentation.from_pretrained(checkpoint).to('cuda').eval()

    url = "http://images.cocodataset.org/val2017/000000002149.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    probs = inference_single_image(image, image_processor, model, resolution=None, pyramid_layers=0)
    visualize_direct_sam_result(probs, image, threshold=0.25)
    ```

The `probs` is the predicted boundary probabilities of the image, which is an ndarray of shape (height, width) between 0 and 1. The `visualize_direct_sam_result` function will show visualizations using `matplotlib`, where the `threshold` controls the binarization of the boundary probabilities.

Quality of segmentation can be improved by increasing the input resolution and the number of pyramid layers. The above two groups of figures are generated using `resolution=3600`, `pyramid_layers=1`/`pyramid_layers=2`, and `threshold=0.03`.

Using half-precision `model.half()` can speed up the inference and reduce the GPU memory requirement.

### Training DirectSAM

We provide an example script to fine-tune DirectSAM on the [ADE20K dataset](https://huggingface.co/datasets/scene_parse_150). The implementation is based on 🤗 HuggingFace Trainer, please see [this blog](https://huggingface.co/docs/transformers/tasks/semantic_segmentation) for a detailed tutorial.

The following command will start a distributed training with 512x512 resolution input and half-precision training, which takes around 9GB memory per GPU. 

```bash
# ADE-20K
    --dataset_keys entityseg \
# ADE-20K
    --dataset_keys ade_20k \
# COCONut where coconut-b is not ready
    --dataset_keys coconut-s coconut-b \
# SAM-HQ Collection
    --dataset_keys COIFT DIS5K-DIS-TR DIS5K-DIS-VD DUTS-TE DUTS-TR ecssd fss_all HRSOD MSRA_10K ThinObject5K \

```


```bash
# cd DirectSAM
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 trainer.py \
    --dataset_keys entityseg \
    --input_resolution 1024 \
    --num_train_epochs 100 \
    --max_steps 20000 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2
```

The following figures compare the segmentation results of DirectSAM before and after the above finetuning on ADE20K.

![DirectSAM finetuning](assets/ade20k_finetuning_visualization.jpg)


### Citation

If you find this work useful, please consider citing:

```bibtex
@article{chen2024subobject,
  author       = {Delong Chen and
                  Samuel Cahyawijaya and
                  Jianfeng Liu and
                  Baoyuan Wang and
                  Pascale Fung},
  title        = {Subobject-level Image Tokenization},
  journal      = {CoRR},
  volume       = {abs/2402.14327},
  year         = {2024},
  url          = {https://doi.org/10.48550/arXiv.2402.14327},
  doi          = {10.48550/ARXIV.2402.14327},
  eprinttype    = {arXiv},
  eprint       = {2402.14327}
}
```
