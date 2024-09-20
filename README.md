
## Install

```bash
conda create -n subobjects_vlm python=3.9 -y

conda activate subobjects_vlm

```

Install pytorch: https://pytorch.org/

```bash
pip install -r requirements.txt
```


## Training @ 116

```bash

cd /home/dchenbs/workspace/subobjects-VLM
conda activate subobjects_vlm
CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node 1 --master_port 29501 train.py \
    --epoch 3 --batch_size 8 --gradient_accumulation_steps 4 \
    --llm HuggingFaceTB/SmolLM-360M-Instruct \
    --dataset clevr_caption --dataset_root /home/dchenbs/workspace/datasets/CLEVR_v1.0 \
    --visual_tokenizer_config configs/visual_tokenizer/directsam_tiny.json \
    --max_visual_tokens 36 \
    --vlm_config      configs/vlm/convnext_in22k_stage2.json \
    --trainer_config  configs/training/default.yaml \
    --lm_loss_weight 1 \
    --dataloader_num_workers 8

```

## Training @ CPFS

```bash

cd /cpfs/shared/research-llm/liujianfeng/08_subobject/subobjects-VLM
conda activate subobjects_vlm
CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node 1 --master_port 29500 train.py \
    --epoch 1 --batch_size 8 --gradient_accumulation_steps 4 \
    --llm HuggingFaceTB/SmolLM-360M-Instruct \
    --dataset imagenet --dataset_root ../data/OpenDataLab___ImageNet-1K/raw/ImageNet-1K \
    --visual_tokenizer_config configs/visual_tokenizer/directsam_tiny.json \
    --max_visual_tokens 256 \
    --vlm_config      configs/vlm/convnext_in22k_stage2.json \
    --trainer_config  configs/training/default.yaml \
    --lm_loss_weight 1 \
    --dataloader_num_workers 8

```


```bash
 
# Tokenizers
    --visual_tokenizer_config configs/visual_tokenizer/directsam_tiny.json \
    --visual_tokenizer_config configs/visual_tokenizer/patch_8_per_side_random.json \
    --visual_tokenizer_config configs/visual_tokenizer/patch_6_per_side_random.json \
    --visual_tokenizer_config configs/visual_tokenizer/patch_16_per_side_random.json \

# Encoders
    --vlm_config      configs/vlm/convnext_in22k_stage2.json \
    --vlm_config      configs/vlm/vae.json \
    --vlm_config      configs/vlm/rgb_pixel.json \

# LLMs
    --llm HuggingFaceTB/SmolLM-135M \
    --llm HuggingFaceTB/SmolLM-360M-Instruct \

# Datasets
    --dataset imagenet --dataset_root /share/datasets/imagenet \
    --dataset clevr_caption --dataset_root /home/dchenbs/workspace/datasets/CLEVR_v1.0 \
    --dataset coco --dataset_root /share/datasets/coco2017 \
    --dataset image_paragraph_captioning --dataset_root /home/dchenbs/workspace/datasets/VisualGenome \

    --dataset imagenet --dataset_root ../data/OpenDataLab___ImageNet-1K/raw/ImageNet-1K \
```