
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

### CLEVR Caption

```bash
cd /home/dchenbs/workspace/subobjects-VLM
conda activate subobjects_vlm
CUDA_VISIBLE_DEVICES=5 torchrun --nproc_per_node 1 --master_port 29512 train.py \
    --epoch 1 --batch_size 8 --gradient_accumulation_steps 1 \
    --dataset clevr_caption --dataset_root /home/dchenbs/workspace/datasets/CLEVR_v1.0 \
    --llm HuggingFaceTB/SmolLM-135M-Instruct \
    --visual_embed_config      configs/visual_embedding/rgb_pixel.json \
    --image_resolution 768 \
    --max_visual_tokens 36 --visual_tokenizer_config configs/visual_tokenizer/patch_6_per_side_raster.json \
    --trainer_config  configs/training/default.yaml \
    --dataloader_num_workers 10

```

### ShareGPT4V

```bash
cd /home/dchenbs/workspace/subobjects-VLM
conda activate subobjects_vlm
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node 2 --master_port 29503 train.py \
    --epoch 1 --batch_size 8 --gradient_accumulation_steps 1 \
    --dataset sharegpt4v --dataset_root /home/dchenbs/workspace/datasets/sharegpt4v/ShareGPT4V/share-captioner_coco_lcs_sam_1246k_1107.json \
    --llm HuggingFaceTB/SmolLM-1.7B-Instruct \
    --visual_embed_config configs/visual_embedding/vae.json \
    --image_resolution 1024 \
    --max_visual_tokens 36 --visual_tokenizer_config configs/visual_tokenizer/patch_6_per_side_raster.json \
    --trainer_config  configs/training/default.yaml \
    --dataloader_num_workers 10

```


```bash
    --dataset sharegpt4v --dataset_root /home/dchenbs/workspace/datasets/sharegpt4v/ShareGPT4V/sharegpt4v_instruct_gpt4-vision_cap100k.json \
    --dataset sharegpt4v --dataset_root /home/dchenbs/workspace/datasets/sharegpt4v/ShareGPT4V/share-captioner_coco_lcs_sam_1246k_1107.json \
    --dataset sharegpt4v --dataset_root /home/dchenbs/workspace/datasets/sharegpt4v/ShareGPT4V/sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json \
```

## Training @ CPFS

```bash

cd /cpfs/shared/research-llm/liujianfeng/08_subobject/subobjects-VLM
conda activate subobjects_vlm
CUDA_VISIBLE_DEVICES=2,7 torchrun --nproc_per_node 2 --master_port 29500 train.py \
    --epoch 10 --batch_size 8 --gradient_accumulation_steps 8 \
    --dataset imagenet --dataset_root ../data/OpenDataLab___ImageNet-1K/raw/ImageNet-1K \
    --llm HuggingFaceTB/SmolLM-1.7B-Instruct \
    --visual_embed_config      configs/visual_embedding/rgb_pixel.json \
    --max_visual_tokens 64 --visual_tokenizer_config configs/visual_tokenizer/directsam_tiny.json \
    --trainer_config  configs/training/cpfs.yaml \
    --dataloader_num_workers 10

```


```bash
 
# Tokenizers
    --max_visual_tokens 16 --visual_tokenizer_config configs/visual_tokenizer/directsam_tiny.json \
    --max_visual_tokens 16 --visual_tokenizer_config configs/visual_tokenizer/patch_4_per_side_random.json \

    --max_visual_tokens 36 --visual_tokenizer_config configs/visual_tokenizer/directsam_tiny.json \
    --max_visual_tokens 36 --visual_tokenizer_config configs/visual_tokenizer/patch_6_per_side_random.json \
    
    --max_visual_tokens 64 --visual_tokenizer_config configs/visual_tokenizer/directsam_tiny.json \
    --max_visual_tokens 64 --visual_tokenizer_config configs/visual_tokenizer/patch_8_per_side_random.json \
    
    --max_visual_tokens 256 --visual_tokenizer_config configs/visual_tokenizer/directsam_tiny.json \
    --max_visual_tokens 256 --visual_tokenizer_config configs/visual_tokenizer/patch_16_per_side_random.json \


    --visual_tokenizer_config configs/visual_tokenizer/patch_8_per_side_random.json \
    --visual_tokenizer_config configs/visual_tokenizer/patch_6_per_side_random.json \
    --visual_tokenizer_config configs/visual_tokenizer/patch_16_per_side_random.json \
    --visual_tokenizer_config configs/visual_tokenizer/patch_16_per_side_raster.json \

# Encoders
    --visual_embed_config configs/visual_embedding/vae.json \
    --visual_embed_config configs/visual_embedding/clip_resnet50.json \
    --visual_embed_config configs/visual_embedding/dinov2_small.json \
    --visual_embed_config configs/visual_embedding/convnext_in22k_stage2.json \
    --visual_embed_config configs/visual_embedding/rgb_pixel.json \

# LLMs
    --llm HuggingFaceTB/SmolLM-135M-Instruct \
    --llm HuggingFaceTB/SmolLM-360M-Instruct \
    --llm HuggingFaceTB/SmolLM-1.7B-Instruct \

# Datasets

    --dataset imagenet --dataset_root /share/datasets/imagenet \
    --dataset coco --dataset_root /share/datasets/coco2017 \
    --dataset image_paragraph_captioning --dataset_root /home/dchenbs/workspace/datasets/VisualGenome \
```