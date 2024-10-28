
## Install

```bash
conda create -n subobjects python=3.10 -y
# conda create -n subobjects python=3.9 -y

conda activate subobjects

```

Install pytorch: https://pytorch.org/

```bash
pip install -r requirements.txt
```


## Training @ 116

### CLEVR Caption

```bash
cd /home/dchenbs/workspace/subobjects-VLM
conda activate subobjects
CUDA_VISIBLE_DEVICES=5 torchrun --nproc_per_node 1 --master_port 29513 train.py \
    --epoch 1 --batch_size 8 --gradient_accumulation_steps 1 \
    --dataset clevr_caption --dataset_root /home/dchenbs/workspace/datasets/CLEVR_v1.0 \
    --llm HuggingFaceTB/SmolLM-135M-Instruct \
    --visual_embed_config      configs/visual_embedding/rgb_pixel.json \
    --image_resolution 768 \
    --max_visual_tokens 36 --visual_tokenizer_config configs/visual_tokenizer/directsam_0424.json \
    --trainer_config  configs/training/default.yaml \
    --dataloader_num_workers 10

```

### ShareGPT4V

```bash
cd /home/dchenbs/workspace/subobjects-VLM
conda activate subobjects
CUDA_VISIBLE_DEVICES=5 torchrun --nproc_per_node 1 --master_port 29505 train.py \
    --epoch 10 --batch_size 8 --gradient_accumulation_steps 4 \
    --dataset coco --dataset_root /share/datasets/coco2017 \
    --llm HuggingFaceTB/SmolLM-360M-Instruct \
    --visual_embed_config configs/visual_embedding/clip_resnet50.json \
    --image_resolution 1024 \
    --max_visual_tokens 36 --visual_tokenizer_config configs/visual_tokenizer/directsam_tiny.json \
    --trainer_config  configs/training/default.yaml \
    --dataloader_num_workers 8

```


```bash
    --dataset sharegpt4v --dataset_root /home/dchenbs/workspace/datasets/sharegpt4v/ShareGPT4V/sharegpt4v_instruct_gpt4-vision_cap100k.json \
    --dataset sharegpt4v --dataset_root /home/dchenbs/workspace/datasets/sharegpt4v/ShareGPT4V/share-captioner_coco_lcs_sam_1246k_1107.json \
    --dataset sharegpt4v --dataset_root /home/dchenbs/workspace/datasets/sharegpt4v/ShareGPT4V/sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json \
```

## Training @ CPFS

```bash

cd /private/home/delong/workspace/subobjects-VLM
conda activate subobjects
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port 29500 train.py \
    --epoch 10 --batch_size 1 --gradient_accumulation_steps 8 \
    --dataset imagenet --dataset_root /datasets01/imagenet_full_size/061417 \
    --llm HuggingFaceTB/SmolLM-135M-Instruct \
    --visual_embed_config      configs/visual_embedding/rgb_pixel.json \
    --max_visual_tokens 36 --visual_tokenizer_config configs/visual_tokenizer/directsam_b0.json \
    --trainer_config  configs/training/cpfs.yaml \
    --dataloader_num_workers 4
    
```


```bash
 
# Tokenizers
    --max_visual_tokens 16 --visual_tokenizer_config configs/visual_tokenizer/directsam_tiny.json \
    --max_visual_tokens 16 --visual_tokenizer_config configs/visual_tokenizer/patch_4_per_side_raster.json \

    --max_visual_tokens 36 --visual_tokenizer_config configs/visual_tokenizer/directsam_tiny.json \
    --max_visual_tokens 36 --visual_tokenizer_config configs/visual_tokenizer/patch_6_per_side_raster.json \
    
    --max_visual_tokens 64 --visual_tokenizer_config configs/visual_tokenizer/directsam_tiny.json \
    --max_visual_tokens 64 --visual_tokenizer_config configs/visual_tokenizer/patch_8_per_side_raster.json \
    
    --max_visual_tokens 256 --visual_tokenizer_config configs/visual_tokenizer/directsam_tiny.json \
    --max_visual_tokens 256 --visual_tokenizer_config configs/visual_tokenizer/patch_16_per_side_raster.json \


    --visual_tokenizer_config configs/visual_tokenizer/patch_8_per_side_raster.json \
    --visual_tokenizer_config configs/visual_tokenizer/patch_6_per_side_raster.json \
    --visual_tokenizer_config configs/visual_tokenizer/patch_16_per_side_raster.json \
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