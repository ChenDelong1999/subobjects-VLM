
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


## Training


```bash
# allocate gpu
srun --gpus-per-node=8 --partition=learnfair --time=4320 --cpus-per-task 80 --mem 512G --pty /bin/zsh -l
```

```bash
# V100 16G
cd /private/home/delong/workspace/subobjects-VLM
conda activate subobjects
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node 8 --master_port 29500 train.py \
    --epoch 10 --batch_size 8 --gradient_accumulation_steps 1 \
    --dataset imagenet --dataset_root /datasets01/imagenet_full_size/061417 \
    --llm HuggingFaceTB/SmolLM-135M-Instruct \
    --visual_embed_config      configs/visual_embedding/rgb_pixel.json \
    --max_visual_tokens 36 --visual_tokenizer_config configs/visual_tokenizer/directsam_b0.json \
    --trainer_config  configs/training/default.yaml \
    --dataloader_num_workers 10
    
```


```bash
 
# Tokenizers
    --max_visual_tokens 16 --visual_tokenizer_config configs/visual_tokenizer/directsam_b0.json \
    --max_visual_tokens 16 --visual_tokenizer_config configs/visual_tokenizer/patch_4_per_side_raster.json \

    --max_visual_tokens 36 --visual_tokenizer_config configs/visual_tokenizer/directsam_b0.json \
    --max_visual_tokens 36 --visual_tokenizer_config configs/visual_tokenizer/patch_6_per_side_raster.json \
    
    --max_visual_tokens 64 --visual_tokenizer_config configs/visual_tokenizer/directsam_b0.json \
    --max_visual_tokens 64 --visual_tokenizer_config configs/visual_tokenizer/patch_8_per_side_raster.json \
    
    --max_visual_tokens 256 --visual_tokenizer_config configs/visual_tokenizer/directsam_b0.json \
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