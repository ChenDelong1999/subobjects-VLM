
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

### ImageNet Classification
```bash
# V100 32G
cd /private/home/delong/workspace/subobjects-VLM
conda activate subobjects
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node 8 --master_port 29500 train.py \
    --epoch 10 --batch_size 16 --gradient_accumulation_steps 1 \
    --dataset imagenet --dataset_root /datasets01/imagenet_full_size/061417 \
    --llm HuggingFaceTB/SmolLM-360M-Instruct \
    --visual_embed_config      configs/visual_embedding/rgb_pixel.json \
    --max_visual_tokens 36 --visual_tokenizer_config configs/visual_tokenizer/patch/patch_6_per_side_raster.json \
    --trainer_config  configs/training/default.yaml \
    --image_resolution 384 \
    --dataloader_num_workers 8
    
```


### ShareGPT4V VLM

```bash
# Pretrain

cd /private/home/delong/workspace/subobjects-VLM
conda activate subobjects
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node 8 train.py \
    --epoch 1 --batch_size 1 --gradient_accumulation_steps 32 \
    --dataset sharegpt4v --dataset_root '/private/home/delong/workspace/data/ShareGPT4V' --split 'share-captioner_coco_lcs_sam_1246k_1107.json' \
    --llm HuggingFaceTB/SmolLM2-1.7B-Instruct \
    --visual_embed_config      configs/visual_embedding/clip_vit_l_14_336.json \
    --max_visual_tokens 36 --visual_tokenizer_config configs/visual_tokenizer/directsam/directsam_tiny_dsa_100ep.json \
    --trainer_config  configs/training/sharegpt4v_pt.yaml \
    --embedding_input_resolution 336 \
    --tokenizer_input_resolution 336 \
    --dataloader_num_workers 8
    
```





```bash
 
# Tokenizers
    --max_visual_tokens 36 --visual_tokenizer_config configs/visual_tokenizer/directsam/directsam_tiny_dsa_75ep.json \
    --max_visual_tokens 576 --visual_tokenizer_config configs/visual_tokenizer/patch/patch_24_per_side_raster.json \


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
    --llm HuggingFaceTB/SmolLM2-1.7B-Instruct \

    --llm HuggingFaceTB/SmolLM-135M-Instruct \
    --llm HuggingFaceTB/SmolLM-360M-Instruct \
    --llm HuggingFaceTB/SmolLM-1.7B-Instruct \

# Datasets

    --dataset imagenet --dataset_root /share/datasets/imagenet \
    --dataset coco --dataset_root /share/datasets/coco2017 \
    --dataset image_paragraph_captioning --dataset_root /home/dchenbs/workspace/datasets/VisualGenome \
```


### Holist Evaluation of Image Tokenization


```zsh
# single debug run

conda activate subobjects
cd /private/home/delong/workspace/subobjects-VLM/HEIT

CUDA_VISIBLE_DEVICES=0 python token_vs_contour_recall.py \
    --split "COCONut_relabeld_COCO_val" \
    --tokenizer_config ../configs/visual_tokenizer/sam/sam_vit_h_48points.json \
    --input_resolution 1024

```


```zsh
# batch parallel runs

conda activate subobjects
cd /private/home/delong/workspace/subobjects-VLM
python scripts/token_vs_contour_recall.py

```

