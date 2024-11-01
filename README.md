
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


### Holist Evaluation of Image Tokenization


```zsh

datasets0=(
    "COCONut_relabeld_COCO_val"
    "EntitySeg"
    "ADE20k"
    "cityscapes"
)
datasets1=(
    "SA1B"
    "PascalPanopticParts"
    "PartImageNetPP"
    "SPIN"
)
datasets2=(
    "EgoHOS"
    "plantorgans"
    "MapillaryMetropolis"
    "NYUDepthv2"
)
datasets3=(
    "tcd"
    "FoodSeg103"
    "WireFrame"
    "ISAID"
)
datasets4=(
    "PhenoBench"
    "LIP"
    "SOBA"
    "CIHP"
)
datasets5=(
    "LoveDA"
    "SUIM"
    "MyFood"
    "DIS5K_DIS_VD"
)
datasets6=(
    "DUTS_TE"
    "Fashionpedia"
    "SeginW"
)
datasets7=(
    "LVIS"
    "PACO"
    "DRAM"
)

tokenizers=(
    # "patch/patch_2_per_side_raster.json"
    # "patch/patch_4_per_side_raster.json"
    # "patch/patch_8_per_side_raster.json"
    # "patch/patch_16_per_side_raster.json"

    # "superpixel/superpixel_slic.json"

    # "directsam/directsam_tiny_dsa_50ep.json"
    # "directsam/directsam_large_gen3_1023.json"

    # "panoptic/panoptic_mask2former_tiny.json"
    # "panoptic/panoptic_mask2former_small.json"
    # "panoptic/panoptic_mask2former_base.json"
    # "panoptic/panoptic_mask2former_large.json"

    # "panoptic/panoptic_oneformer_tiny.json"
    # "panoptic/panoptic_oneformer_large.json"

    # "sam/sam_vit_b.json"
    # "sam/sam_vit_l.json"
    # "sam/sam_vit_h.json"
    # "sam/sam_vit_h_48points.json"
    # "sam/sam_vit_h_64points.json"
    "sam/sam_vit_h_64points_1layer.json"
)

conda activate subobjects
cd /private/home/delong/workspace/subobjects-VLM/HEIT
clear

for dataset in $datasets7; do
    for tokenizer in $tokenizers; do

        CUDA_VISIBLE_DEVICES=7 python token_vs_contour_recall.py \
            --split $dataset \
            --tokenizer_config ../configs/visual_tokenizer/$tokenizer \
            --input_resolution 1024

    done
done


```