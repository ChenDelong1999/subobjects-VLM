
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


### ShareGPT4V VLM

```bash
# Pretrain

cd /private/home/delong/workspace/subobjects-VLM
conda activate subobjects
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --nproc_per_node=4 --master_port 29500 train.py \
    --epoch 3 \
    --batch_size 1 \
    --gradient_accumulation_steps 32 \
    --dataset sharegpt4v \
    --dataset_root /private/home/delong/workspace/data/ShareGPT4V \
    --split share-captioner_coco_lcs_sam_1246k_1107.json \
    --llm HuggingFaceTB/SmolLM2-1.7B-Instruct \
    --visual_embed_config configs/visual_embedding/clip_convnext_all.json \
    --max_visual_tokens 81 \
    --visual_tokenizer_config configs/visual_tokenizer/patch/patch_9_per_side_random.json \
    --trainer_config configs/training/sharegpt4v_pt.yaml \
    --embedding_input_resolution 384 \
    --tokenizer_input_resolution 384 \
    --dataloader_num_workers 8 
```



```bash
 
# Tokenizers
    --max_visual_tokens 36 --visual_tokenizer_config configs/visual_tokenizer/directsam/directsam_tiny_dsa_100ep@0.1.json \
    
    --max_visual_tokens 36 --visual_tokenizer_config configs/visual_tokenizer/patch/patch_6_per_side_raster.json \
    --max_visual_tokens 64 --visual_tokenizer_config configs/visual_tokenizer/patch/patch_8_per_side_raster.json \
    --max_visual_tokens 144 --visual_tokenizer_config configs/visual_tokenizer/patch/patch_12_per_side_raster.json \
    --max_visual_tokens 256 --visual_tokenizer_config configs/visual_tokenizer/patch/patch_16_per_side_raster.json \
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

### VLM Evaluation

```bash
# allocate gpu
srun --gpus-per-node=1 --partition=learnfair --time=4320 --cpus-per-task 10 --mem 60G --pty /bin/zsh -l
```


```python
import os
import subprocess
folder = '/private/home/delong/workspace/subobjects-VLM/runs/sharegpt4v/SmolLM2-1_7B-Instruct'
checkpoints = os.listdir(folder)
checkpoints.sort()
print(len(checkpoints))

for i, checkpoint in enumerate(checkpoints):
    for split in ["sharegpt4v_instruct_gpt4-vision_cap100k.json", "share-captioner_coco_lcs_sam_1246k_1107.json"]:
        print(f'>>> Runing {i+1} / {len(checkpoints)} checkpoint: {checkpoint} on {split} split')
        cmd = f"""
cd /private/home/delong/workspace/subobjects-VLM
source /private/home/delong/miniconda3/etc/profile.d/conda.sh
conda activate subobjects
python eval.py \
    --dataset ShareGPT4V \
    --dataset_root /private/home/delong/workspace/data/ShareGPT4V \
    --split {split} \
    --num_samples 5000 \
    --model_checkpoint "{folder}/{checkpoint}/runs/checkpoint-4870" \
    --llm_class smollm
"""
        print(cmd)
        subprocess.run(cmd, shell=True, executable="/bin/zsh")

```

  --dataset_root /private/home/delong/workspace/data/ShareGPT4V \
  --split sharegpt4v_instruct_gpt4-vision_cap100k.json \
  --split share-captioner_coco_lcs_sam_1246k_1107.json \



### Holist Evaluation of Image Tokenization


```zsh
# single debug run

conda activate subobjects
cd /private/home/delong/workspace/subobjects-VLM/evaluation_intrinsic

CUDA_VISIBLE_DEVICES=0 python segmentation.py \
    --split "SA1B" \
    --tokenizer_config ../configs/visual_tokenizer/directsam/directsam_tiny_sa1b_2ep@0.25.json \
    --input_resolution 768 \
    --output_dir outputs/segmentation

```


```zsh
# batch parallel runs

conda activate subobjects
cd /private/home/delong/workspace/subobjects-VLM
python scripts/run_heit_inference.py

```

compute metrics

```
conda activate subobjects
cd /private/home/delong/workspace/subobjects-VLM/evaluation_intrinsic
python compute_metrics.py
```