
## Install

```bash
conda create -n subobjects_vlm python=3.9 -y

conda activate subobjects_vlm

```

Install pytorch: https://pytorch.org/

```bash
pip install -r requirements.txt
```


## Training

```bash

cd /home/dchenbs/workspace/subobjects-VLM
conda activate subobjects_vlm
CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node 1 --master_port 29500 train.py \
    --epoch 1 --batch_size 8 --gradient_accumulation_steps 2 \
    --dataset coco --dataset_root /share/datasets/coco2017 \
    --llm HuggingFaceTB/SmolLM-360M-Instruct \
    --visual_tokenizer_config configs/visual_tokenizer/patch_8_per_side_random.json \
    --max_visual_tokens 64 \
    --vlm_config      configs/vlm/vae.json \
    --trainer_config  configs/training/default.yaml \
    --lm_loss_weight 1 \
    --dataloader_num_workers 8


```


# options    

    --visual_tokenizer_config configs/visual_tokenizer/directsam_tiny.json \
    --visual_tokenizer_config configs/visual_tokenizer/patch_8_per_side_random.json \


    --lm_loss_weight 1 \
    --lm_loss_weight 1 --vm_loss_weight 0.5 --insert_queries \

    --llm gpt2 \
    --llm gpt2-medium \
    --llm gpt2-large \
    --llm gpt2-xl \
    # --llm configs/llm/phi3_small.json \
    # --llm microsoft/Phi-3-mini-128k-instruct \
    --llm HuggingFaceTB/SmolLM-135M \
    --llm HuggingFaceTB/SmolLM-360M-Instruct \


    --dataset imagenet --dataset_root /share/datasets/imagenet \
    --dataset clevr_caption --dataset_root /home/dchenbs/workspace/datasets/CLEVR_v1.0 \
    --dataset coco --dataset_root /share/datasets/coco2017 \
    --dataset image_paragraph_captioning --dataset_root /home/dchenbs/workspace/datasets/VisualGenome \

    --vlm_config      configs/vlm/vae.json \
