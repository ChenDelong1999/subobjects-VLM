
## Install

conda create -n subobjects_vlm python=3.9 -y
conda activate subobjects_vlm

Install pytorch https://pytorch.org/

pip install -r requirements.txt


## Training

```bash

cd /home/dchenbs/workspace/subobjects-VLM
conda activate subobjects_vlm
CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node 1 --master_port 29500 train.py \
    --epoch 1 --batch_size 8 --gradient_accumulation_steps 1 \
    --dataset coco --dataset_root /share/datasets/coco2017 \
    --llm HuggingFaceTB/SmolLM-360M-Instruct \
    --trainer_config  configs/training/default.yaml \
    --vlm_config      configs/vlm/vae.json \
    --max_visual_tokens 128 \
    --lm_loss_weight 1 \
    --dataloader_num_workers 8


```


# options    


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
