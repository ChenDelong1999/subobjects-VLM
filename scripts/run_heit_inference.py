#!/usr/bin/env python3

import subprocess
import os
from multiprocessing.pool import ThreadPool

# List of datasets
datasets = [
    "COCONut_relabeld_COCO_val",
    "EntitySeg",
    "ADE20k",
    "cityscapes",
    "SA1B",
    "PascalPanopticParts",
    "PartImageNetPP",
    "SPIN",
    "EgoHOS",
    "plantorgans",
    "MapillaryMetropolis",
    "NYUDepthv2",
    "tcd",
    "FoodSeg103",
    "WireFrame",
    "ISAID",
    "PhenoBench",
    "LIP",
    "SOBA",
    "CIHP",
    "LoveDA",
    "SUIM",
    "MyFood",
    "DIS5K_DIS_VD",
    "DUTS_TE",
    "Fashionpedia",
    "SeginW",
    "LVIS",
    "PACO",
    "DRAM",
]

# List of tokenizers
tokenizers = [
    # "patch/patch_2_per_side_raster.json",
    # "patch/patch_4_per_side_raster.json",
    # "patch/patch_8_per_side_raster.json",
    # "patch/patch_16_per_side_raster.json",

    "superpixel/superpixel_slic.json",

    # "directsam/directsam_tiny_dsa_100ep@0.01.json",
    # "directsam/directsam_tiny_dsa_100ep@0.05.json",
    # "directsam/directsam_tiny_dsa_100ep@0.1.json",
    # "directsam/directsam_tiny_dsa_100ep@0.15.json",
    # "directsam/directsam_tiny_dsa_100ep@0.2.json",
    # "directsam/directsam_tiny_dsa_100ep@0.3.json",
    # "directsam/directsam_tiny_dsa_100ep@0.4.json",
    # "directsam/directsam_tiny_dsa_100ep@0.5.json",

    # "directsam/directsam_large_gen3_1023.json",

    # "panoptic/panoptic_mask2former_tiny.json",
    # "panoptic/panoptic_mask2former_small.json",
    # "panoptic/panoptic_mask2former_base.json",
    # "panoptic/panoptic_mask2former_large.json",

    # "panoptic/panoptic_oneformer_tiny.json",
    # "panoptic/panoptic_oneformer_large.json",

    # "sam/sam_vit_b.json",
    # "sam/sam_vit_l.json",
    # "sam/sam_vit_h.json",
    # "sam/sam_vit_h_48points.json",
    # "sam/sam_vit_h_64points.json",
    # "sam/sam_vit_h_64points_1layer.json",
    # "sam/sam_vit_h_64points_2layer.json",
    
    # "sam/fastsam.json",
    # "sam/mobilesamv2.json",
]

NUM_GPUS = 8
WORK_DIR = '/private/home/delong/workspace/subobjects-VLM/HEIT'

# Function to run a single job
def run_job(args):
    dataset, tokenizer, gpu_id, input_resolution = args
    env = os.environ.copy()

    command = f"""
    cd {WORK_DIR} && \
    CUDA_VISIBLE_DEVICES={str(gpu_id)}  python heit_inference.py \
        --split {dataset} \
        --tokenizer_config ../configs/visual_tokenizer/{tokenizer} \
        --input_resolution {input_resolution} \
        --output_dir outputs/tokenized_HEIT
    """

    # Run the command
    subprocess.run(command, shell=True, executable="/bin/zsh", env=env)

# Create list of jobs
jobs = []
gpu_id = 0

for dataset in datasets:
    for tokenizer in tokenizers:
        for input_resolution in [1024]:
            jobs.append((dataset, tokenizer, gpu_id, input_resolution))
            gpu_id = (gpu_id + 1) % NUM_GPUS

for job in jobs:
    print(job)

# Run jobs with ThreadPool
pool = ThreadPool(NUM_GPUS)
pool.map(run_job, jobs)
pool.close()
pool.join()
