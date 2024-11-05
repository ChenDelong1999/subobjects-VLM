#!/usr/bin/env python3

import subprocess
import os
from multiprocessing.pool import ThreadPool

# List of datasets
datasets = [
    # "COCONut_relabeld_COCO_val",
    # "EntitySeg",
    # "ADE20k",
    # "cityscapes",
    "SA1B",
    # "PascalPanopticParts",
    # "PartImageNetPP",
    # "SPIN",
    # "EgoHOS",
    # "plantorgans",
    # "MapillaryMetropolis",
    # "NYUDepthv2",
    # "tcd",
    # "FoodSeg103",
    # "WireFrame",
    # "ISAID",
    # "PhenoBench",
    # "LIP",
    # "SOBA",
    # "CIHP",
    # "LoveDA",
    # "SUIM",
    # "MyFood",
    # "DIS5K_DIS_VD",
    # "DUTS_TE",
    # "Fashionpedia",
    # "SeginW",
    # "LVIS",
    # "PACO",
    # "DRAM",
]

# List of tokenizers
tokenizers = [
    # "patch/patch_2_per_side_raster.json",
    # "patch/patch_4_per_side_raster.json",
    # "patch/patch_8_per_side_raster.json",
    # "patch/patch_16_per_side_raster.json",

    # "superpixel/superpixel_slic.json",

    # "directsam/directsam_tiny_sa1b_2ep.json",
    # "directsam/directsam_tiny_dsa_50ep.json",
    "directsam/directsam_tiny_dsa_75ep.json",

    # "directsam/directsam_large_sa1b_2ep.json",
    # "directsam/directsam_large_gen1_1008.json",
    # "directsam/directsam_large_gen2_1014.json",
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
]

# Number of GPUs / Max concurrent jobs
NUM_GPUS = 8

# Base command components
base_command = [
    'python', 'token_vs_contour_recall.py',
    '--input_resolution', '1024'
]

# Paths
WORK_DIR = '/private/home/delong/workspace/subobjects-VLM/HEIT'

# Function to run a single job
def run_job(args):
    dataset, tokenizer, gpu_id, threshold = args

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Activate conda environment and change directory
    command = f"""
    cd {WORK_DIR} && \
    python token_vs_contour_recall.py \
        --split {dataset} \
        --tokenizer_config ../configs/visual_tokenizer/{tokenizer} \
        --input_resolution 1024 \
        --output_dir outputs/token_vs_contour_recall/directsam_threshold_ablation \
        --threshold {threshold}
    """

    # Run the command
    subprocess.run(command, shell=True, executable="/bin/zsh", env=env)

# Create list of jobs
jobs = []
gpu_id = 0

for dataset in datasets:
    for tokenizer in tokenizers:
        for threshold in [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
            jobs.append((dataset, tokenizer, gpu_id, threshold))
            gpu_id = (gpu_id + 1) % NUM_GPUS

# Run jobs with ThreadPool
pool = ThreadPool(NUM_GPUS)
pool.map(run_job, jobs)
pool.close()
pool.join()