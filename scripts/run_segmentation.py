#!/usr/bin/env python3
import os
import submitit
import subprocess
from datetime import datetime

class Segmenter:
    def __init__(self, split, input_resolution, tokenizer_config):
        """
        Holds parameters for the job, plus environment setup for where to run.
        """
        self.split = split
        self.input_resolution = input_resolution
        self.tokenizer_config = tokenizer_config

        # Adjust these paths/env as needed for your cluster:
        self.cwd = "/private/home/delong/workspace/subobjects-VLM"     # where your code lives
        self.conda_env = "subobjects"                                  # conda env name
        self.conda_path = "/private/home/delong/miniconda3"            # path to conda

        if '|' in tokenizer_config:
            self.tokenizer_config = tokenizer_config.split('|')[0]
            self.max_tokens = tokenizer_config.split('|')[1]
        else:
            self.max_tokens = 1024

    def __call__(self):
        """
        This is what will actually run on the compute node.
        We:
          1) cd into self.cwd
          2) source conda
          3) run segmentation.py with correct arguments
        """
        os.chdir(self.cwd)
        cmd = f"""

cd /private/home/delong/workspace/subobjects-VLM/evaluation_intrinsic
source {self.conda_path}/etc/profile.d/conda.sh
conda activate {self.conda_env}
export MKL_THREADING_LAYER=GNU
hash -r

echo "Using python: $(which python)"
echo "Python version: $(python --version)"

python segmentation.py \\
  --split {self.split} \\
  --tokenizer_config {self.tokenizer_config} \\
  --input_resolution {self.input_resolution} \\
  --max_tokens {self.max_tokens} \\
  --output_dir "outputs/segmentation_results"
"""
        print(f"Running: {cmd}")
        subprocess.run(cmd, shell=True, check=True, executable="/bin/zsh")


if __name__ == "__main__":
    
    config_root = "/private/home/delong/workspace/subobjects-VLM/configs/visual_tokenizer"

    splits = [
        # "SA1B", 
        # "COCONut_relabeld_COCO_val", 
        # "PascalPanopticParts", 
        "ADE20k"
    ]
    resolutions = [
        # 384, 
        # 768, 
        1024, 
        # 1500
        ]

    tokenizer_configs = []

    tokenizer_configs += [
        # "panoptic/panoptic_mask2former_tiny.json",
        # "panoptic/panoptic_mask2former_small.json",
        # "panoptic/panoptic_mask2former_base.json",
        # "panoptic/panoptic_mask2former_large.json",
        # "panoptic/panoptic_oneformer_tiny.json",
        # "panoptic/panoptic_oneformer_large.json",
        # "sam/fastsam.json",       # using ultralytics==8.0.120 
        "sam/mobilesamv2.json",     # using its own ultralytics
        # "sam/sam_vit_b.json",
        # "sam/sam_vit_l.json",
        # "sam/sam_vit_h.json",
        # "sam/sam_vit_h_48points.json",
        # "sam/sam_vit_h_64points.json",
        # "sam/sam_vit_h_64points_1layer.json",
    ]

    # for patch_per_side in range(2, 32):
    #     tokenizer_configs.append(f"patch/patch_{patch_per_side}_per_side_raster.json")

    # for superpixel_max_tokens in [100]:
    # # for superpixel_max_tokens in [4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256]:
    #     tokenizer_configs.append(f"superpixel/superpixel_slic.json|{superpixel_max_tokens}")

    # for size in ['tiny', 'large']:
    #     for directsam_threshold in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    #         tokenizer_configs.append(f"directsam/directsam_{size}_sa1b_2ep@{directsam_threshold}.json")

    print(f'splits: {splits}')
    print(f'resolutions: {resolutions}')
    print(f"Found {len(tokenizer_configs)} tokenizer configurations:")
    for tokenizer_config in tokenizer_configs:
        print(' -', tokenizer_config)

    input("Press Enter to continue...")

    for split in splits:
        for res in resolutions:
            for tok_cfg in tokenizer_configs:
                log_folder = f"/private/home/delong/workspace/subobjects-VLM/evaluation_intrinsic/outputs/segmentation_logs/{split}/{res}/{os.path.basename(tok_cfg).replace('.json', '')}"
                os.makedirs(log_folder, exist_ok=True)

                tok_cfg = os.path.join(config_root, tok_cfg)
                # Set up an AutoExecutor
                executor = submitit.AutoExecutor(folder=log_folder)
                executor.update_parameters(
                    name=f"segmentation-{split}-{res}-{os.path.basename(tok_cfg)}",
                    mem_gb=60,
                    gpus_per_node=int('patch' not in tok_cfg),
                    cpus_per_task=10,
                    nodes=1,
                    timeout_min=4320,
                    slurm_partition="learnfair",
                    slurm_constraint="volta32gb",
                    slurm_exclude='learnfair7570,learnfair7009,learnfair7641,learnfair5103,learnfair5107,learnfair7031,learnfair7563,learnfair7058,learnfair7716,learnfair7717,learnfair7699,learnfair7698,learnfair7697,learnfair7672,learnfair7643,learnfair7673,learnfair6000,learnfair7639,learnfair7645,learnfair7646,learnfair7708,learnfair7711,learnfair7714,learnfair7015,learnfair7713,learnfair7691,learnfair7654,learnfair7667,learnfair7690,learnfair7666,learnfair7656,learnfair7709,learnfair7715,learnfair7636,learnfair7677,learnfair7678,learnfair7679,learnfair7729,learnfair7731,learnfair7728,learnfair7637',

                )

                job = executor.submit(Segmenter(split, res, tok_cfg))
                print(f"Submitted job {job.job_id}: split={split}, res={res}, tokenizer={os.path.basename(tok_cfg)}")

        #         break
        #     break
        # break