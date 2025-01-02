#!/usr/bin/env python3
import os
import submitit
import subprocess

class MetricsComputer:
    def __init__(self, split, resolution, model_name):
        """
        Store parameters for the job, plus environment and path setup.
        """
        self.split = split
        self.resolution = resolution
        self.model_name = model_name

        # Adjust these to match your environment
        self.cwd = "/private/home/delong/workspace/subobjects-VLM"
        self.conda_env = "subobjects"
        self.conda_path = "/private/home/delong/miniconda3"

        # Optional: other arguments
        self.results_dir = "outputs/segmentation_results"
        self.output_dir = "outputs/segmentation_metrics_0102"
        self.tolerance_recall = 5
        self.tolerance_monosemanticity = 25
        self.max_eval = None  # or an integer

    def __call__(self):
        """
        This runs on the compute node.
        1) cd into self.cwd
        2) source conda
        3) call compute_metrics.py with the chosen arguments
        """
        os.chdir(self.cwd)

        # Build command
        cmd = f"""
cd evaluation_intrinsic
source {self.conda_path}/etc/profile.d/conda.sh
conda activate {self.conda_env}
export MKL_THREADING_LAYER=GNU

python segmentation_metrics.py \\
  --split {self.split} \\
  --resolution {self.resolution} \\
  --model_name {self.model_name} \\
  --results_dir {self.results_dir} \\
  --output_dir {self.output_dir} \\
  --tolerance_recall {self.tolerance_recall} \\
  --tolerance_monosemanticity {self.tolerance_monosemanticity} \\
  {"--max_eval " + str(self.max_eval) if self.max_eval else ""}
"""
        print("[MetricsComputer] Running command:")
        print(cmd)
        subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")


if __name__ == "__main__":
    
    splits = [
        "ADE20k",
        # "SA1B",
        # "COCONut_relabeld_COCO_val",
        "PascalPanopticParts",
    ]
    resolutions = [
        # 384,
        768,
        # 1024,
        # 1500,
    ]

    log_folder_root = "evaluation_intrinsic/outputs/segmentation_metrics_logs"
    os.makedirs(log_folder_root, exist_ok=True)

    for split in splits:
        for res in resolutions:
            for model_name in os.listdir(f"evaluation_intrinsic/outputs/segmentation_results/{split}/{res}"):

                job_name = f"segmentation_metrics_{split}_{res}_{model_name}"
                log_folder = os.path.join(log_folder_root, split, str(res), model_name)
                os.makedirs(log_folder, exist_ok=True)

                executor = submitit.AutoExecutor(folder=log_folder)
                executor.update_parameters(
                    name=job_name,
                    mem_gb=60,
                    gpus_per_node=1,
                    cpus_per_task=10,
                    nodes=1,
                    timeout_min=4320,
                    slurm_partition="learnfair",
                    slurm_constraint="volta32gb",
                    slurm_exclude='learnfair7570,learnfair7009,learnfair7641,learnfair5103,learnfair5107,learnfair7031,learnfair7563,learnfair7058,learnfair7716,learnfair7717,learnfair7699,learnfair7698,learnfair7697,learnfair7672,learnfair7643,learnfair7673,learnfair6000,learnfair7639,learnfair7645,learnfair7646,learnfair7708,learnfair7711,learnfair7714,learnfair7015,learnfair7713,learnfair7691,learnfair7654,learnfair7667,learnfair7690,learnfair7666,learnfair7656,learnfair7709,learnfair7715,learnfair7636,learnfair7677,learnfair7678,learnfair7679,learnfair7729,learnfair7731,learnfair7728,learnfair7637',
                )

                # Create the job object
                job = executor.submit(MetricsComputer(split, res, model_name))
                print(f"Submitted job {job.job_id} for split={split}, resolution={res}, model_name={model_name}")
                print(f"Log file: {log_folder}")

        #         break 
        #     break 
        # break