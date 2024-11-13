import submitit
import os
import datetime

class Trainer:
    def __init__(self, folder):
        self.cwd = "/private/home/delong/workspace/subobjects-VLM"
        self.conda_env_name = "subobjects"
        self.conda_path = "/private/home/delong/miniconda3"  # Path to your Conda installation
        self.folder = folder


    def __call__(self):
        import os
        import subprocess
        # Change to the working directory
        os.chdir(self.cwd)
        # Construct the training command
        cmd = f"""
        # Initialize Conda for the shell session
        source {self.conda_path}/etc/profile.d/conda.sh
        conda activate {self.conda_env_name}
        export MKL_THREADING_LAYER=GNU
        hash -r
        
        echo "Using python: $(which python)"
        echo "Python version: $(python --version)"
        echo "Using torchrun: $(which torchrun)"
        echo "Conda envs: $(conda env list)"
        
        python -m torch.distributed.run --nproc_per_node=8 train.py \\
            --epoch 1 --batch_size 1 --gradient_accumulation_steps 32 \\
            --dataset sharegpt4v --dataset_root '/private/home/delong/workspace/data/ShareGPT4V' \\
            --split 'share-captioner_coco_lcs_sam_1246k_1107.json' \\
            --llm HuggingFaceTB/SmolLM2-1.7B-Instruct \\
            --visual_embed_config configs/visual_embedding/clip_resnet50.json \\
            --max_visual_tokens 256 \\
            --visual_tokenizer_config configs/visual_tokenizer/patch/patch_16_per_side_raster.json \\
            --trainer_config configs/training/sharegpt4v_pt.yaml \\
            --embedding_input_resolution 448 \\
            --tokenizer_input_resolution 16 \\
            --dataloader_num_workers 8 \\
            --output_dir {self.folder}

        """
        # Execute the command
        subprocess.run(cmd, shell=True, check=True, executable="/bin/zsh")

if __name__ == "__main__":
    # Define the output folder with a timestamp
    folder = os.path.join(
        "runs", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    os.makedirs(folder, exist_ok=True)
    # Initialize the executor
    executor = submitit.AutoExecutor(folder=folder)
    # Update job parameters
    executor.update_parameters(
        mem_gb=512,
        gpus_per_node=8,
        cpus_per_task=80,
        nodes=1,
        timeout_min=4320,  # 72 hours
        slurm_partition="learnfair",
    )
    # Submit the job
    job = executor.submit(Trainer(folder))
    print(f"Submitted job with ID: {job.job_id}")

