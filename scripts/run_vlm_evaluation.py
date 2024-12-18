import os
import submitit
import subprocess

class Evaluator:
    def __init__(self, llm_class, folder, checkpoint, checkpoint_step, dataset, dataset_root, split):
        self.llm_class = llm_class
        self.folder = folder
        self.checkpoint = checkpoint
        self.checkpoint_step = checkpoint_step
        self.dataset = dataset
        self.dataset_root = dataset_root
        self.split = split
        self.cwd = "/private/home/delong/workspace/subobjects-VLM"
        self.conda_env = "subobjects"
        self.conda_path = "/private/home/delong/miniconda3"

    def __call__(self):
        os.chdir(self.cwd)

        cmd = f"""
source {self.conda_path}/etc/profile.d/conda.sh
conda activate {self.conda_env}
export MKL_THREADING_LAYER=GNU
hash -r

python eval.py \
    --dataset {self.dataset} \
    --dataset_root {self.dataset_root} \
    --split {self.split} \
    --num_samples 5000 \
    --model_checkpoint "{self.folder}/{self.checkpoint}/runs/checkpoint-{self.checkpoint_step}" \
    --llm_class {self.llm_class}
"""
        print(f"Running evaluation for checkpoint: {self.checkpoint}, split: {self.split}")
        subprocess.run(cmd, shell=True, check=True, executable="/bin/zsh")

"""

    --dataset ShareGPT4V \
    --dataset_root /private/home/delong/workspace/data/ShareGPT4V \
    --dataset pixmo_cap \
    --dataset_root /private/home/delong/workspace/data/pixmo-cap \

"""

if __name__ == "__main__":
    
    splits = ["train", "val"]

    # - - - - - - - - - - - - - - - - - - - - 

    # llm_class = "llama"
    # folder = '/private/home/delong/workspace/subobjects-VLM/runs/pixmo_cap/Llama-3_2-1B'
    # checkpoint_step = 8283
    
    # dataset = "pixmo_cap"
    # dataset_root = "/private/home/delong/workspace/data/pixmo-cap"

    # - - - - - - - - - - - - - - - - - - - - 

    llm_class = "smollm"
    folder = "runs/clevr_caption/SmolLM2-360M-Instruct/panoptic"
    checkpoint_step = 4080

    dataset = "clevr_caption"
    dataset_root = "/private/home/delong/workspace/data/clevr-caption"

    # - - - - - - - - - - - - - - - - - - - - 

    # llm_class = "smollm"
    # folder = "/private/home/delong/workspace/subobjects-VLM/runs/sharegpt4v/SmolLM2-1_7B-Instruct"
    # checkpoint_step = 4870

    # dataset = "sharegpt4v"
    # dataset_root = "/private/home/delong/workspace/data/ShareGPT4V"

    # splits = [
    #     "sharegpt4v_instruct_gpt4-vision_cap100k.json",
    #     "share-captioner_coco_lcs_sam_1246k_1107.json"
    # ]

    # - - - - - - - - - - - - - - - - - - - - 

    checkpoints = os.listdir(folder)
    checkpoints.sort()
    for checkpoint in checkpoints:
        checkpoint_path = os.path.join(folder, checkpoint)
        if not os.path.isdir(checkpoint_path):
            continue
        for split in splits:

            executor = submitit.AutoExecutor(folder=os.path.join(checkpoint_path, 'eval'))
            executor.update_parameters(
                mem_gb=60,
                gpus_per_node=1,
                cpus_per_task=10,
                nodes=1,
                timeout_min=4320, 
                slurm_partition="learnfair",
                slurm_constraint='volta32gb'
            )

            job = executor.submit(Evaluator(llm_class, folder, checkpoint, checkpoint_step, dataset, dataset_root, split))
            print(f"Submitted job with ID: {job.job_id}")
            print(checkpoint)

        # break 