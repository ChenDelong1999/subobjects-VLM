import os
import submitit
import subprocess

class Evaluator:
    def __init__(self, llm_class, tokenizer_folder, checkpoint, checkpoint_step, dataset, dataset_root, split):
        self.llm_class = llm_class
        self.folder = tokenizer_folder
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

if __name__ == "__main__":


    tokenizer_families = [
        'patch', 
        'panoptic', 
        'directsam', 
        'superpixel'
        ]
    

    # - - - - - - - - - - - - - - - - - - - - 

    # llm_class = "smollm"
    # # folder = "/private/home/delong/workspace/subobjects-VLM/runs/clevr_caption/SmolLM2-135M-Instruct-dinov2_small(384px)"
    # folder = "/private/home/delong/workspace/subobjects-VLM/runs/clevr_caption/SmolLM2-135M-Instruct-vae(384px)"
    # checkpoint_step = 4080

    # dataset = "clevr_caption"
    # dataset_root = "/private/home/delong/workspace/data/clevr-caption"
    # splits = ["train", "val"]

    # - - - - - - - - - - - - - - - - - - - - 

    # llm_class = "smollm"
    # folder = "/private/home/delong/workspace/subobjects-VLM/runs/imagenet/SmolLM2-135M-Instruct-dinov2_small(768px)"
    # checkpoint_step = 5004

    # dataset = "imagenet"
    # dataset_root = "/datasets01/imagenet_full_size/061417"
    # splits = ["train", "val"]

    # - - - - - - - - - - - - - - - - - - - - 

    # llm_class = "llama"
    # folder = '/private/home/delong/workspace/subobjects-VLM/runs/pixmo_cap/Llama-3_2-1B-dinov2_small(768px)'
    # checkpoint_step = 8283
    
    # dataset = "pixmo_cap"
    # dataset_root = "/private/home/delong/workspace/data/pixmo-cap"
    # splits = ["train", "val"]

    # - - - - - - - - - - - - - - - - - - - - 

    llm_class = "llama"
    folder = "/private/home/delong/workspace/subobjects-VLM/runs/sharegpt4v/Llama-3_2-1B-dinov2_small(768px)"
    checkpoint_step = 4870

    dataset = "sharegpt4v"
    dataset_root = "/private/home/delong/workspace/data/ShareGPT4V"

    splits = [
        "sharegpt4v_instruct_gpt4-vision_cap100k.json",
        "share-captioner_coco_lcs_sam_1246k_1107.json"
    ]

    # - - - - - - - - - - - - - - - - - - - - 
    for tokenizer_family in tokenizer_families:
        tokenizer_folder = os.path.join(folder, tokenizer_family)
        checkpoints = os.listdir(tokenizer_folder)
        checkpoints.sort()
        for checkpoint in checkpoints:
            checkpoint_path = os.path.join(tokenizer_folder, checkpoint)
            if not os.path.isdir(checkpoint_path):
                continue
            for split in splits:

                # if not (split == "val" and checkpoint in [
                #     "1224-1558-superpixel_slic(25t-768px)",
                #     "1224-1558-superpixel_slic(49t-768px)",
                #     "1224-1558-superpixel_slic(36t-768px)",
                #     ]):
                #     continue

                executor = submitit.AutoExecutor(folder=os.path.join(checkpoint_path, 'vlm_eval'))
                executor.update_parameters(
                    name=f"vlm_eval_{dataset}-{folder.split('/')[-1]}-{checkpoint}-{split}",
                    mem_gb=60,
                    gpus_per_node=1,
                    cpus_per_task=10,
                    nodes=1,
                    timeout_min=4320, 
                    slurm_partition="learnfair",
                    slurm_constraint='volta32gb'
                )

                job = executor.submit(Evaluator(llm_class, tokenizer_folder, checkpoint, checkpoint_step, dataset, dataset_root, split))
                print(f"Submitted job with ID: {job.job_id}")
                print(checkpoint)

            #     break 
            # break 