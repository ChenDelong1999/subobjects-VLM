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
        # 'panoptic', 
        'directsam', 
        'superpixel'
        ]
    

    # - - - - - - - - - - - - - - - - - - - - 

    # llm_class = "smollm"
    # # # folder = "/private/home/delong/workspace/subobjects-VLM/runs/clevr_caption/SmolLM2-135M-Instruct-dinov2_small(384px)"
    # # folder = "/private/home/delong/workspace/subobjects-VLM/runs/clevr_caption/SmolLM2-135M-Instruct-vae(384px)"
    # # folder = "/private/home/delong/workspace/subobjects-VLM/runs/clevr_caption/SmolLM2-360M-Instruct-vae(384px)"
    # folder = "/private/home/delong/workspace/subobjects-VLM/runs/clevr_caption/SmolLM2-1_7B-Instruct-vae(384px)"
    # checkpoint_step = 4080

    # dataset = "clevr_caption"
    # dataset_root = "/private/home/delong/workspace/data/clevr-caption"
    # splits = ["val"]

    # - - - - - - - - - - - - - - - - - - - - 

    # llm_class = "smollm"
    # folder = "/private/home/delong/workspace/subobjects-VLM/runs/imagenet/SmolLM2-135M-Instruct-dinov2_small(768px)"
    # checkpoint_step = 5004

    # dataset = "imagenet"
    # dataset_root = "/datasets01/imagenet_full_size/061417"
    # splits = ["train", "val"]

    # - - - - - - - - - - - - - - - - - - - - 

    llm_class = "llama"
    folders = [
        # '/private/home/delong/workspace/subobjects-VLM/runs/pixmo_cap/Llama-3_2-1B-dinov2_small(768px)',
        '/private/home/delong/workspace/subobjects-VLM/runs/pixmo_cap/Llama-3_2-1B-clip_resnet50(768px)',
        '/private/home/delong/workspace/subobjects-VLM/runs/pixmo_cap/Llama-3_2-1B-clip_vit_tinyclip(224px)',
        '/private/home/delong/workspace/subobjects-VLM/runs/pixmo_cap/Llama-3_2-1B-dinov2_base(768px)',
        '/private/home/delong/workspace/subobjects-VLM/runs/pixmo_cap/Llama-3_2-1B-dinov2_small_reg(768px)',
        '/private/home/delong/workspace/subobjects-VLM/runs/pixmo_cap/Llama-3_2-1B-dinov2_small(256px)',
        '/private/home/delong/workspace/subobjects-VLM/runs/pixmo_cap/Llama-3_2-1B-dinov2_small(320px)',
        '/private/home/delong/workspace/subobjects-VLM/runs/pixmo_cap/Llama-3_2-1B-dinov2_small(384px)',
        '/private/home/delong/workspace/subobjects-VLM/runs/pixmo_cap/Llama-3_2-1B-dinov2_small(448px)',
        '/private/home/delong/workspace/subobjects-VLM/runs/pixmo_cap/Llama-3_2-1B-dinov2_small(512px)',
        '/private/home/delong/workspace/subobjects-VLM/runs/pixmo_cap/Llama-3_2-1B-dinov2_small(576px)',
        '/private/home/delong/workspace/subobjects-VLM/runs/pixmo_cap/Llama-3_2-1B-dinov2_small(640px)',
        '/private/home/delong/workspace/subobjects-VLM/runs/pixmo_cap/Llama-3_2-1B-dinov2_small(704px)',
        '/private/home/delong/workspace/subobjects-VLM/runs/pixmo_cap/Llama-3_2-1B-dinov2_small(832px)',
        '/private/home/delong/workspace/subobjects-VLM/runs/pixmo_cap/Llama-3_2-1B-dinov2_small(896px)',
        '/private/home/delong/workspace/subobjects-VLM/runs/pixmo_cap/Llama-3_2-1B-dinov2_small(960px)',
        '/private/home/delong/workspace/subobjects-VLM/runs/pixmo_cap/Llama-3_2-1B-dinov2_small(1024px)',
        '/private/home/delong/workspace/subobjects-VLM/runs/pixmo_cap/Llama-3_2-1B-in1k_mobilenetv3(768px)',
        '/private/home/delong/workspace/subobjects-VLM/runs/pixmo_cap/Llama-3_2-1B-in22k_convnext(768px)',
        '/private/home/delong/workspace/subobjects-VLM/runs/pixmo_cap/Llama-3_2-1B-rgb_pixel(768px)',
        '/private/home/delong/workspace/subobjects-VLM/runs/pixmo_cap/Llama-3_2-1B-zero_pixel(768px)',
        ]

    checkpoint_step = 8283
    
    dataset = "pixmo_cap"
    dataset_root = "/private/home/delong/workspace/data/pixmo-cap"
    splits = ["val"]

    # - - - - - - - - - - - - - - - - - - - - 

    # llm_class = "llama"
    # folder = "/private/home/delong/workspace/subobjects-VLM/runs/sharegpt4v/Llama-3_2-1B-dinov2_small(768px)"
    # checkpoint_step = 4870

    # dataset = "sharegpt4v"
    # dataset_root = "/private/home/delong/workspace/data/ShareGPT4V"

    # splits = [
    #     "sharegpt4v_instruct_gpt4-vision_cap100k.json",
    #     "share-captioner_coco_lcs_sam_1246k_1107.json"
    # ]

    # - - - - - - - - - - - - - - - - - - - - 
    for folder in folders:
        for tokenizer_family in tokenizer_families:
            tokenizer_folder = os.path.join(folder, tokenizer_family)
            checkpoints = os.listdir(tokenizer_folder)
            checkpoints.sort()
            for checkpoint in checkpoints:

                # if checkpoint not in [
                #     '1224-1558-superpixel_slic(25t-768px)',
                #     '1224-1558-superpixel_slic(36t-768px)',
                #     '1224-1558-superpixel_slic(49t-768px)',
                # ]:
                #     continue

                checkpoint_path = os.path.join(tokenizer_folder, checkpoint)
                if not os.path.isdir(checkpoint_path):
                    print(f"Skipping {checkpoint_path}")
                    continue
                for split in splits:

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
                    print(checkpoint_path)

                #     break 
                # break 