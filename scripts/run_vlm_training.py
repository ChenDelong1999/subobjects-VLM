import submitit
import os
import datetime
import argparse
import yaml


class Trainer:
    def __init__(self, output_dir, config, seed=42):
        self.cwd = config.get("cwd", "/private/home/delong/workspace/subobjects-VLM")
        self.conda_env_name = config.get("conda_env_name", "subobjects")
        self.conda_path = config.get("conda_path", "/private/home/delong/miniconda3")
        self.output_dir = output_dir
        self.training_args = config.get("training_args", {})
        self.seed = seed

    def create_cmd(self):
        args = " \\\n".join(
            f"--{key} \"{value}\"" for key, value in self.training_args.items()
        )
        cmd = f"""
source {self.conda_path}/etc/profile.d/conda.sh
conda activate {self.conda_env_name}
export MKL_THREADING_LAYER=GNU
hash -r

echo "Using python: $(which python)"
echo "Python version: $(python --version)"
echo "Using torchrun: $(which torchrun)"
echo "Conda envs: $(conda env list)"

python -m torch.distributed.run --nproc_per_node=8 train.py --seed {self.seed} --output_dir "{self.output_dir}/runs" {args} 
        """
        print(cmd)
        return cmd

    def __call__(self):
        import os
        import subprocess
        os.chdir(self.cwd)
        cmd = self.create_cmd()
        subprocess.run(cmd, shell=True, check=True, executable="/bin/zsh")


def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Submitit Training Script")
    parser.add_argument(
        "--config", type=str, default="scripts/args.yaml", help="Path to the YAML config file"
    )
    parser.add_argument(
        "--partition", type=str, default="learnfair", help="Slurm partition to use"
    )
    parser.add_argument(
        "--timeout", type=int, default=4320, help="Timeout in minutes (default: 72 hours)"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42], help="Random seeds to use"
    )
    return parser.parse_args()


    
def get_run_output_dir(args, seed):

    llm = args['llm'].split('/')[-1].replace('.', '_')
    visual_embed_config = args['visual_embed_config'].split('/')[-1].replace('.json', '').replace('.', '_') + f"({args['embedding_input_resolution']}px)"
    visual_tokenizer_config = args['visual_tokenizer_config'].split('/')[-1].replace('.json', '').replace('.', '_')
    visual_tokenizer_family = args['visual_tokenizer_config'].split('/')[-2]

    description = 'runs/' + args['dataset'] + '/' + llm + '-' + visual_embed_config + '/' + visual_tokenizer_family + '/' 
    description += datetime.datetime.now().strftime("%m%d-%H%M")
    description += '-' + visual_tokenizer_config
    description += f"({args['max_visual_tokens']}t-{args['tokenizer_input_resolution']}px)"
    description += f"-seed={seed}" if seed != 42 else ''

    return description

if __name__ == "__main__":
    args = parse_args()

    # Submit the job
    for seed in args.seeds:

        # Load configuration
        config = load_config(args.config)
        output_dir = get_run_output_dir(config['training_args'], seed)
        os.makedirs(output_dir, exist_ok=True)

        # Initialize the executor
        executor = submitit.AutoExecutor(folder=output_dir)
        executor.update_parameters(
            mem_gb=512,
            gpus_per_node=8,
            cpus_per_task=80,
            nodes=1,
            timeout_min=args.timeout,
            slurm_partition=args.partition,
            # slurm_constraint='ampere80gb',
            slurm_constraint='volta32gb',
            slurm_exclude='learnfair7570,learnfair7009,learnfair7641,learnfair5103,learnfair5107,learnfair7031,learnfair7563,learnfair7058,learnfair7716,learnfair7717,learnfair7699,learnfair7698,learnfair7697,learnfair7672,learnfair7643,learnfair7673,learnfair6000,learnfair7639,learnfair7645,learnfair7646,learnfair7708,learnfair7711,learnfair7714,learnfair7015,learnfair7713,learnfair7691,learnfair7654,learnfair7667,learnfair7690,learnfair7666,learnfair7656,learnfair7709,learnfair7715,learnfair7636,learnfair7677,learnfair7678,learnfair7679,learnfair7729,learnfair7731,learnfair7728,learnfair7637',
        )
        job = executor.submit(Trainer(output_dir, config, seed))
        print(f"Submitted job with ID: {job.job_id}")
        print(f"Seed: {seed}")
        print(f'Output directory: {output_dir}')