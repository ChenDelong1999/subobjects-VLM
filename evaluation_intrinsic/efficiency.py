#!/usr/bin/env python3

"""
clear

srun --gpus-per-node=1 --partition=learnfair --time=4320 --cpus-per-task 30 --mem 60G --constraint volta32gb --pty /bin/zsh -l

measure_efficiency.py

Revised version:
  - No warm-up.
  - Each process runs exactly `num_images` inferences on distinct SA1B samples.
  - Tracks GPU utilization & memory each iteration via nvitop.
  - Catches OOM in each process, logs it, and stops further processing in that process.
  - Saves comprehensive but simpler metrics (aggregated vs parallel throughput, FPS, etc.).

Usage Example Commands (you can keep using your tokenizer configs as before):

cd /private/home/delong/workspace/subobjects-VLM/evaluation_intrinsic
conda activate subobjects
python efficiency.py \
    --split "SA1B" \
    --tokenizer_config "../configs/visual_tokenizer/directsam/directsam_tiny_sa1b_2ep@0.5.json" \
    --input_resolution 768 \
    --min_nproc 1 --max_nproc 32 \
    --num_images 100 \
    --output_dir "outputs/efficiency_test/V100-30cpu-debug"

# More tokenizer configs you might test:
    --tokenizer_config "../configs/visual_tokenizer/directsam/directsam_tiny_sa1b_2ep@0.1.json" \
    
    --tokenizer_config "../configs/visual_tokenizer/panoptic/panoptic_mask2former_tiny.json" \
    --tokenizer_config "../configs/visual_tokenizer/panoptic/panoptic_mask2former_large.json" \
    --tokenizer_config "../configs/visual_tokenizer/panoptic/panoptic_oneformer_tiny.json" \
    --tokenizer_config "../configs/visual_tokenizer/panoptic/panoptic_oneformer_large.json" \
    
    --tokenizer_config "../configs/visual_tokenizer/sam/sam_vit_b_8points.json" \
    --tokenizer_config "../configs/visual_tokenizer/sam/sam_vit_b_32points.json" \
    --tokenizer_config "../configs/visual_tokenizer/sam/sam_vit_h.json" \
    --tokenizer_config "../configs/visual_tokenizer/sam/sam_vit_h_64points.json" \
    
    --tokenizer_config "../configs/visual_tokenizer/sam/fastsam.json" \
    --tokenizer_config "../configs/visual_tokenizer/sam/mobilesamv2.json" \
    
    --tokenizer_config "../configs/visual_tokenizer/superpixel/superpixel_slic.json" --max_tokens 64 \
    --tokenizer_config "../configs/visual_tokenizer/superpixel/superpixel_slic.json" --max_tokens 128 \
    --tokenizer_config "../configs/visual_tokenizer/superpixel/superpixel_slic.json" --max_tokens 256 \

"""


import os
import sys
import time
import json
import argparse
import tqdm
import torch
import multiprocessing as mp
from multiprocessing import Manager
from datasets import load_dataset
from nvitop import Device

# Adjust this import path according to your project layout.
sys.path.append("..")
from visual_tokenizer import get_visual_tokenizer


def process_inference(
    proc_id,
    images,
    tokenizer_config,
    image_resolution,
    max_tokens,
    return_dict
):
    """
    Worker function for each process:
      1) Loads the tokenizer on GPU.
      2) Iterates over the provided images (one per step).
      3) Records GPU utilization/memory each step, as well as total time, OOM flags, etc.
    """
    torch.cuda.set_device(0)  # If only one GPU, always "cuda:0"; adjust as needed if multiple GPUs.

    # Load the tokenizer config
    if isinstance(tokenizer_config, str):
        config_dict = json.load(open(tokenizer_config, 'r'))
    else:
        config_dict = tokenizer_config

    # Create the tokenizer
    device_name = "cuda"
    visual_tokenizer = get_visual_tokenizer(
        **config_dict,
        image_resolution=image_resolution,
        max_tokens=max_tokens,
        device=device_name
    )

    # Prepare GPU usage stats
    device = Device(0)
    mem_info = device.memory_info()
    init_memory_gb = mem_info.used / 1024**3
    gpu_util_samples = []
    gpu_mem_samples = []

    # Processing loop
    real_count = 0
    oom_flag = False
    t_start = time.time()

    for step_idx, img in enumerate(tqdm.tqdm(images, desc=f"Process {proc_id}")):
        # Sample GPU usage at each iteration
        mem_info = device.memory_info()
        usage = device.gpu_utilization()  # integer percent
        mem_used_gb = mem_info.used / 1024**3

        gpu_util_samples.append(usage)
        gpu_mem_samples.append(mem_used_gb)

        # Run inference
        try:
            _ = visual_tokenizer([img] * 10)  # e.g. batch of 10 copies, if desired
            real_count += 1
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[Process {proc_id}] CUDA out of memory at step {step_idx}!")
                oom_flag = True
                break
            else:
                raise e

    t_end = time.time()
    real_time = t_end - t_start

    # Final memory usage
    mem_info_final = device.memory_info()
    final_memory_gb = mem_info_final.used / 1024**3

    # Summaries
    if gpu_util_samples:
        avg_gpu_util_percent = float(sum(gpu_util_samples)) / len(gpu_util_samples)
        max_gpu_util_percent = float(max(gpu_util_samples))
    else:
        avg_gpu_util_percent = 0.0
        max_gpu_util_percent = 0.0

    if gpu_mem_samples:
        avg_gpu_mem_gb = float(sum(gpu_mem_samples)) / len(gpu_mem_samples)
        max_gpu_mem_gb = float(max(gpu_mem_samples))
    else:
        avg_gpu_mem_gb = 0.0
        max_gpu_mem_gb = 0.0

    return_dict[proc_id] = {
        "real_count": real_count * 10,  # 10 copies per step
        "real_time": real_time,
        "oom_flag": oom_flag,
        "init_memory_gb": init_memory_gb,
        "final_memory_gb": final_memory_gb,
        "avg_gpu_util_percent": avg_gpu_util_percent,
        "max_gpu_util_percent": max_gpu_util_percent,
        "avg_gpu_mem_gb": avg_gpu_mem_gb,
        "max_gpu_mem_gb": max_gpu_mem_gb
    }


def run_experiment(
    n_proc,
    dataset,
    tokenizer_config,
    image_resolution,
    max_tokens,
    num_images
):
    """
    Spawns `n_proc` child processes. Each process gets `num_images` from the dataset.
    Returns comprehensive metrics:
      - n_proc
      - sum_of_real_count
      - sum_of_real_time
      - parallel_time
      - aggregated_throughput (img/s) = total_images / sum_of_real_time
      - parallel_throughput (img/s) = total_images / parallel_time
      - fps (same as aggregated_throughput)
      - oom_happened
      - per_proc (dict of each process's stats)
    """
    manager = Manager()
    return_dict = manager.dict()

    # We need n_proc * num_images total from the dataset
    total_required = n_proc * num_images
    total_available = len(dataset)
    if total_available < total_required:
        raise ValueError(f"Dataset has only {total_available} images, but need {total_required}.")

    # Partition the dataset so each process has exactly num_images
    procs_data = []
    start_idx = 0
    for _ in range(n_proc):
        end_idx = start_idx + num_images
        sub_dataset = dataset[start_idx:end_idx]
        procs_data.append(sub_dataset)
        start_idx = end_idx

    # Launch
    procs = []
    t_start = time.time()
    for proc_id in range(n_proc):
        p = mp.Process(
            target=process_inference,
            args=(
                proc_id,
                procs_data[proc_id],   # sub-list of images
                tokenizer_config,
                image_resolution,
                max_tokens,
                return_dict
            ),
        )
        p.start()
        procs.append(p)

    # Wait for them all to finish
    for p in procs:
        p.join()
    t_end = time.time()

    # Summarize
    all_results = dict(return_dict)  # convert manager dict to normal
    sum_of_real_count = sum(r["real_count"] for r in all_results.values())
    sum_of_real_time = sum(r["real_time"] for r in all_results.values())
    parallel_time = t_end - t_start
    total_images = sum_of_real_count

    # Check if any had OOM
    oom_happened = any(r["oom_flag"] for r in all_results.values())

    if sum_of_real_time > 0:
        aggregated_throughput = total_images / sum_of_real_time
    else:
        aggregated_throughput = 0.0

    if parallel_time > 0:
        parallel_throughput = total_images / parallel_time
    else:
        parallel_throughput = 0.0

    # FPS is often the same as images/sec, so it's the same as aggregated_throughput
    fps = aggregated_throughput

    details = {
        "n_proc": n_proc,
        "sum_of_real_count": sum_of_real_count,
        "sum_of_real_time": sum_of_real_time,
        "parallel_time": parallel_time,
        "aggregated_throughput": aggregated_throughput,
        "parallel_throughput": parallel_throughput,
        "fps": fps,
        "oom_happened": oom_happened,
        "per_proc": all_results
    }
    return details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, required=True,
                        help="Split name in the chendelong/HEIT dataset, e.g. 'SA1B'.")
    parser.add_argument("--tokenizer_config", type=str, required=True,
                        help="Path to the tokenizer config JSON (or direct JSON).")
    parser.add_argument("--input_resolution", type=int, default=768,
                        help="Resolution for resizing images in the tokenizer.")
    parser.add_argument("--max_tokens", type=int, default=128,
                        help="Max tokens per image.")
    parser.add_argument("--min_nproc", type=int, default=1,
                        help="Minimum number of parallel processes.")
    parser.add_argument("--max_nproc", type=int, default=8,
                        help="Maximum number of parallel processes.")
    parser.add_argument("--output_dir", type=str, default="outputs/efficiency_test",
                        help="Directory to save JSON logs.")
    parser.add_argument("--num_images", type=int, default=10,
                        help="Number of images processed *per* process.")
    args = parser.parse_args()

    # Prepare output folder
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    # We'll just directly load the specified split from "chendelong/HEIT".
    print(f"Loading dataset split = {args.split} from chendelong/HEIT ...")
    dataset = load_dataset("chendelong/HEIT", split=args.split)

    # If you prefer to limit the dataset size (optional), you can do so:
    # But typically we rely on needing n_proc*num_images images total.
    # dataset = dataset.select(range(n_proc * num_images)) # Not necessary; we do a check inside run_experiment.

    # Convert dataset to list of PIL images
    # (The dataset sample is typically: {"image": <PIL>, ...})
    images_list = []
    for i in range(len(dataset)):
        sample = dataset[i]
        pil_img = sample["image"].convert("RGB")
        images_list.append(pil_img)

    # We'll store partial results after each n_proc
    results = {
        "args": vars(args),
        "data": {}
    }

    for n_proc in range(args.min_nproc, args.max_nproc + 1):
        try:
            print(f"==== Testing n_proc = {n_proc} ====")
            details = run_experiment(
                n_proc=n_proc,
                dataset=images_list,
                tokenizer_config=args.tokenizer_config,
                image_resolution=args.input_resolution,
                max_tokens=args.max_tokens,
                num_images=args.num_images
            )
            results["data"][n_proc] = details

            print(
                f" -> n_proc = {n_proc}, "
                f"sum_of_real_count = {details['sum_of_real_count']}, "
                f"sum_of_real_time = {details['sum_of_real_time']:.2f} s"
            )
            print(
                f" -> aggregated_throughput = {details['aggregated_throughput']:.2f} img/s, "
                f"parallel_throughput = {details['parallel_throughput']:.2f} img/s, "
                f"FPS = {details['fps']:.2f}"
            )
            if details["oom_happened"]:
                print(f"[WARNING] OOM happened for n_proc={n_proc}.")

            # Save partial results to disk
            out_name = f"efficiency_{args.split}_{os.path.basename(args.tokenizer_config).replace('.json','')}.json"
            if 'superpixel' in args.tokenizer_config:
                out_name = out_name.replace('.json', f'_{args.max_tokens}tokens.json')
            out_path = os.path.join(args.output_dir, out_name)

            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Partial results saved to {out_path}")

            # If OOM, stop
            if details["oom_happened"]:
                print(f"[OOM] Stopping at n_proc={n_proc} due to out-of-memory.")
                break

        except RuntimeError as e:
            # e.g. "CUDA out of memory"
            if "out of memory" in str(e).lower():
                print(f"[OOM] Stopping at n_proc={n_proc}.")
                break
            else:
                raise e

    print("All done.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # Safe for CUDA ops in subprocesses
    main()