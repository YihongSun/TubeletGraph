import os, sys, argparse
import subprocess
from multiprocessing import Process

def run_command(cmd, description, env=None):
    """Run a command and handle errors (Single Process)."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    if env and 'CUDA_VISIBLE_DEVICES' in env:
        print(f"CUDA_VISIBLE_DEVICES: {env['CUDA_VISIBLE_DEVICES']}")
    print(f"{'='*60}")

    env = env or os.environ.copy()
    result = subprocess.run(cmd, capture_output=False, env=env)
    if result.returncode != 0:
        print(f"Error: {description} failed with return code {result.returncode}")
        sys.exit(1)

def run_worker(cmd, description, wid, gpus):
    """Run a single worker process."""
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpus[wid])
    run_command(cmd + ["--wid", str(wid)], f"{description} (WID={wid})", env=env)

def run_parallel(base_cmd, description, gpus):
    """Run command with multiple workers in parallel."""
    processes = []
    for wid in range(len(gpus)):
        p = Process(target=run_worker, args=(base_cmd, description, wid, gpus))
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
        if p.exitcode != 0:
            print(f"Error: Worker process failed with exit code {p.exitcode}")
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run TubeletGraph pipeline")
    parser.add_argument("-c", "--config", required=True, help="Config file path")
    parser.add_argument("-d", "--dataset", required=True, help="Dataset name")
    parser.add_argument("-s", "--split", required=True, help="Dataset split")
    parser.add_argument("-m", "--method", default="Ours", help="Method name for predictions")
    parser.add_argument("--gpus", type=int, nargs='+', default=[0], help="List of GPU IDs to use (e.g., --gpus 0 1 2 3)")
    parser.add_argument("--skip_vlm", action="store_true", help="Skip VLM step")
    
    args = parser.parse_args()
    args.num_workers = len(args.gpus)
    
    # Change to the directory containing TubeletGraph
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    
    # Step 1: Computing entities (region proposals)
    cmd = [
        "python3", "TubeletGraph/entity_segmentation/cropformer.py",
        "-c", args.config,
        "-d", args.dataset,
        "-s", args.split,
        "--num_workers", str(args.num_workers)
    ]
    run_parallel(cmd, "Computing entities (region proposals)", args.gpus)
    
    # Step 2: Computing tubelets
    cmd = [
        "python3", "TubeletGraph/tubelet/compute_tubelets_sam.py",
        "-c", args.config,
        "-d", args.dataset,
        "-s", args.split,
        "--num_workers", str(args.num_workers)
    ]
    run_parallel(cmd, "Computing tubelets", args.gpus)
    
    # Step 3: Computing semantic similarity
    tubelet_name = f"tubelets_{args.dataset}_cropformer"    # assuming cropformer is used
    cmd = [
        "python3", "TubeletGraph/semantic_sim/compute_sim_fcclip.py",
        "-c", args.config,
        "-d", args.dataset,
        "-s", args.split,
        "-t", tubelet_name,
        "--num_workers", str(args.num_workers)
    ]
    run_parallel(cmd, "Computing semantic similarity", args.gpus)
    
    # Step 4: Compute predictions (single worker only)
    cmd = [
        "python3", "TubeletGraph/get_prediction.py",
        "-c", args.config,
        "-d", args.dataset,
        "-s", args.split,
        "-m", args.method
    ]
    run_command(cmd, "Computing predictions")

    if args.skip_vlm:
        print("Skipping VLM step as per user request.")
        return

    # Step 5: Obtain state graph description (single worker only)
    pred_name = f"{args.dataset}-{args.split}-{args.method}"
    cmd = [
        "python3", "TubeletGraph/vlm/prompt_vlm.py",
        "-c", args.config,
        "-p", pred_name
    ]
    run_command(cmd, "Obtaining state graph description")


if __name__ == "__main__":
    main()