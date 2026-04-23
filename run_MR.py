import os
import subprocess
import argparse

parser = argparse.ArgumentParser(description="Run the full material reasoning pipeline")
parser.add_argument("--scenes", nargs='+', default=['firewoods_sand_dark'], help="Scene names to process")
parser.add_argument("--factors", nargs='+', default=['2'], help="Downsample factors per scene")
parser.add_argument("--gpu_id", type=int, default=0, help="GPU device ID")
parser.add_argument("--data_base_path", default='data', help="Base path for input data")
parser.add_argument("--out_base_path", default='output', help="Base path for output")
parser.add_argument("--data_device", default='cuda', help="Device for data loading")
parser.add_argument("--extra_train_args", default='', help="Extra args for train_scene_PGSR.py")
parser.add_argument("--start_step", type=int, default=1, help="Step to start from (1-9), useful for resuming after interruption")
args = parser.parse_args()

env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

# Auto-load .env file if present
env_file = os.path.join(os.path.dirname(__file__) or '.', '.env')
if os.path.exists(env_file):
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, _, value = line.partition('=')
                env[key.strip()] = value.strip().strip('"').strip("'")

def run_step(step_num, total, description, cmd):
    if step_num < args.start_step:
        print(f"\n[Step {step_num}/{total}] Skipping: {description}")
        return
    print(f"\n[Step {step_num}/{total}] {description}: {cmd}")
    subprocess.run(cmd, shell=True, check=True, env=env)

for id, scene in enumerate(args.scenes):
    factor = args.factors[id] if id < len(args.factors) else args.factors[-1]
    data_path = f'{args.data_base_path}/{scene}'
    out_path = f'{args.out_base_path}/{scene}'
    config_file = f'./arguments/{scene}.yaml'

    total_steps = 10

    run_step(1, total_steps, "Training 3D Gaussians",
             f'python train_scene_PGSR.py -s {data_path} -m {out_path} '
             f'-r {factor} --data_device {args.data_device} '
             f'--densify_abs_grad_threshold 0.0002 '
             f'--config_file {config_file} {args.extra_train_args}')

    run_step(2, total_steps, "Extracting SAM masks",
             f'python extract_sam2_mask.py --image_root {data_path} '
             f'-s sam --downsample {factor} -v')

    run_step(3, total_steps, "Computing scales",
             f'python get_scale_PGSR.py --image_root {data_path} '
             f'--model_path {out_path} --config_file {config_file}')

    run_step(4, total_steps, "Training contrastive features",
             f'python train_contrastive_feature_PGSR.py -m {out_path} '
             f'--iterations 10000 --num_sampled_rays 1000 --config_file {config_file}')

    run_step(5, total_steps, "Rendering scene",
             f'python render_scene_PGSR.py -s {data_path} -m {out_path} --config_file {config_file}')

    run_step(6, total_steps, "Extracting part tree",
             f'python extract_part_tree.py -m {out_path} --config_file {config_file}')

    run_step(7, total_steps, "VLM material prediction",
             f'python vlm_predict.py --dataset_path {out_path}/gpt_input '
             f'--vlm gpttool')

    run_step(8, total_steps, "Visualizing material segmentation",
             f'python vis_material_seg.py -m {out_path} --config_file {config_file}')

    run_step(9, total_steps, "Extracting voxels",
             f'python extract_voxel.py -s {data_path} -m {out_path} '
             f'--config_file {config_file}')

    run_step(10, total_steps, "Consolidating outputs to data/",
             f'python consolidate_to_data.py -s {data_path} -m {out_path} '
             f'--config_file {config_file}')

    print(f"\nPipeline complete for scene: {scene}")
