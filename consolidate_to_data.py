"""
Step 10: Consolidate pipeline outputs into data/<scene> directory.
Copies selected files/directories from output/<scene> (and simulation/) to data/<scene>.
"""
import os
import shutil
import argparse
import yaml
from argparse import Namespace


def copy_item(src, dst, label):
    """Copy a file or directory from src to dst. Skip if src does not exist."""
    if not os.path.exists(src):
        print(f"  [SKIP] {label}: {src} not found")
        return False
    if os.path.isdir(src):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"  [OK]   {label}: {src} -> {dst}")
    else:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  [OK]   {label}: {src} -> {dst}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Consolidate output files into data/<scene>")
    parser.add_argument('-s', '--source_path', required=True, help='data/<scene> path')
    parser.add_argument('-m', '--model_path', required=True, help='output/<scene> path')
    parser.add_argument('--config_file', type=str, default='', help='YAML config file')
    args = parser.parse_args()

    data_path = args.source_path
    output_path = args.model_path

    # Load YAML config to get fire_sim_root
    fire_sim_root = None
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file) as f:
            cfg = yaml.safe_load(f)
        fire_sim_root = cfg.get('fire_sim_root')

    os.makedirs(data_path, exist_ok=True)

    print(f"Consolidating outputs to {data_path}")
    print(f"  Source (output): {output_path}")
    if fire_sim_root:
        print(f"  Source (sim):    {fire_sim_root}")

    # --- Directories from output/<scene> ---
    copy_item(os.path.join(output_path, "360"),         os.path.join(data_path, "360"),         "360")
    copy_item(os.path.join(output_path, "app_model"),   os.path.join(data_path, "app_model"),   "app_model")
    copy_item(os.path.join(output_path, "fg"),          os.path.join(data_path, "fg"),          "fg")
    copy_item(os.path.join(output_path, "point_cloud"), os.path.join(data_path, "point_cloud"), "point_cloud")

    # Flatten train/ours_30000/depth -> data/<scene>/depth
    copy_item(os.path.join(output_path, "train", "ours_30000", "depth"),
              os.path.join(data_path, "depth"), "depth")

    # Flatten train/ours_30000/renders_depth -> data/<scene>/renders_depth
    copy_item(os.path.join(output_path, "train", "ours_30000", "renders_depth"),
              os.path.join(data_path, "renders_depth"), "renders_depth")

    # --- sim_output from fire_sim_root ---
    if fire_sim_root:
        copy_item(fire_sim_root, os.path.join(data_path, "sim_output"), "sim_output")

    # --- Individual files from output/<scene> ---
    files_to_copy = [
        "material_grid.npy",
        "voxel_grid.npy",
        "mask_pts_in_grids.pth",
        "indices_pts_in_grids.pth",
        "cfg_args",
        "cameras.json",
        "input.ply",
    ]
    for fname in files_to_copy:
        copy_item(os.path.join(output_path, fname),
                  os.path.join(data_path, fname), fname)

    # Rewrite paths in cfg_args to use relative paths
    cfg_args_path = os.path.join(data_path, "cfg_args")
    if os.path.exists(cfg_args_path):
        with open(cfg_args_path, 'r') as f:
            content = f.read()
        ns = eval(content)
        ns_dict = vars(ns)
        # Convert source_path to relative: data/<scene>
        if 'source_path' in ns_dict:
            ns_dict['source_path'] = data_path
        # Convert model_path to point to data/<scene> (same as source_path)
        if 'model_path' in ns_dict:
            ns_dict['model_path'] = data_path
        with open(cfg_args_path, 'w') as f:
            f.write(repr(ns))
        print(f"  [OK]   cfg_args: rewrote source_path and model_path to '{data_path}'")

    print("\nConsolidation complete.")


if __name__ == "__main__":
    main()
