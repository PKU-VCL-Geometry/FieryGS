#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
"""Render PGSR scenes with volumetric fire: loads per-frame fuel/occupancy, updates Gaussians, calls render_fire."""

import json
import os
import sys
from pathlib import Path
from argparse import ArgumentParser, Namespace
from os import makedirs

import cv2
import imageio
import matplotlib.pyplot as plt
import noise
import numpy as np
import open3d as o3d
import torch
import yaml
from tqdm import tqdm

from arguments import ModelParams, PipelineParams
from gaussian_renderer import GaussianModel
from gaussian_renderer.renderer_with_fire import render_fire
from simulation.color_mapping import TemperatureToRGB
from simulation.fuel_temperature import FuelToTemperatureLUT
from scene import Scene
from utils.general_utils import safe_state
from utils.graphics_utils import fov2focal

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)

def create_voxel_grid(min_coords, max_coords, voxel_size=128):
    grid_size = (max_coords - min_coords) / voxel_size
    grid = torch.zeros((voxel_size, voxel_size, voxel_size), dtype=torch.int)
    return grid, grid_size


def expand_bounding_box(min_coords, max_coords, scale_factor=1.5):
    center = (min_coords + max_coords) / 2
    box = max_coords - min_coords
    size = max(box[0] * scale_factor, max(box[1] * scale_factor, box[2] * scale_factor))
    new_min_coords = center - size / 2
    new_max_coords = center + size / 2
    return new_min_coords, new_max_coords

def map_points_to_voxels(points, min_coords, _opacities, grid_size, voxel_grid):
    """Fill voxels containing ``points`` (opacity reserved for future filtering)."""
    normalized_points = (points - min_coords) / grid_size
    indices = normalized_points.long()
    indices = torch.clamp(indices, 0, voxel_grid.size(0) - 1)
    voxel_grid[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
    return voxel_grid

def print_voxel_statistics(voxel_grid):
    num_ones = torch.sum(voxel_grid == 1).item()
    total_voxels = voxel_grid.numel()
    ratio_ones = num_ones / total_voxels
    print(f"Number of voxels with value 1: {num_ones}")
    print(f"Total number of voxels: {total_voxels}")
    print(f"Ratio of voxels with value 1: {ratio_ones * 100:.2f}%")
    

def visualize_voxel_grid(voxel_grid, output_filename='voxel_grid.png'):
    x, y, z = np.where(voxel_grid == 1)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='r', marker='o', s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=30, azim=45)
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图像已保存为 {output_filename}")

def voxel_grid_to_point_cloud(voxel_grid, voxel_size, min_coords):
    x, y, z = np.where(voxel_grid.cpu() == 1)
    points = np.vstack((x, y, z)).T * voxel_size.cpu().numpy() + min_coords.cpu().numpy()
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    
    return point_cloud

def xyz_to_uv(xyz_map, K):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = xyz_map[..., 0]
    y = xyz_map[..., 1]
    z = xyz_map[..., 2]
    u = fx * (x / z) + cx
    v = fy * (y / z) + cy
    u = u.flatten()
    v = v.flatten()
    z = z.flatten()
    u_int = u.round().long()
    v_int = v.round().long()
    return u_int, v_int

def render_set(model_path, name, iteration, views, scene, gaussians, pipeline, background, app_model=None, max_depth=5.0, volume=None, use_depth_filter=False,
               query_tensor=None, temp2rgb_converter=None, frame_idx=-1, args=None, scale_ratio=0.12, fuel2temp=None, color_field=None):
    """Run `render_fire` for each view; save RGB, depth, and normals under the model output tree."""
    render_root = os.path.join("output", args.scene, "render_output")
    gts_path = os.path.join(render_root, "gt")
    render_path = os.path.join(render_root, "renders")
    depth_path = os.path.join(render_root, "depth")
    render_depth_path = os.path.join(render_root, "renders_depth")
    render_normal_path = os.path.join(render_root, "renders_normal")

    makedirs(gts_path, exist_ok=True)
    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(render_depth_path, exist_ok=True)
    makedirs(render_normal_path, exist_ok=True)

    GS_render_timing = []
    fire_render_timing = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        gt, _ = view.get_image()

        out = render_fire(view, gaussians, pipeline, background, app_model=app_model, query_tensor=query_tensor, temp2rgb_converter=temp2rgb_converter, args=args, scale_ratio=scale_ratio, add_sepc_light=args.add_sepc_light, fuel2temp=fuel2temp, color_field=color_field)

        GS_render_timing.append(out["GS_render_timing"])
        fire_render_timing.append(out["fire_render_timing"])

        rendering = out["render"].clamp(0.0, 1.0)
        _, H, W = rendering.shape

        depth = out["plane_depth"].squeeze()
        depth = depth.detach().cpu().numpy()
        np.save(os.path.join(depth_path, '{0:05d}'.format(idx) + ".npy"), depth)
        depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
        depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)
        
        normal = out["rendered_normal"].permute(1,2,0)
        normal = normal/(normal.norm(dim=-1, keepdim=True)+1.0e-8)
        normal = normal.detach().cpu().numpy()
        normal = ((normal+1) * 127.5).astype(np.uint8).clip(0, 255)

        if args.render_single:
            rendering_np = (rendering.permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
            os.makedirs(f"data/test_render/{args.scene}", exist_ok=True)
            cv2.imwrite(f'data/test_render/{args.scene}/{args.render_frame:04d}.png', rendering_np)
        elif args.is_render_360:
            rendering_np = (rendering.permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
            cv2.imwrite(os.path.join(render_path, '{0:05d}'.format(frame_idx) + ".png"), rendering_np)
            cv2.imwrite(os.path.join(render_depth_path, '{0:05d}'.format(frame_idx) + ".png"), depth_color)
            cv2.imwrite(os.path.join(render_normal_path, '{0:05d}'.format(frame_idx) + ".png"), normal)
        else:
            rendering_np = (rendering.permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
            cv2.imwrite(os.path.join(render_path, '{0:05d}'.format(frame_idx) + ".png"), rendering_np)
            cv2.imwrite(os.path.join(render_depth_path, '{0:05d}'.format(frame_idx) + ".png"), depth_color)
            cv2.imwrite(os.path.join(render_normal_path, '{0:05d}'.format(frame_idx) + ".png"), normal)
    
    avg_GS_render_timing = sum(GS_render_timing) / len(GS_render_timing)
    avg_fire_render_timing = sum(fire_render_timing) / len(fire_render_timing)
    return avg_GS_render_timing, avg_fire_render_timing


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool,
                 max_depth : float, voxel_size : float, num_cluster: int, use_depth_filter : bool):
    """Per-frame loop: fuel volume → `query_tensor`, occupancy → prune Gaussians, then `render_fire`."""
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=4.0*voxel_size,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

        args.dual_smoke = getattr(args, 'dual_smoke', False)
        args.sigma_a = getattr(args, 'sigma_a', 1.0)
        args.time_freez = getattr(args, 'time_freez', False)

        T_white = args.T_white
        T_max = args.T_max
        T_air = args.T_air
        device = "cuda"
        temp2rgb_converter = TemperatureToRGB(T_max=T_white, device=device)

        if args.is_render_360:
            from utils.camera_utils import generate_cam_path
            selected_cams = generate_cam_path(scene.getTrainCameras(), args.camera_params, args.sim_frames)
        else:
            selected_cams  = scene.getTrainCameras()[args.render_camera:(args.render_camera+1)]

        indices_pts_in_grids = torch.load(args.load_path_indices_pts_in_grids).to(device)
        mask_pts_in_grids = torch.load(args.load_path_mask_pts_in_grids).to(device)
        indices_pts_ingrids_for_inall = torch.nonzero(mask_pts_in_grids, as_tuple=False).squeeze()

        fuel2temp = FuelToTemperatureLUT(device=device)

        frame_start = 0
        frame_end = args.sim_frames

        all_frames = {}
        GS_render_timing = []
        fire_render_timing = []

        original_spec_intensity = float(getattr(args, 'spec_intensity', 1e-8))
        use_perturb = getattr(args, 'use_perturb', False)
        perturbation_sequence = []
        if use_perturb:
            avg_period = getattr(args, 'perturb_period', 5.0)
            amplitude = getattr(args, 'perturb_amplitude', 0.3)
            noise_scale = 1.0 / avg_period
            noise_offset = 42.0
            total_frames = frame_end - frame_start
            for f in range(total_frames):
                x = f * noise_scale + noise_offset
                noise_value = noise.pnoise1(x, octaves=3, persistence=0.5, lacunarity=2.0, repeat=1024, base=0)
                perturbation_sequence.append(amplitude * noise_value)

        for frame_idx in tqdm(range(frame_start, frame_end)):
            if args.render_single:
                frame_idx = args.render_frame

            if args.is_render_360:
                selected_cams_new = selected_cams[frame_idx:frame_idx+1]
            else:
                selected_cams_new = selected_cams

            all_frames[f"frame_{frame_idx:04d}"] = {
                "w2c": selected_cams_new[0].world_view_transform.transpose(1, 0).cpu().numpy().tolist(),
                "K": selected_cams_new[0].K.cpu().numpy().tolist()
            }
            xyz = gaussians.get_xyz

            if args.time_freez:
                data = np.load(f'{args.fire_sim_root}/fuel/fuel_{args.render_frame}.npz')
            else:
                data = np.load(f'{args.fire_sim_root}/fuel/fuel_{frame_idx}.npz')

            fuel_field = torch.from_numpy(data['arr_0']).to(device).float()
            fuel_field = fuel_field.unsqueeze(-1)
            query_tensor = fuel_field.permute(3, 2, 1, 0).unsqueeze(0)

            if args.time_freez:
                occ_data = np.load(f'{args.fire_sim_root}/occupancy/occupancy_{args.render_frame}.npz')
            else:
                occ_data = np.load(f'{args.fire_sim_root}/occupancy/occupancy_{frame_idx}.npz')
            occ_field = torch.from_numpy(occ_data['arr_0']).to(device)

            keep_roi = occ_field[indices_pts_in_grids[:, 0], indices_pts_in_grids[:, 1], indices_pts_in_grids[:, 2]]
            keep_mask = torch.ones(xyz.shape[0], dtype=torch.bool, device=xyz.device)
            corbon_threshold_up = 1.0
            corbon_threshold_down = 0.7
            keep_roi_bool = (keep_roi.abs() >= corbon_threshold_down) & (keep_roi.abs() <= corbon_threshold_up)
            keep_mask[indices_pts_ingrids_for_inall] = keep_roi_bool

            occ_carbon = torch.zeros(xyz.shape[0], device=xyz.device)
            occ_carbon[indices_pts_ingrids_for_inall] = keep_roi
            occ_carbon = occ_carbon.unsqueeze(-1).repeat(1, 3)

            gaussians.update_per_frame(keep_mask, corbon_threshold_down, occ_carbon)

            min_bound = torch.tensor(args.bounding_box['min'], dtype=torch.float32, device=device)
            voxel_size = args.voxel_grid['voxel_size']
            dims = args.voxel_grid['dims']
            new_xyz = gaussians.get_xyz
            grid_coords = (new_xyz - min_bound) / voxel_size
            grid_indices = torch.floor(grid_coords).long()
            
            valid_mask = (
                (grid_indices[:, 0] >= 0) & (grid_indices[:, 0] < dims[0]) &
                (grid_indices[:, 1] >= 0) & (grid_indices[:, 1] < dims[1]) &
                (grid_indices[:, 2] >= 0) & (grid_indices[:, 2] < dims[2])
            )

            valid_mask = valid_mask & mask_pts_in_grids

            indices_pts_in_grids = grid_indices[valid_mask]
            indices_pts_ingrids_for_inall = torch.nonzero(valid_mask, as_tuple=False).squeeze()

            if args.dual_smoke:
                if args.time_freez:
                    color_data = np.load(f'{args.fire_sim_root}/color/color_{args.render_frame}.npz')
                else:
                    color_data = np.load(f'{args.fire_sim_root}/color/color_{frame_idx}.npz')
                color_field = torch.from_numpy(color_data['arr_0']).to(device)
                color_field = color_field.permute(3, 2, 1, 0).unsqueeze(0)
            else:
                color_field = None

            if use_perturb:
                if args.render_single:
                    perturbation_idx = args.render_frame - frame_start
                else:
                    perturbation_idx = frame_idx - frame_start
                perturbation = perturbation_sequence[perturbation_idx] if 0 <= perturbation_idx < len(perturbation_sequence) else 0.0
                args.spec_intensity = float(original_spec_intensity * (1 + perturbation))
            else:
                args.spec_intensity = float(original_spec_intensity)

            GS_time, fire_time = render_set(dataset.model_path, "360", scene.loaded_iter, selected_cams_new, scene, gaussians, pipeline, background, 
                    max_depth=max_depth, volume=volume, use_depth_filter=use_depth_filter, query_tensor=query_tensor, 
                    temp2rgb_converter=temp2rgb_converter, frame_idx=frame_idx, args=args, scale_ratio=float(T_max / T_white), fuel2temp=fuel2temp, color_field=color_field)
            GS_render_timing.append(GS_time)
            fire_render_timing.append(fire_time)
            if args.render_single:
                break
        
        os.makedirs(f'data/camera_info/{args.scene}', exist_ok=True)
        if args.is_render_360:
            output_file = f'data/camera_info/{args.scene}/camera_info_360.json'
        else:
            output_file = f'data/camera_info/{args.scene}/camera_info_fix.json'
        with open(output_file, 'w') as f:
            json.dump(all_frames, f, indent=4)
        
        avg_GS_render_timing = sum(GS_render_timing) / len(GS_render_timing)
        avg_fire_render_timing = sum(fire_render_timing) / len(fire_render_timing)
        print(f"Average GS render time: {avg_GS_render_timing:.4f} seconds")
        print(f"Average fire render time: {avg_fire_render_timing:.4f} seconds")


def extract_fg(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    """Optional utility: vote foreground Gaussians from depth + image masks, then build voxel_grid / yaml (scene paths are hard-coded)."""
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        points = gaussians.get_xyz
        points_fg_num = torch.zeros((gaussians.get_xyz.size()[0], 2), dtype=torch.int32, device=points.device)
        ones = torch.ones((points.size()[0], 1), device=points.device)
        points_h = torch.cat([points, ones], dim=1)
        views = scene.getTrainCameras()
        for idx, view in enumerate(tqdm(views, desc="Extracting foreground")):
            camera_coords = (view.world_view_transform.T @ points_h.T).T
            depths = camera_coords[:, 2:3]
            depth_path = os.path.join(scene.model_path, "train", "ours_30000", "depth", '{0:05d}'.format(idx) + ".npy")
            depth_map = torch.tensor(np.load(depth_path), dtype=torch.float32, device=points.device)
            H, W = depth_map.shape
            fx = fov2focal(view.FoVx, W)
            fy = fov2focal(view.FoVy, H)
            cx = W / 2 - 0.5
            cy = H / 2 - 0.5
            K = torch.tensor([[fx, 0, cx],
                            [0, fy, cy],
                            [0, 0, 1]]).float().cuda()
            u, v = xyz_to_uv(camera_coords[:, :3], K)
            mask_valid = (u >= 0) & (u < depth_map.shape[1]) & (v >= 0) & (v < depth_map.shape[0])
            depth_values = depth_map[v.int()[mask_valid], u.int()[mask_valid]]
            depths_valid = depths[mask_valid].squeeze()
            visible_mask = (depths_valid <= depth_values) & (depths_valid >= 0)
            visible_mask_all = torch.zeros(points.size()[0], dtype=torch.bool, device=points.device)
            visible_mask_all[mask_valid] = visible_mask
            mask_path = os.path.join("/mnt/c/Users/20579/Desktop/tnx/codes/FireGaussian1/data/garden/fg_mask/bg", view.image_name + ".JPG")
            mask_image = torch.tensor(imageio.imread(mask_path), dtype=torch.float32, device=points.device)
            mask_values = mask_image[v.int()[visible_mask_all], u.int()[visible_mask_all]]
            fg_mask = mask_values > 128
            all_mask = torch.zeros(points.size()[0], dtype=torch.bool, device=points.device)
            all_mask[visible_mask_all] = fg_mask
            all_mask_ = torch.zeros(points.size()[0], dtype=torch.bool, device=points.device)
            all_mask_[visible_mask_all] = ~fg_mask
            points_fg_num[all_mask, 0] += 1
            points_fg_num[all_mask_, 1] += 1

        selected_points = (points_fg_num[:, 0] > points_fg_num[:, 1])
        print(selected_points.size())
        print(selected_points.sum())

        gaussians.crop_setup(selected_points)
        save_path = os.path.join(scene.model_path, "fg", "fg_opac_0_.ply")
        gaussians.save_ply(save_path)
        
        points = gaussians.get_xyz
        print(points.shape[0])
        # Manual scene AABB (edit for your dataset); overrides data-driven bbox.
        max_coords = torch.tensor([1.5, 2.45, -0.15], device=points.device)
        min_coords = torch.tensor([-2.1, -1.25, -3.38], device=points.device)
        print(f"Original Bounding Box: min={min_coords}, max={max_coords}")

        new_min_coords, new_max_coords = expand_bounding_box(min_coords, max_coords, scale_factor=args.scale_factor)
        print(f"Expanded Bounding Box: min={new_min_coords}, max={new_max_coords}")
        
        voxel_grid, grid_size = create_voxel_grid(new_min_coords, new_max_coords, voxel_size=args.voxel_size)
        print(f"Voxel Grid size: {voxel_grid.shape}, Grid size per voxel: {grid_size}")
        config = {
            "bounding_box": {
                "min": new_min_coords.tolist(),
                "max": new_max_coords.tolist(),
            },
            "voxel_grid": {
                "dims": list(voxel_grid.shape),
                "voxel_size": grid_size[0].item(),
            },
        }
        cfg_path = Path(os.path.join(scene.model_path, "simulation_voxel.yaml"))
        cfg_path.write_text(yaml.safe_dump(config, sort_keys=False))
        print(f"Config saved to {cfg_path.resolve()}")
        voxel_grid = map_points_to_voxels(points, new_min_coords, gaussians.get_opacity, grid_size, voxel_grid)
        
        print_voxel_statistics(voxel_grid)
        
        visualize_voxel_grid(voxel_grid=voxel_grid)
        
        point_cloud = voxel_grid_to_point_cloud(voxel_grid=voxel_grid, voxel_size=grid_size, min_coords=new_min_coords)
        voxel_path = os.path.join(scene.model_path, "voxel.ply")
        o3d.io.write_point_cloud(voxel_path, point_cloud)
        
        voxel_grid_path = os.path.join(scene.model_path, "voxel_grid.npy")
        np.save(voxel_grid_path,voxel_grid)
        voxel = np.load(voxel_grid_path)
        print(voxel.shape)


def load_yaml_config(yaml_path):
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)

def update_args_with_yaml(args, yaml_config):
    for key, value in yaml_config.items():
        if not hasattr(args, key):
            setattr(args, key, value)


if __name__ == "__main__":
    torch.set_num_threads(8)
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--max_depth", default=5.0, type=float)
    parser.add_argument("--num_cluster", default=1, type=int)
    parser.add_argument("--use_depth_filter", action="store_true")
    parser.add_argument("-v", "--voxel_size", type=int, default=128)
    parser.add_argument("--scale_factor", type=float, default=1.5)

    parser.add_argument("--config_file", type=str, help="Path to the YAML configuration file")

    args = get_combined_args(parser)

    if args.config_file:
        yaml_config = load_yaml_config(args.config_file)
        update_args_with_yaml(args, yaml_config)

    print("Rendering " + args.model_path)

    safe_state(args.quiet)
    print(f"multi_view_num {model.multi_view_num}")
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.max_depth, args.voxel_size, args.num_cluster, args.use_depth_filter)

