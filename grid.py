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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import open3d as o3d

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import  numpy as np

def depth_pcd2normal(xyz):
    hd, wd, _ = xyz.shape
    bottom_point = xyz[..., 2:hd, 1:wd - 1, :]
    top_point = xyz[..., 0:hd - 2, 1:wd - 1, :]
    right_point = xyz[..., 1:hd - 1, 2:wd, :]
    left_point = xyz[..., 1:hd - 1, 0:wd - 2, :]

    # Filter out zero-value regions before gradient computation
    mask = (bottom_point[..., 2] != 0) & (top_point[..., 2] != 0) & (right_point[..., 2] != 0) & (
                left_point[..., 2] != 0)
    left_to_right = right_point - left_point
    bottom_to_top = top_point - bottom_point

    # Set gradients of zero-value regions to 0 to avoid affecting normal computation
    left_to_right[~mask] = 0
    bottom_to_top[~mask] = 0

    xyz_normal = torch.cross(left_to_right, bottom_to_top, dim=-1)
    xyz_normal = torch.nn.functional.normalize(xyz_normal, p=2, dim=-1)
    xyz_normal = torch.nn.functional.pad(xyz_normal.permute(2, 0, 1), (1, 1, 1, 1), mode='constant').permute(1, 2, 0)
    return xyz_normal

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "normal")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(normal_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        # print(rendering.shape)
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        
        # # print(render(view, gaussians, pipeline, background)["depth"])
        # depth = render(view, gaussians, pipeline, background)["depth"]
        # depth_render = apply_depth_colormap(-render_pkg["depth"][0][...,None]).permute(2,0,1)
        # torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))
        # depth = depth.permute(1, 2, 0) # (3, h, w) -> (h, w, 3)
        # normal = depth_pcd2normal(depth).permute(2, 0, 1)
        # print(normal.shape)
        # normal = torch.nn.functional.normalize(normal, p=2, dim=0)
                
        # #     # transform to world space
        # c2w = (view.world_view_transform.T).inverse()
        # normal2 = c2w[:3, :3] @ normal.reshape(3, -1)
        # normal = normal2.reshape(3, *normal.shape[1:])
        # normal = (normal + 1.) / 2.
        
        # torchvision.utils.save_image(normal, os.path.join(normal_path, '{0:05d}'.format(idx) + ".png"))
        
        
        

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            from utils.camera_utils import generate_cam_path
            cam_traj = generate_cam_path(scene.getTrainCameras(), n_frames=400)
            # render_set(dataset.model_path, "traj_1", scene.loaded_iter, cam_traj, gaussians, pipeline, background)
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
             
def create_voxel_grid(min_coords, max_coords, voxel_size=128):
    # Compute the size of each voxel
    grid_size = (max_coords - min_coords) / voxel_size
    # Get the boundary range of the voxel grid
    grid = torch.zeros((voxel_size, voxel_size, voxel_size), dtype=torch.int)

    return grid, grid_size

def get_bounding_box(points):
    # Compute the minimum and maximum values of the point cloud along each axis
    min_coords = torch.min(points, dim=0).values  # Minimum value per dimension
    max_coords = torch.max(points, dim=0).values  # Maximum value per dimension
    return min_coords, max_coords

def expand_bounding_box(min_coords, max_coords, scale_factor=1.5):
    center = (min_coords + max_coords) / 2
    box = max_coords - min_coords
    size = max(box[0] * scale_factor, max(box[1] * scale_factor, box[2] * scale_factor))
    new_min_coords = center - size / 2
    new_max_coords = center + size / 2
    return new_min_coords, new_max_coords

def map_points_to_voxels(points, min_coords, opacities,  grid_size, voxel_grid):
    # points = points
    opacities = opacities.squeeze()
    print(opacities.min())
    print(opacities.max())
    points = points[opacities > 0.05]
    plt.figure(figsize=(8, 6))
    plt.hist(opacities.cpu().numpy(), bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Opacity Distribution', fontsize=14)
    plt.xlabel('Opacity Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True)

    plt.savefig('opacity', dpi=300, bbox_inches='tight')  # Save as PNG format, 300 DPI

    # Close the current figure to avoid excessive memory usage
    plt.close()
    normalized_points = (points - min_coords) / grid_size
    indices = normalized_points.long()
    indices = torch.clamp(indices, 0, voxel_grid.size(0) - 1)
    voxel_grid[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
    return voxel_grid

def print_voxel_statistics(voxel_grid):
    # Count the number of voxels with value 1 in the voxel grid
    num_ones = torch.sum(voxel_grid == 1).item()
    
    # Compute the total number of voxels in the grid
    total_voxels = voxel_grid.numel()
    
    # Compute the ratio of voxels with value 1
    ratio_ones = num_ones / total_voxels

    # Print results
    print(f"Number of voxels with value 1: {num_ones}")
    print(f"Total number of voxels: {total_voxels}")
    print(f"Ratio of voxels with value 1: {ratio_ones * 100:.2f}%")
    
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_voxel_grid(voxel_grid, output_filename='voxel_grid.png'):
    # Get coordinates of all voxels with value 1 in the grid
    x, y, z = np.where(voxel_grid == 1)

    # Create 3D figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot voxel points
    ax.scatter(x, y, z, c='r', marker='o', s=1)  # Red dots represent voxels with value 1

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set viewing angle
    ax.view_init(elev=30, azim=45)

    # Save figure to file
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')  # Save as PNG format, 300 DPI

    # Close the current figure to avoid excessive memory usage
    plt.close()

    print(f"Image saved as {output_filename}")

def voxel_grid_to_point_cloud(voxel_grid, voxel_size, min_coords):
    # Get coordinates of all voxels with value 1 in the voxel grid
    x, y, z = np.where(voxel_grid.cpu() == 1)
    # print(x.device, min_coords.device)
    
    # Compute the actual coordinates of each point (based on voxel grid and voxel size)
    points = np.vstack((x, y, z)).T * voxel_size.cpu().numpy() + min_coords.cpu().numpy()  # Restore to actual spatial coordinates

    
    # Create Open3D point cloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    
    return point_cloud

    
             
def grid(dataset : ModelParams, iteration : int, pipeline : PipelineParams, voxel_size : int, scale_factor : float):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        print(gaussians._xyz.shape)
        print(gaussians._opacity.shape)
        
        points = gaussians.get_xyz
        # points[..., -1] = -points[..., -1]
        min_coords, max_coords = get_bounding_box(points)
        #garden
        max_coords = torch.tensor([1.3, 1.5, 0.0], device=points.device)
        min_coords = torch.tensor([-2.3, -2.1, -3.5], device=points.device)
        #garden_8 PGSR
        max_coords = torch.tensor([1.5, 2.45, -0.15], device=points.device)
        min_coords = torch.tensor([-2.1, -1.25, -3.38], device=points.device)
        print(f"Original Bounding Box: min={min_coords}, max={max_coords}")

        # Expand bounding box
        new_min_coords, new_max_coords = expand_bounding_box(min_coords, max_coords, scale_factor=scale_factor)
        print(f"Expanded Bounding Box: min={new_min_coords}, max={new_max_coords}")
        
        voxel_grid, grid_size = create_voxel_grid(new_min_coords, new_max_coords, voxel_size=voxel_size)
        print(f"Voxel Grid size: {voxel_grid.shape}, Grid size per voxel: {grid_size}")
        
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
        assert voxel.shape == (128, 128, 128)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    # parser.add_argument("--skip_train", action="store_true")
    # parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("-v", "--voxel_size", type=int, default=128)
    parser.add_argument("--scale_factor", type=float, default=1.5)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    grid(model.extract(args), args.iteration, pipeline.extract(args), args.voxel_size, args.scale_factor)