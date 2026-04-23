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
import json
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, render_PGSR
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import cv2
import open3d as o3d
from scene.app_model import AppModel
import copy
from collections import deque
import imageio
from utils.graphics_utils import fov2focal
from argparse import ArgumentParser, Namespace
import sys
import os
import yaml
# Lazy import: TrelllisGaussian requires utils3d, only needed for object insertion
# from scene.trellis_gaussian import TrelllisGaussian

os.environ['OMP_NUM_THREADS'] = '1'
# === helpers: build Open3D intrinsics / extrinsics from 3DGS view ===
def view_to_o3d_intrinsic(view, H, W):
    fx = fov2focal(view.FoVx, W)
    fy = fov2focal(view.FoVy, H)
    cx = W/2 - 0.5
    cy = H/2 - 0.5
    return o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)

def view_to_world2cam_extrinsic(view):
    # Elsewhere you use (view.world_view_transform.T @ [X,1]) -> camera coords
    # Open3D's extrinsic is exactly world->camera, so just use .T directly
    return view.world_view_transform.cpu().numpy().T


def create_voxel_grid(min_coords, max_coords, voxel_size=128):
    # Calculate the size of each voxel
    grid_size = (max_coords - min_coords) / voxel_size
    # Get the boundary range of the voxel grid
    grid = torch.zeros((voxel_size, voxel_size, voxel_size), dtype=torch.int)

    return grid, grid_size

def get_bounding_box(points):
    # Calculate the min and max values of the point cloud along each axis
    min_coords = torch.min(points, dim=0).values  # Min value per dimension
    max_coords = torch.max(points, dim=0).values  # Max value per dimension
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
    # points = points[opacities > 0.05]
    plt.figure(figsize=(8, 6))
    plt.hist(opacities.cpu().numpy(), bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Opacity Distribution', fontsize=14)
    plt.xlabel('Opacity Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True)

    # plt.savefig('opacity', dpi=300, bbox_inches='tight')  # Save as PNG format, 300 DPI

    # Close the current figure to avoid excessive memory usage
    plt.close()
    normalized_points = (points - min_coords) / grid_size
    indices = normalized_points.long()
    # indices = torch.clamp(indices, 0, voxel_grid.size(0) - 1)
    valid_indices_mask = (indices[:, 0] >= 0) & (indices[:, 0] < voxel_grid.size(0)) & \
                          (indices[:, 1] >= 0) & (indices[:, 1] < voxel_grid.size(1)) & \
                          (indices[:, 2] >= 0) & (indices[:, 2] < voxel_grid.size(2))
    valid_indices = indices[valid_indices_mask]
    voxel_grid[valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]] = 1
    return voxel_grid, indices

def print_voxel_statistics(voxel_grid):
    # Count the number of voxels with value 1 in the voxel grid
    num_ones = torch.sum(voxel_grid == 1).item()
    
    # Calculate the total number of voxels in the grid
    total_voxels = voxel_grid.numel()
    
    # Calculate the ratio of voxels with value 1
    ratio_ones = num_ones / total_voxels

    # Print results
    print(f"Number of voxels with value 1: {num_ones}")
    print(f"Total number of voxels: {total_voxels}")
    print(f"Ratio of voxels with value 1: {ratio_ones * 100:.2f}%")
    
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_voxel_grid(voxel_grid, output_filename='voxel_grid.png'):
    # Get coordinates of all voxels with value 1 in the voxel grid
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
    # plt.savefig(output_filename, dpi=300, bbox_inches='tight')  # Save as PNG format, 300 DPI

    # Close the current figure to avoid excessive memory usage
    plt.close()

    print(f"Image saved as {output_filename}")

def voxel_grid_to_point_cloud(voxel_grid, voxel_size, min_coords):
    # Get the coordinates of all voxels with value 1 in the voxel grid
    x, y, z = np.where(voxel_grid.cpu() == 1)
    # print(x.device, min_coords.device)
    
    # Calculate the actual coordinates of each point (based on voxel grid and voxel size)
    points = np.vstack((x, y, z)).T * voxel_size.cpu().numpy() + min_coords.cpu().numpy()  # Restore to actual spatial coordinates

    
    # Create Open3D point cloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    
    return point_cloud

def depth_to_xyz_map(depth_map, H, W, K):
    u_coords = torch.arange(0, W, dtype=torch.float32, device=depth_map.device)
    v_coords = torch.arange(0, H, dtype=torch.float32, device=depth_map.device)
    u, v = torch.meshgrid(u_coords, v_coords, indexing='xy')

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x = (u - cx) / fx
    y = (v - cy) / fy
    z = depth_map

    x = x * z
    y = y * z

    # Create mask for valid depth values
    nonzero_depth_mask = depth_map > 0
    x = x * nonzero_depth_mask
    y = y * nonzero_depth_mask
    z = z * nonzero_depth_mask

    xyz_map = torch.stack((x, y, z), dim=-1)    # [h,w,3]
    #point_cloud = point3d.view(-1, 3)          # [N, 3]

    return xyz_map, nonzero_depth_mask

def xyz_to_uv(xyz_map, K):
    # Extract the camera intrinsics
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Extract x, y, z from the xyz_map
    x = xyz_map[..., 0]
    y = xyz_map[..., 1]
    z = xyz_map[..., 2]

    # Project the 3D points back onto the 2D image plane (u, v)
    u = fx * (x / z) + cx  # Projection in the u (horizontal) direction
    v = fy * (y / z) + cy  # Projection in the v (vertical) direction

    # Flatten the coordinates for easy indexing in the depth map
    u = u.flatten()
    v = v.flatten()
    z = z.flatten()

    # Create the depth map (or return depth values) and their corresponding pixel locations
    # depth_map = torch.zeros((H, W), dtype=torch.float32, device=xyz_map.device)

    # Convert float pixel coordinates (u, v) to integers for indexing
    u_int = u.round().long()  # Make sure u is within bounds
    v_int = v.round().long()  # Make sure v is within bounds

    return u_int, v_int

def clean_mesh(mesh, min_len=1000):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < min_len
    mesh_0 = copy.deepcopy(mesh)
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    return mesh_0

def post_process_mesh(mesh, cluster_to_keep=1):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy
    print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50) # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0

def render_set(model_path, name, iteration, views, scene, gaussians, pipeline, background, 
               app_model=None, max_depth=5.0, volume=None, use_depth_filter=False):
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    render_depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_depth")
    render_normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_normal")

    makedirs(gts_path, exist_ok=True)
    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(render_depth_path, exist_ok=True)
    makedirs(render_normal_path, exist_ok=True)

    depths_tsdf_fusion = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        gt, _ = view.get_image()
        out = render_PGSR(view, gaussians, pipeline, background, app_model=app_model)
        rendering = out["render"].clamp(0.0, 1.0)
        _, H, W = rendering.shape

        depth = out["plane_depth"].squeeze()
        depth_tsdf = depth.clone()
        depth = depth.detach().cpu().numpy()
        np.save(os.path.join(depth_path, '{0:05d}'.format(idx) + ".npy"), depth)
        depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
        depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)
        
        normal = out["rendered_normal"].permute(1,2,0)
        normal = normal/(normal.norm(dim=-1, keepdim=True)+1.0e-8)
        normal = normal.detach().cpu().numpy()
        normal = ((normal+1) * 127.5).astype(np.uint8).clip(0, 255)

        if name == 'test':
            torchvision.utils.save_image(gt.clamp(0.0, 1.0), os.path.join(gts_path, view.image_name + ".png"))
            torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + ".png"))
        else:
            torchvision.utils.save_image(gt.clamp(0.0, 1.0), os.path.join(gts_path, view.image_name + ".png"))
            torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + ".png"))
            # rendering_np = (rendering.permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
            # cv2.imwrite(os.path.join(render_path, view.image_name + ".jpg"), rendering_np)
        cv2.imwrite(os.path.join(render_depth_path, view.image_name + ".jpg"), depth_color)
        cv2.imwrite(os.path.join(render_normal_path, view.image_name + ".jpg"), normal)
        
        # if use_depth_filter:
        #     view_dir = torch.nn.functional.normalize(view.get_rays(), p=2, dim=-1)
        #     depth_normal = out["depth_normal"].permute(1,2,0)
        #     depth_normal = torch.nn.functional.normalize(depth_normal, p=2, dim=-1)
        #     dot = torch.sum(view_dir*depth_normal, dim=-1).abs()
        #     angle = torch.acos(dot)
        #     mask = angle > (80.0 / 180 * 3.14159)
        #     depth_tsdf[mask] = 0
        # depths_tsdf_fusion.append(depth_tsdf.squeeze().cpu())
        
    # if volume is not None:
    #     depths_tsdf_fusion = torch.stack(depths_tsdf_fusion, dim=0)
    #     for idx, view in enumerate(tqdm(views, desc="TSDF Fusion progress")):
    #         ref_depth = depths_tsdf_fusion[idx].cuda()

    #         if view.mask is not None:
    #             ref_depth[view.mask.squeeze() < 0.5] = 0
    #         ref_depth[ref_depth>max_depth] = 0
    #         ref_depth = ref_depth.detach().cpu().numpy()
                
    #         pose = np.identity(4)
    #         pose[:3,:3] = view.R.transpose(-1,-2)
    #         pose[:3, 3] = view.T
    #         color = o3d.io.read_image(os.path.join(render_path, view.image_name + ".jpg"))
    #         depth = o3d.geometry.Image((ref_depth*1000).astype(np.uint16))
    #         rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    #             color, depth, depth_scale=1000.0, depth_trunc=max_depth, convert_rgb_to_intensity=False)
    #         volume.integrate(
    #                 rgbd,
    #                 o3d.camera.PinholeCameraIntrinsic(W, H, view.Fx, view.Fy, view.Cx, view.Cy),
    #                 pose)
        
def quantize_to_nearest_colors_with_black_filter(image: torch.Tensor, palette: np.ndarray, black_thresh=0.05) -> torch.Tensor:
    """
    Args:
        image: torch.Tensor of shape [3, H, W], values in [0, 1]
        palette: np.ndarray of shape [K, 3], RGB in [0, 1]
        black_thresh: brightness threshold under which we treat a pixel as black
    Returns:
        Cleaned image with each pixel replaced by closest color from palette
    """
    _, H, W = image.shape
    img_flat = image.permute(1, 2, 0).reshape(-1, 3)  # [N, 3]
    palette_unique = np.unique(palette, axis=0)

    palette_tensor = torch.from_numpy(palette_unique).float().to(image.device)  # [K, 3]
    dists = torch.cdist(img_flat, palette_tensor)  # [N, K]

    # Determine which pixels are “very dark”, brightness below threshold
    brightness = img_flat.mean(dim=1)  # [N]
    is_black_pixel = brightness < black_thresh

    # Find indices of non-black palette entries
    palette_is_black = (palette_tensor.mean(dim=1) < black_thresh)  # [K]
    non_black_palette = palette_tensor[~palette_is_black]           # [K_nonblack]
    
    # For black pixels, find the nearest color in non-black palette
    black_pixels = img_flat[is_black_pixel]
    if len(black_pixels) > 0:
        dists_black = torch.cdist(black_pixels, non_black_palette)  # [M, K_nonblack]
        nearest_black = torch.argmin(dists_black, dim=1)            # [M]
        replaced_colors = non_black_palette[nearest_black]          # [M, 3]
    else:
        replaced_colors = torch.empty((0, 3), device=image.device)

    # For all pixels, find the nearest palette color (normal flow)
    nearest_all = torch.argmin(dists, dim=1)             # [N]
    full_result = palette_tensor[nearest_all]            # [N, 3]

    # Replace colors at original black pixel positions
    full_result[is_black_pixel] = replaced_colors        # Replace black portions

    # Reshape back to original image size
    result = full_result.reshape(H, W, 3).permute(2, 0, 1)  # [3, H, W]
    return result

from PIL import Image
import torch

def save_with_transparency(image_tensor, path, black_thresh=0.2):
    """
    Args:
        image_tensor: torch.Tensor of shape [3, H, W], RGB in [0, 1]
        path: str, output file path
        black_thresh: float, mean RGB below this value is considered 'black'
    """
    assert image_tensor.shape[0] == 3, "Expect [3, H, W] image"
    image_rgb = (image_tensor * 255).byte().permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
    h, w, _ = image_rgb.shape

    # Calculate alpha channel (black regions become transparent)
    alpha = (image_rgb.mean(axis=2) > int(black_thresh * 255)).astype(np.uint8) * 255  # [H, W]

    # Concatenate RGBA
    image_rgba = np.dstack([image_rgb, alpha])  # [H, W, 4]

    # Save
    Image.fromarray(image_rgba, mode='RGBA').save(path)
    print(f"Saved RGBA image to: {path}")
    
def render_sets_filling(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool,
                 max_depth : float, voxel_size : float, num_cluster: int, use_depth_filter : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, filled=True)
        # app_model = AppModel()
        # app_model.load_weights(scene.model_path)
        # app_model.eval()
        # app_model.cuda()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        min_coords = np.array(args.bounding_box['min'])
        max_coords = np.array(args.bounding_box['max'])
        center = (min_coords + max_coords) / 2.0
        side = float(np.max(max_coords - min_coords))
        volume = o3d.pipelines.integration.UniformTSDFVolume(
            length=side,
            resolution=128,  # Set as needed, e.g. 128/256/512
            sdf_trunc=4.0 * (side / 256),
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
            origin=center  # Place the TSDF cube at this center position
        )
        
        # volume = o3d.pipelines.integration.ScalableTSDFVolume(
        #     voxel_length=voxel_size,
        #     sdf_trunc=4.0*voxel_size,
        #     color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        
        print(gaussians.compute_Laniso(a=0.0))
        print(gaussians.get_xyz.shape)

        if not skip_train:
            render_set(dataset.model_path, "Filled", scene.loaded_iter, scene.getTrainCameras()[:5], scene, gaussians, pipeline, background, 
                       max_depth=max_depth, volume=volume, use_depth_filter=use_depth_filter)

        # if not skip_test:
            # render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), scene, gaussians, pipeline, background)
        
def render_sets_insert(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool,
                 max_depth : float, voxel_size : float, num_cluster: int, use_depth_filter : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        # app_model = AppModel()
        # app_model.load_weights(scene.model_path)
        # app_model.eval()
        # app_model.cuda()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        min_coords = np.array(args.bounding_box['min'])
        max_coords = np.array(args.bounding_box['max'])
        center = (min_coords + max_coords) / 2.0
        side = float(np.max(max_coords - min_coords))
        volume = o3d.pipelines.integration.UniformTSDFVolume(
            length=side,
            resolution=128,  # Set as needed, e.g. 128/256/512
            sdf_trunc=4.0 * (side / 256),
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
            origin=center  # Place the TSDF cube at this center position
        )
        
        # volume = o3d.pipelines.integration.ScalableTSDFVolume(
        #     voxel_length=voxel_size,
        #     sdf_trunc=4.0*voxel_size,
        #     color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        
        print(gaussians.compute_Laniso(a=0.0))
        print(gaussians.get_xyz.shape)
        # Nx, Ny, Nz = [256, 256, 256]

        # # --- Directly copied mask generation logic ---
        # x_center = Nx // 2 + 0.04 * Nx
        # y_center = Ny // 2
        # x_half_width = int(0.06 * Nx)
        # y_half_width = int(0.25 * Ny)
        # # Note: z_center is hardcoded to 160 in your code,
        # # if the loaded grid's z-axis is not large enough it may throw an error; here we assume the grid is large enough
        # z_center = 160 
        # z_half_thickness = int(0.02 * Nz)

        # # Create coordinate grid
        # i_indices, j_indices, k_indices = np.ogrid[:Nx, :Ny, :Nz]

        # # Create mask
        # x_mask = (i_indices >= x_center - x_half_width) & (i_indices <= x_center + x_half_width)
        # y_mask = (j_indices >= y_center - y_half_width) & (j_indices <= y_center + y_half_width)
        # z_mask = (k_indices >= z_center - z_half_thickness) & (k_indices <= z_center + z_half_thickness)

        # boundary_mask = x_mask & y_mask & z_mask
        
        # # Get all voxel indices in this region
        # # indices shape: [N_points, 3] -> (x_idx, y_idx, z_idx)
        # indices = np.argwhere(boundary_mask)
        
        # if indices.shape[0] == 0:
        #     print("Warning: The mask is empty! Check z_center vs grid dimensions.")
        #     return

        # # -------------------------------------------------
        # # 3. Convert voxel indices to world coordinates
        # # -------------------------------------------------
        # # Formula: World = Origin + Index * VoxelSize
        # # For more precision, typically use the voxel center, hence adding 0.5
        # voxel_size = 0.010312500409781933
        # min_coords = np.array([-1.6200001239776611, -1.6200001239776611, -3.2200000286102295])
        # world_positions = min_coords + (indices * voxel_size) + (voxel_size * 0.5)

        # # -------------------------------------------------
        # # 4. Output result statistics
        # # -------------------------------------------------
        # obj_min = world_positions.min(axis=0)
        # obj_max = world_positions.max(axis=0)
        # obj_center = (obj_min + obj_max) / 2

        # print("-" * 30)
        # print(">>> Added Object World Statistics <<<")
        # print(f"Point Count: {world_positions.shape[0]}")
        # print(f"World Center (X, Y, Z): \n  {obj_center}")
        # print(f"World Bounding Box Min: \n  {obj_min}")
        # print(f"World Bounding Box Max: \n  {obj_max}")
        # print("-" * 30)

        # # -------------------------------------------------
        # # 5. (Optional) Save as ply or txt for verification in visualization software
        # # -------------------------------------------------
        # # Save as txt (X Y Z)
        # # save_path = os.path.join(scene.model_path, "added_obstacle_world.txt")
        # # np.savetxt(save_path, world_positions, fmt='%.6f')
        # # print(f"Saved world coordinates to: {save_path}")
        
        # gaussians.append_gaussians_from_new_xyz_iclr_rebuttal(torch.from_numpy(world_positions))
        # gaussians.save_ply(os.path.join(scene.model_path, "gaussians_after_insertion.ply"))
        
        from scene.trellis_gaussian import TrelllisGaussian
        trellis_gaussians = TrelllisGaussian(sh_degree=0)
        trellis_gaussians.load_ply(os.path.join(scene.model_path, "brick.ply"), transform_pos=np.array([-0.19, -0.32, -0.98]))
        gaussians.append_gaussians_from_trellis(trellis_gaussians)
        gaussians.save_ply(os.path.join(scene.model_path, "gaussians_after_insertion_trellis.ply"))
        

        if not skip_train:
            render_set(dataset.model_path, "train_insert", scene.loaded_iter, scene.getTrainCameras(), scene, gaussians, pipeline, background, 
                       max_depth=max_depth, volume=volume, use_depth_filter=use_depth_filter)

        if not skip_test:
            render_set(dataset.model_path, "test_insert", scene.loaded_iter, scene.getTestCameras(), scene, gaussians, pipeline, background)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool,
                 max_depth : float, voxel_size : float, num_cluster: int, use_depth_filter : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        # app_model = AppModel()
        # app_model.load_weights(scene.model_path)
        # app_model.eval()
        # app_model.cuda()
        
        # trellis_gaussians = TrelllisGaussian(dataset.sh_degree)
        # trellis_gaussians.load_ply(os.path.join(scene.model_path, "brick.ply"))

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        min_coords = np.array(args.bounding_box['min'])
        max_coords = np.array(args.bounding_box['max'])
        center = (min_coords + max_coords) / 2.0
        side = float(np.max(max_coords - min_coords))
        # volume = o3d.pipelines.integration.UniformTSDFVolume(
        #     length=side,
        #     resolution=128,  # Set as needed, e.g. 128/256/512
        #     sdf_trunc=4.0 * (side / 256),
        #     color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        #     origin=center  # Place the TSDF cube at this center position
        # )
        
        # volume = o3d.pipelines.integration.ScalableTSDFVolume(
        #     voxel_length=voxel_size,
        #     sdf_trunc=4.0*voxel_size,
        #     color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        
        print(gaussians.compute_Laniso(a=0.0))
        print(gaussians.get_xyz.shape)

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), scene, gaussians, pipeline, background,
                       max_depth=max_depth, use_depth_filter=use_depth_filter)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), scene, gaussians, pipeline, background)
        
        # if hasattr(volume, "extract_voxel_grid"):           # Ensure it is a UniformTSDFVolume
        #     vg = volume.extract_voxel_grid()                # -> open3d.geometry.VoxelGrid
        #     voxels = vg.get_voxels()                        # Sparse voxel list with grid_index
        #     # For dense occupancy:
        #     N = args.voxel_size
        #     bbox = vg.get_axis_aligned_bounding_box()
        #     cell = bbox.get_max_extent() / N
        #     occ = np.zeros((N,N,N), np.uint8)
        #     for v in voxels:
        #         cx, cy, cz = vg.get_voxel_center_coordinate(v.grid_index)
        #         ix = int(np.clip(np.floor((cx-bbox.min_bound[0])/cell),0,N-1))
        #         iy = int(np.clip(np.floor((cy-bbox.min_bound[1])/cell),0,N-1))
        #         iz = int(np.clip(np.floor((cz-bbox.min_bound[2])/cell),0,N-1))
        #         occ[ix,iy,iz] = 1
        #     np.save(os.path.join(dataset.model_path, f"tsdf_occ_{N}.npy"), occ)


        # ply_file_path = os.path.join(dataset.model_path, 'material_colored_hierach_knn.ply')
        # from plyfile import PlyData
        # ply_data = PlyData.read(ply_file_path)

        # points = np.vstack([ply_data['vertex']['x'], ply_data['vertex']['y'], ply_data['vertex']['z']]).T
        # colors = np.vstack([ply_data['vertex']['red'], ply_data['vertex']['green'], ply_data['vertex']['blue']]).T / 255.0
        # rendered_seg_map = render_PGSR(scene.getTrainCameras()[61], gaussians, pipeline, background, override_color=torch.from_numpy(colors).cuda().float(), return_plane=False, return_depth_normal=False)['render']
        # torchvision.utils.save_image(rendered_seg_map, os.path.join(dataset.model_path, "vis_material.png"))


@torch.no_grad()
def build_dense_tsdf_levelset(views, depth_dir, aabb_min, aabb_max,
                              N=128, mu_ratio=4.0, max_depth=50.0, device="cuda"):
    """
    Generate a dense N^3 TSDF voxel grid (level set) directly within the Axis-Aligned Bounding Box (AABB).
    Returns: tsdf [N,N,N], weight [N,N,N], occ [N,N,N] (bool)
    """
    aabb_min = torch.as_tensor(aabb_min, dtype=torch.float32, device=device)
    aabb_max = torch.as_tensor(aabb_max, dtype=torch.float32, device=device)
    extent   = aabb_max - aabb_min
    voxel_len = float(torch.max(extent).item() / N)
    mu = mu_ratio * voxel_len

    # Build voxel center coordinates (world frame)
    xs = torch.linspace(aabb_min[0], aabb_max[0], N, device=device)
    ys = torch.linspace(aabb_min[1], aabb_max[1], N, device=device)
    zs = torch.linspace(aabb_min[2], aabb_max[2], N, device=device)
    X, Y, Z = torch.meshgrid(xs, ys, zs, indexing="ij")     # [N,N,N]
    P_world = torch.stack([X, Y, Z, torch.ones_like(X)], dim=-1).view(-1, 4).T  # [4, N^3]

    tsdf_sum   = torch.zeros(N**3, device=device)
    weight_sum = torch.zeros(N**3, device=device)

    # Process in chunks to avoid projecting all N^3 points at once
    CHUNK = 800_000

    for i, view in enumerate(views):
        depth_path = os.path.join(depth_dir, f"{i:05d}.npy")
        if not os.path.exists(depth_path):
            continue
        depth = np.load(depth_path).astype(np.float32)             # [H,W] (meters)
        H, W = depth.shape
        depth_t = torch.from_numpy(depth).to(device)

        # intrinsics
        fx = fov2focal(view.FoVx, W); fy = fov2focal(view.FoVy, H)
        cx = W/2 - 0.5;            cy = H/2 - 0.5
        K = torch.tensor([[fx,0,cx],[0,fy,cy],[0,0,1]], device=device, dtype=torch.float32)

        # extrinsic: world -> cam
        E = torch.from_numpy(view.world_view_transform.cpu().numpy().T).to(device).float()

        # Process chunk by chunk
        for s in range(0, P_world.shape[1], CHUNK):
            e = min(s+CHUNK, P_world.shape[1])
            Pw = P_world[:, s:e]                                    # [4,M]
            Pc = (E @ Pw)[:3, :]                                    # Camera frame [3,M]
            z  = Pc[2, :]
            valid = z > 0

            # Project to pixel coordinates
            u = K[0,0] * (Pc[0]/z.clamp(min=1e-6)) + K[0,2]
            v = K[1,1] * (Pc[1]/z.clamp(min=1e-6)) + K[1,2] # Pixel position of each voxel
            valid &= (u >= 0) & (u < W-1) & (v >= 0) & (v < H-1)

            if valid.sum() == 0:
                continue

            idx_flat = torch.arange(s, e, device=device)[valid]      # Flat indices of these voxels in the full grid
            u = u[valid]; v = v[valid]; z_vox = z[valid]

            # Bilinear sampling of depth
            u0 = torch.floor(u); v0 = torch.floor(v)
            du = u - u0;         dv = v - v0
            u0 = u0.long();      v0 = v0.long()
            u1 = (u0 + 1).clamp(max=W-1); v1 = (v0 + 1).clamp(max=H-1)

            d00 = depth_t[v0, u0]; d10 = depth_t[v0, u1]
            d01 = depth_t[v1, u0]; d11 = depth_t[v1, u1]
            d = (d00*(1-du)*(1-dv) + d10*du*(1-dv) + d01*(1-du)*dv + d11*du*dv) # Depth at the pixel corresponding to each voxel

            # Filter invalid / too-far depth values
            m = (d > 0) & (d <= max_depth) & torch.isfinite(d)
            if m.sum() == 0:
                continue

            idx_flat = idx_flat[m]
            d = d[m]; z_vox = z_vox[m] # Actual depth of the voxel

            # Normalized truncated TSDF
            s_norm = (d - z_vox) / mu
            s_norm = torch.clamp(s_norm, -1.0, 1.0)

            # (Optional) Angular weight: suppress grazing angles
            # wi = torch.clamp((z_vox / d), min=0.0, max=1.0)  # Rough proxy
            wi = torch.ones_like(s_norm)

            # Accumulate into the corresponding voxels
            tsdf_sum.index_add_(0, idx_flat, wi * s_norm)
            weight_sum.index_add_(0, idx_flat, wi)

    # Normalize to get the final TSDF
    tsdf = torch.ones(N**3, device=device)
    m = weight_sum > 0
    tsdf[m] = tsdf_sum[m] / weight_sum[m]
    tsdf = tsdf.view(N, N, N)
    weight = weight_sum.view(N, N, N)

    occ = (tsdf < 0) & (weight > 0)   # level set -> occupancy
    print(occ.sum())
    return tsdf, weight, occ, voxel_len



def save_occ_points_as_ply(occ, aabb_min, voxel_len, out_path):
    """
    occ: (N,N,N) bool/uint8, occupied=1
    aabb_min: (3,) AABB minimum point in world coordinates [xmin, ymin, zmin]
    voxel_len: float, edge length of each voxel (meters)
    out_path: output .ply file path
    Convention: occ[i,j,k] corresponds to world coordinates:
          x = xmin + (i + 0.5) * voxel_len
          y = ymin + (j + 0.5) * voxel_len
          z = zmin + (k + 0.5) * voxel_len
    """
    occ_np = occ.astype(bool) if isinstance(occ, np.ndarray) else occ.cpu().numpy().astype(bool)
    idx = np.argwhere(occ_np)   # [M,3] -> (i,j,k)

    if idx.shape[0] == 0:
        print("No occupied voxels; skip writing.")
        return

    idx_center = (idx + 0.5) * voxel_len
    pts = idx_center + np.asarray(aabb_min, dtype=np.float32)[None, :]

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    # Optional: uniform coloring
    # pcd.colors = o3d.utility.Vector3dVector(np.tile([[1,0,0]], (pts.shape[0],1)))  # Red

    o3d.io.write_point_cloud(out_path, pcd, write_ascii=False, compressed=False)
    print(f"[OK]saved point-cloud PLY with {pts.shape[0]} points → {out_path}")



def extract_occ(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool,
                 max_depth : float, voxel_size : float, num_cluster: int, use_depth_filter : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        # app_model = AppModel()
        # app_model.load_weights(scene.model_path)
        # app_model.eval()
        # app_model.cuda()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        min_coords = np.array(args.bounding_box['min'])
        max_coords = np.array(args.bounding_box['max'])
        center = (min_coords + max_coords) / 2.0
        side = float(np.max(max_coords - min_coords))
        
        depth_dir = os.path.join(dataset.model_path, "train", "ours_30000", "depth")
        tsdf, wts, occ, voxel_len = build_dense_tsdf_levelset(
            views=scene.getTrainCameras(),
            depth_dir=depth_dir,
            aabb_min=min_coords.tolist(),
            aabb_max=max_coords.tolist(),
            # N=128,            # e.g. 128
            mu_ratio=1.0,
            # max_depth=args.max_depth,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        np.save(os.path.join(dataset.model_path, f"tsdf_128.npy"), tsdf.detach().cpu().numpy())
        np.save(os.path.join(dataset.model_path, f"occ_128.npy"),  occ.detach().cpu().numpy().astype(np.uint8))
        save_occ_points_as_ply(occ, aabb_min=min_coords.tolist(),
                       voxel_len=float(voxel_len),
                       out_path=os.path.join(dataset.model_path, "occ_points_128.ply"))
        print("[OK]saved dense TSDF & occupancy. voxel_len(m) =", voxel_len)
            
def extract_fg(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        # points_fg_num = torch.zeros((gaussians.get_xyz.size()[0], 2), dtype=torch.uint32, device=)
        points = gaussians.get_xyz
        points_fg_num = torch.zeros((gaussians.get_xyz.size()[0], 2), dtype=torch.int32, device=points.device)
        ones = torch.ones((points.size()[0], 1), device=points.device)
        points_h = torch.cat([points, ones], dim=1) # [N, 4]
        views = scene.getTrainCameras()
        opacities = gaussians.get_opacity.squeeze()
        burnability = torch.from_numpy(np.load(os.path.join(scene.model_path, "points_burnability.npy"))).cuda()
        
        # selected_points = selected_points & (burnability == 0)
        
        selected_points = (burnability == 0)
        print(selected_points.sum())
        
        points = gaussians.get_xyz
        print(points.shape[0])

        # Read initial tight bounding box from YAML config, then expand
        voxel_cfg = args.extract_voxel
        min_coords = torch.tensor(voxel_cfg['bbox_min'], device=points.device)
        max_coords = torch.tensor(voxel_cfg['bbox_max'], device=points.device)
        scale_factor = voxel_cfg.get('scale_factor', args.scale_factor)
        new_min_coords, new_max_coords = expand_bounding_box(min_coords, max_coords, scale_factor=scale_factor)
        print(f"Expanded Bounding Box: min={new_min_coords}, max={new_max_coords}")
        
        in_voxel_mask = (
            (points[:, 0] >= new_min_coords[0]) & (points[:, 0] <= new_max_coords[0]) &
            (points[:, 1] >= new_min_coords[1]) & (points[:, 1] <= new_max_coords[1]) &
            (points[:, 2] >= new_min_coords[2]) & (points[:, 2] <= new_max_coords[2])
        )
        selected_points = selected_points & in_voxel_mask
        
        gaussians.crop_setup(selected_points)
        save_path = os.path.join(scene.model_path, "fg", "fg_opac_0_burnable.ply")
        gaussians.save_ply(save_path)
        
        points = gaussians.get_xyz
        print(points.shape[0])
        
        voxel_grid, grid_size = create_voxel_grid(new_min_coords, new_max_coords, voxel_size=args.voxel_size)
        print(f"Voxel Grid size: {voxel_grid.shape}, Grid size per voxel: {grid_size}")
        config = {
            "bounding_box": {
                "min": new_min_coords.tolist(),   # ==> list[float]
                "max": new_max_coords.tolist(),
            },
            "voxel_grid": {
                "dims": list(voxel_grid.shape),    # ==> list[int]
                "voxel_size": grid_size[0].item()  # If isotropic, store a single number; if anisotropic, store a list
            },
        }
        import yaml
        from pathlib import Path
        cfg_path = Path(os.path.join(scene.model_path, "simulation_voxel.yaml"))
        cfg_path.write_text(yaml.safe_dump(config, sort_keys=False))
        print(f"[OK]Config saved to {cfg_path.resolve()}")
        voxel_grid, indices_pts_in_grids = map_points_to_voxels(points, new_min_coords, gaussians.get_opacity, grid_size, voxel_grid)
        
        print_voxel_statistics(voxel_grid)
        
        visualize_voxel_grid(voxel_grid=voxel_grid)
        
        point_cloud = voxel_grid_to_point_cloud(voxel_grid=voxel_grid, voxel_size=grid_size, min_coords=new_min_coords)
        voxel_path = os.path.join(scene.model_path, "voxel.ply")
        o3d.io.write_point_cloud(voxel_path, point_cloud)
        
        voxel_grid_path = os.path.join(scene.model_path, "voxel_grid.npy")
        np.save(voxel_grid_path,voxel_grid)
        voxel = np.load(voxel_grid_path)
        print(voxel.shape)
        # assert voxel.shape == (128, 128, 128)
        
        print("save indices_pts_in_grids")  # Grid coordinates of each Gaussian point in the grids
        indices_pts_in_grids_path = os.path.join(scene.model_path, "indices_pts_in_grids.pth")
        torch.save(indices_pts_in_grids, indices_pts_in_grids_path)

        # print("save indices_pts_ingrids_for_inall")   # Index of each Gaussian point in the grids among all Gaussians
        # indices_pts_ingrids_for_inall_path = os.path.join(scene.model_path, "indices_pts_ingrids_for_inall.pth")
        # indices_pts_ingrids_for_inall = torch.nonzero(selected_points, as_tuple=False).squeeze()
        # torch.save(indices_pts_ingrids_for_inall, indices_pts_ingrids_for_inall_path)

        print("save selected_points")  # Mask of each Gaussian point in the grids among all Gaussians
        mask_pts_in_grids_path = os.path.join(scene.model_path, "mask_pts_in_grids.pth")
        torch.save(selected_points, mask_pts_in_grids_path)


def load_yaml_config(yaml_path):
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)

def update_args_with_yaml(args, yaml_config):
    for key, value in yaml_config.items():
        if not hasattr(args, key):
            setattr(args, key, value)
            

if __name__ == "__main__":
    torch.set_num_threads(8)
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--max_depth", default=5.0, type=float)
    # parser.add_argument("--voxel_size", default=0.002, type=float)
    parser.add_argument("--num_cluster", default=1, type=int)
    parser.add_argument("--use_depth_filter", action="store_true")
    parser.add_argument("-v", "--voxel_size", type=int, default=128)
    parser.add_argument("--scale_factor", type=float, default=1.5)
    parser.add_argument("--config_file", type=str, help="Path to the YAML configuration file")
    
    args = get_combined_args(parser)

    if args.config_file:
        yaml_config = load_yaml_config(args.config_file)
        update_args_with_yaml(args, yaml_config)
        
    # print(args.y_rotation)

    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    print(f"multi_view_num {model.multi_view_num}")
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.max_depth, args.voxel_size, args.num_cluster, args.use_depth_filter)
    # render_sets_filling(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.max_depth, args.voxel_size, args.num_cluster, args.use_depth_filter)
    # extract_fg(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
    # extract_occ(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.max_depth, args.voxel_size, args.num_cluster, args.use_depth_filter)