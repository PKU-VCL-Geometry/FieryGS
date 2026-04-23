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
from utils.filling_utils import *
from argparse import ArgumentParser, Namespace
import sys
import os
import yaml
import taichi as ti
ti.init(arch=ti.gpu)

def create_voxel_grid(min_coords, max_coords, voxel_size=128):
    # Calculate the size of each voxel
    grid_size = (max_coords - min_coords) / voxel_size
    # Get the boundary range of the voxel grid
    grid = torch.zeros((voxel_size, voxel_size, voxel_size), dtype=torch.int)

    return grid, grid_size

def get_bounding_box(points):
    # Calculate the minimum and maximum values of the point cloud along each axis
    min_coords = torch.min(points, dim=0).values  # Minimum value along each dimension
    max_coords = torch.max(points, dim=0).values  # Maximum value along each dimension
    return min_coords, max_coords

def expand_bounding_box(min_coords, max_coords, scale_factor=1.5):
    center = (min_coords + max_coords) / 2
    box = max_coords - min_coords
    size = max(box[0] * scale_factor, max(box[1] * scale_factor, box[2] * scale_factor))
    new_min_coords = center - size / 2
    new_max_coords = center + size / 2
    return new_min_coords, new_max_coords

import torch
import matplotlib.pyplot as plt
from collections import Counter

def map_points_to_voxels_materials(points, min_coords, opacities, grid_size, voxel_grid, materials):
    # Ensure opacity is a 1D tensor
    opacities = opacities.squeeze()
    print(opacities.min())
    print(opacities.max())

    # Plot the histogram of opacities
    plt.figure(figsize=(8, 6))
    plt.hist(opacities.cpu().numpy(), bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Opacity Distribution', fontsize=14)
    plt.xlabel('Opacity Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True)
    # plt.close()
    plt.savefig('chair_opacity_vis.png')

    # Normalize point coordinates and map to voxel grid coordinates
    normalized_points = (points - min_coords) / grid_size
    indices = normalized_points.long()
    # indices = torch.floor(normalized_points).to(torch.long)

    # Create a material grid with the same shape as the voxel grid
    material_grid = torch.zeros_like(voxel_grid, dtype=torch.long)

    # Check valid voxel indices, ensuring they are within the voxel grid bounds
    valid_indices_mask = (indices[:, 0] >= 0) & (indices[:, 0] < voxel_grid.size(0)) & \
                          (indices[:, 1] >= 0) & (indices[:, 1] < voxel_grid.size(1)) & \
                          (indices[:, 2] >= 0) & (indices[:, 2] < voxel_grid.size(2))
    valid_indices = indices[valid_indices_mask]

    for i in range(len(valid_indices)):
        x, y, z = valid_indices[i]
        material_value = materials[valid_indices_mask][i]  # Get the material value for the corresponding point
        # Mark the corresponding position in the voxel grid with this material value
        material_grid[x, y, z] = material_value + 1
        voxel_grid[x, y, z] = 1

    # H, W, D = voxel_grid.shape
    # valid = (
    #     (indices[:, 0] >= 0) & (indices[:, 0] < H) &
    #     (indices[:, 1] >= 0) & (indices[:, 1] < W) &
    #     (indices[:, 2] >= 0) & (indices[:, 2] < D)
    # )
    # valid_idx = indices[valid]                          # (K,3)

    # # Vectorized material gathering (squeeze to 1D first, then index all at once)
    # mats = materials.squeeze()
    # assert mats.ndim == 1, "materials should be (N,) or (N,1)"
    # mats_valid = mats[valid].to(torch.long)             # (K,)

    # # Fill voxels and materials (vectorized, no loop)
    # material_grid = torch.zeros_like(voxel_grid, dtype=torch.long, device=device)
    # voxel_grid[valid_idx[:, 0], valid_idx[:, 1], valid_idx[:, 2]] = 1
    # material_grid[valid_idx[:, 0], valid_idx[:, 1], valid_idx[:, 2]] = mats_valid + 1

    # return voxel_grid, material_grid, valid_idx
    return voxel_grid, material_grid, valid_indices

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

    plt.savefig('opacity.png', dpi=300, bbox_inches='tight')  # Save as PNG format, 300 DPI

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
    # Get the coordinates of all voxels with value 1 in the grid
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
    # Get the coordinates of all voxels with value 1 in the grid
    x, y, z = np.where(voxel_grid.cpu() != 0)
    # print(x.device, min_coords.device)

    # Calculate the actual coordinates of each point (based on voxel grid and voxel size)
    points = np.vstack((x, y, z)).T * voxel_size.cpu().numpy() + min_coords.cpu().numpy()  # Restore to actual spatial coordinates


    # Create Open3D point cloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    
    return point_cloud

def voxel_grid_to_point_cloud_color(voxel_grid, voxel_size, min_coords):
    # Get the coordinates of all voxels with non-zero values in the grid
    x, y, z = np.where(voxel_grid.cpu() != 0)
    
    # Calculate the actual coordinates of each point (based on voxel grid and voxel size)
    points = np.vstack((x, y, z)).T * voxel_size.cpu().numpy() + min_coords.cpu().numpy()  # Restore to actual spatial coordinates

    # Get voxel values (corresponding to different colors)
    voxel_values = voxel_grid[x, y, z].cpu().numpy()
    
    # Assign colors to each point; assuming voxel_values are in the [0, 1] range, we can use them to generate colors
    # Here we use a simple linear mapping: map voxel values to colors in the [0, 1] range
    # colors = plt.cm.jet(voxel_values)[:, :3]  # Use jet colormap and discard the alpha channel (last column)
    colors = np.zeros((len(voxel_values), 3))  # Initialize color array

    # Assign colors based on voxel values
    colors[voxel_values == 1] = [1, 0, 0]  # Set voxels with value 1 to red
    colors[voxel_values == 6] = [0, 1, 0]  # Set voxels with value 6 to green
    colors[voxel_values == 12] = [0, 0, 1]

    # Create Open3D point cloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    
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
            rendering_np = (rendering.permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
            cv2.imwrite(os.path.join(render_path, view.image_name + ".jpg"), rendering_np)
        cv2.imwrite(os.path.join(render_depth_path, view.image_name + ".jpg"), depth_color)
        cv2.imwrite(os.path.join(render_normal_path, view.image_name + ".jpg"), normal)


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool,
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
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=4.0*voxel_size,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), scene, gaussians, pipeline, background, 
                       max_depth=max_depth, volume=volume, use_depth_filter=use_depth_filter)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), scene, gaussians, pipeline, background)
  
def exp_depth(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        # points_fg_num = torch.zeros((gaussians.get_xyz.size()[0], 2), dtype=torch.uint32, device=)
        points = gaussians.get_xyz
        points_fg_num = torch.zeros((gaussians.get_xyz.size()[0], 2), dtype=torch.int32, device=points.device)
        ones = torch.ones((points.size()[0], 1), device=points.device)
        points_h = torch.cat([points, ones], dim=1) # [N, 4]
        views = scene.getTrainCameras()
        for idx, view in enumerate(tqdm(views, desc="Exracting foreground")):
            # print(gaussians.get_xyz.size()) # [N, 3]
            camera_coords = (view.world_view_transform.T @ points_h.T).T # [N, 4]
            np.savetxt(os.path.join(scene.model_path, "train", "ours_30000", "pc_c", f"point_cloud_c_{idx}.txt"), camera_coords[:, 0:3].cpu().numpy())
            # np.savetxt(os.path.join(scene.model_path, "train", "ours_30000", "pc_c", f"point_cloud_w_{idx}.txt"), points.cpu().numpy())
            depths = camera_coords[:, 2:3] 
            # print(depths[:5])
            
            # ndc_coords = (view.projection_matrix @ camera_coords.T).T # [N, 4]
            # ndc_coords = (view.full_proj_transform.T @ points_h.T).T
            # # print(ndc_coords[:5, ...])
            # ndc_coords[:, :3] /= ndc_coords[:, 3:4]  # (N, 3)  -->  Normalize to [-1, 1] range

            # u = (ndc_coords[:, 0] + 1) * 0.5 * view.image_width   # Map to [0, width]
            # v = (1 - ndc_coords[:, 1]) * 0.5 * view.image_height  # Map to [0, height]

            # pixels = torch.stack([u, v], dim=1)  # (N, 2)
            # print(pixels[:5, ...])
            
            depth_path = os.path.join(scene.model_path, "train", "ours_30000", "depth", '{0:05d}'.format(idx) + ".npy")
            depth_map = torch.tensor(np.load(depth_path), dtype=torch.float32, device=points.device)
            
            # visualize depth map in 3d
            H, W = depth_map.shape
            fx = fov2focal(view.FoVx, W)
            fy = fov2focal(view.FoVy, H)
            cx = W / 2 - 0.5
            cy = H / 2 - 0.5
            K = torch.tensor([[fx, 0, cx],
                            [0, fy, cy],
                            [0, 0, 1]]).float().cuda()
            xyz_map, nonzero_depth_mask = depth_to_xyz_map(depth_map, H, W, K)  # [h,w,3]
            point_cloud_c = xyz_map.view(-1, 3)  # [h*w, 3]
            np.savetxt(os.path.join(scene.model_path, "train", "ours_30000", "pc_c_d", f"point_cloud_c_depth_{idx}.txt"), point_cloud_c.cpu().numpy())          



# @ti.func
# def compute_density(index, pos, opacity, cov, grid_dx):
#     gaussian_weight = 0.0
#     for i in range(0, 2):
#         for j in range(0, 2):
#             for k in range(0, 2):
#                 node_pos = (index + ti.Vector([i, j, k])) * grid_dx
#                 dist = pos - node_pos
#                 gaussian_weight += ti.exp(-0.5 * dist.dot(cov @ dist))

#     return opacity * gaussian_weight / 8.0

# @ti.kernel
# def densify_grids(
#     init_particles: ti.template(),
#     opacity: ti.template(),
#     cov_upper: ti.template(),
#     grid: ti.template(),
#     grid_density: ti.template(),
#     grid_dx: float,
# ):
#     for pi in range(init_particles.shape[0]):
#         # print(pi)
#         pos = init_particles[pi]
#         x = pos[0]
#         y = pos[1]
#         z = pos[2]
#         i = ti.floor(x / grid_dx, dtype=int)
#         j = ti.floor(y / grid_dx, dtype=int)
#         k = ti.floor(z / grid_dx, dtype=int)
#         # ti.atomic_add(grid[i, j, k], 1)
#         cov = ti.Matrix(
#             [
#                 [cov_upper[pi][0], cov_upper[pi][1], cov_upper[pi][2]],
#                 [cov_upper[pi][1], cov_upper[pi][3], cov_upper[pi][4]],
#                 [cov_upper[pi][2], cov_upper[pi][4], cov_upper[pi][5]],
#             ]
#         )
        
#         # Check for NaN or inf
#         valid = True
#         for p in ti.static(range(3)):
#             for q in ti.static(range(3)):
#                 if ti.math.isnan(cov[p, q]) or ti.math.isinf(cov[p, q]):
#                     valid = False
#         if not valid:
#             print(f'[ERROR] Particle {pi} has invalid value in cov')
#             continue  # skip this particle
        
#         sig, Q = ti.sym_eig(cov)
#         sig[0] = ti.max(sig[0], 1e-8)
#         sig[1] = ti.max(sig[1], 1e-8)
#         sig[2] = ti.max(sig[2], 1e-8)
#         sig_mat = ti.Matrix(
#             [[1.0 / sig[0], 0, 0], [0, 1.0 / sig[1], 0], [0, 0, 1.0 / sig[2]]]
#         )
#         cov = Q @ sig_mat @ Q.transpose()
#         r = 0.0
#         for idx in ti.static(range(3)):
#             if sig[idx] < 0:
#                 sig[idx] = ti.sqrt(-sig[idx])
#             else:
#                 sig[idx] = ti.sqrt(sig[idx])

#             r = ti.max(r, sig[idx])

#         r = ti.ceil(r / grid_dx, dtype=int)
#         for dx in range(-r, r + 1):
#             for dy in range(-r, r + 1):
#                 for dz in range(-r, r + 1):
#                     if (
#                         i + dx >= 0
#                         and i + dx < grid_density.shape[0]
#                         and j + dy >= 0
#                         and j + dy < grid_density.shape[1]
#                         and k + dz >= 0
#                         and k + dz < grid_density.shape[2]
#                     ):
#                         density = compute_density(
#                             ti.Vector([i + dx, j + dy, k + dz]),
#                             pos,
#                             opacity[pi],
#                             cov,
#                             grid_dx,
#                         )
#                         ti.atomic_add(grid_density[i + dx, j + dy, k + dz], density)
#         # print(2)
                        
# @ti.kernel
# def fill_dense_grids(
#     grid: ti.template(),
#     grid_density: ti.template(),
#     grid_dx: float,
#     density_thres: float,
#     # new_particles: ti.template(),
#     # start_idx: int,
#     # max_particles_per_cell: int,
# ):
#     # new_start_idx = start_idx
#     for i, j, k in grid_density:
#         if grid_density[i, j, k] > density_thres:
#             grid[i, j, k] = 1
#             # if grid[i, j, k] < max_particles_per_cell:
#             #     diff = max_particles_per_cell - grid[i, j, k]
#             #     grid[i, j, k] = max_particles_per_cell
#             #     tmp_start_idx = ti.atomic_add(new_start_idx, diff)

#             #     for index in range(tmp_start_idx, tmp_start_idx + diff):
#             #         di = ti.random()
#             #         dj = ti.random()
#             #         dk = ti.random()
#             #         new_particles[index] = ti.Vector([i + di, j + dj, k + dk]) * grid_dx

#     # return new_start_idx
    
# @ti.func
# def collision_search(
#     grid: ti.template(), grid_density: ti.template(), index, dir_type, size, threshold
# ) -> bool:
#     dir = ti.Vector([0, 0, 0])
#     if dir_type == 0:
#         dir[0] = 1
#     elif dir_type == 1:
#         dir[0] = -1
#     elif dir_type == 2:
#         dir[1] = 1
#     elif dir_type == 3:
#         dir[1] = -1
#     elif dir_type == 4:
#         dir[2] = 1
#     elif dir_type == 5:
#         dir[2] = -1

#     flag = False
#     index += dir
#     i, j, k = index
#     while ti.max(i, j, k) < size and ti.min(i, j, k) >= 0:
#         if grid_density[index] > threshold:
#             flag = True
#             break
#         index += dir
#         i, j, k = index

#     return flag


# @ti.func
# def collision_times(
#     grid: ti.template(), grid_density: ti.template(), index, dir_type, size, threshold
# ) -> int:
#     dir = ti.Vector([0, 0, 0])
#     times = 0
#     if dir_type > 5 or dir_type < 0:
#         times = 1
#     else:
#         if dir_type == 0:
#             dir[0] = 1
#         elif dir_type == 1:
#             dir[0] = -1
#         elif dir_type == 2:
#             dir[1] = 1
#         elif dir_type == 3:
#             dir[1] = -1
#         elif dir_type == 4:
#             dir[2] = 1
#         elif dir_type == 5:
#             dir[2] = -1

#         state = grid[index] > 0
#         index += dir
#         i, j, k = index
#         while ti.max(i, j, k) < size and ti.min(i, j, k) >= 0:
#             new_state = grid_density[index] > threshold
#             if new_state != state and state == False:
#                 times += 1
#             state = new_state
#             index += dir
#             i, j, k = index

#     return times
    
# @ti.kernel
# def internal_filling(
#     grid: ti.template(),
#     grid_density: ti.template(),
#     grid_dx: float,
#     # new_particles: ti.template(),
#     # start_idx: int,
#     # max_particles_per_cell: int,
#     exclude_dir: int,
#     ray_cast_dir: int,
#     threshold: float,
# ):
#     # new_start_idx = start_idx
#     for i, j, k in grid:
#         if grid[i, j, k] == 0:
#             collision_hit = True
#             for dir_type in ti.static(range(6)):
#                 if dir_type != exclude_dir:
#                     hit_test = collision_search(
#                         grid=grid,
#                         grid_density=grid_density,
#                         index=ti.Vector([i, j, k]),
#                         dir_type=dir_type,
#                         size=grid.shape[0],
#                         threshold=threshold,
#                     )
#                     collision_hit = collision_hit and hit_test

#             if collision_hit:
#                 hit_times = collision_times(
#                     grid=grid,
#                     grid_density=grid_density,
#                     index=ti.Vector([i, j, k]),
#                     dir_type=ray_cast_dir,
#                     size=grid.shape[0],
#                     threshold=threshold,
#                 )

#                 if ti.math.mod(hit_times, 2) == 1:
#                     # diff = max_particles_per_cell - grid[i, j, k]
#                     grid[i, j, k] = 1
#                     # tmp_start_idx = ti.atomic_add(new_start_idx, diff)
#                     # for index in range(tmp_start_idx, tmp_start_idx + diff):
#                     #     di = ti.random()
#                     #     dj = ti.random()
#                     #     dk = ti.random()
#                     #     new_particles[index] = (
#                     #         ti.Vector([i + di, j + dj, k + dk]) * grid_dx
#                         # )

torch.no_grad()
def concat_materials_by_nn(
    points: torch.Tensor,       # (N,3)
    materials: torch.Tensor,    # (N,) or (N,1)
    new_pos: torch.Tensor,      # (M,3)
    batch_points: int = 200_000 # Batch cdist computation to avoid GPU memory overflow
):
    """
    Returns:
      all_materials: (N+M,) or original shape (if input is (N,1) then returns (N+M,1))
      nn_idx:        (M,)  index of the nearest neighbor in points for each new_pos
    """
    if new_pos is None or new_pos.numel() == 0:
        # No new points, return original materials directly
        return materials.clone(), torch.empty(0, dtype=torch.long, device=materials.device)

    # Unify device / dtype / shape
    device = points.device
    dtype  = points.dtype
    points  = points.contiguous()
    new_pos = new_pos.to(device=device, dtype=dtype).contiguous()

    mats = materials.to(device)
    squeeze_back = False
    if mats.ndim == 2 and mats.shape[1] == 1:
        mats = mats.view(-1)      # Reshape to (N,)
        squeeze_back = True
    elif mats.ndim != 1:
        raise ValueError("materials must be of shape (N,) or (N,1)")

    assert points.shape[0] == mats.shape[0], "points and materials count mismatch"
    assert points.shape[1] == 3 and new_pos.shape[1] == 3, "points/new_pos must be of shape (.,3)"

    # Batched nearest neighbor search
    M = new_pos.shape[0]
    idx_chunks = []
    for s in range(0, M, batch_points):
        e = min(s + batch_points, M)
        # (bs, N) Euclidean distance
        dist = torch.cdist(new_pos[s:e], points, p=2)
        nn_idx = torch.argmin(dist, dim=1)   # (bs,)
        idx_chunks.append(nn_idx)
        del dist

    nn_idx = torch.cat(idx_chunks, dim=0)     # (M,)
    new_mats = mats[nn_idx]                    # (M,)

    # Concatenate
    all_mats = torch.cat([mats, new_mats], dim=0)  # (N+M,)
    # if squeeze_back:
    # all_mats = all_mats.view(-1, 1)           # Restore to (N+M,1)
    print("all_mats shape: ", all_mats.shape)

    return all_mats #, nn_idx           

def extract_fg_filling(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        # points_fg_num = torch.zeros((gaussians.get_xyz.size()[0], 2), dtype=torch.uint32, device=)
        points = gaussians.get_xyz
        points_fg_num = torch.zeros((gaussians.get_xyz.size()[0], 2), dtype=torch.int32, device=points.device)
        ones = torch.ones((points.size()[0], 1), device=points.device)
        burnability = torch.from_numpy(np.load(os.path.join(scene.model_path, "points_burnability_knn.npy"))).cuda()
        materials = torch.from_numpy(np.load(os.path.join(scene.model_path, "points_materials_knn.npy"))).cuda()
        
        selected_points = (burnability == 0)
        print(selected_points.sum())
        
        points = gaussians.get_xyz
        print(points.shape[0])

        # Read bounding box from YAML config
        new_min_coords = np.array(args.bounding_box['min'])
        new_max_coords = np.array(args.bounding_box['max'])
        print(f"Bounding Box from config: min={new_min_coords}, max={new_max_coords}")
        in_voxel_mask = (
            (points[:, 0] >= new_min_coords[0]) & (points[:, 0] <= new_max_coords[0]) &
            (points[:, 1] >= new_min_coords[1]) & (points[:, 1] <= new_max_coords[1]) &
            (points[:, 2] >= new_min_coords[2]) & (points[:, 2] <= new_max_coords[2])
        )
        selected_points = selected_points & in_voxel_mask
        
        materials = materials[selected_points]
        print(materials.shape)
        
        # gaussians.crop_setup(selected_points)
        # save_path = os.path.join(scene.model_path, "fg", "fg_opac_0_burnable.ply")
        # gaussians.save_ply(save_path)
        
        points = gaussians.get_xyz
        print(points.shape[0])
        cov = gaussians.get_covariance()
        opacity = gaussians.get_opacity
        print("median opacity: ", np.median(opacity.cpu().numpy()))
        print(cov.shape)
        
        voxel_grid, grid_size = create_voxel_grid(new_min_coords, new_max_coords, voxel_size=256)
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
        init_pos = gaussians.get_xyz[selected_points].detach()
        init_cov = gaussians.get_covariance()
        init_cov = init_cov[selected_points].detach()
        init_opacity = gaussians.get_opacity[selected_points].detach()
        init_shs = gaussians.get_opacity[selected_points].detach()
        print(init_pos.shape)
        print(args.voxel_grid['dims'][0])
        print(grid_size[0].item())
        # print(args.voxel_grid['voxel_size'])
        new_pos, occ_grid = fill_particles(
            pos=init_pos,
            opacity=init_opacity,
            cov=init_cov,
            grid_n=args.voxel_grid['dims'][0],
            max_samples=500000,
            grid_dx=grid_size[0].item(),
            density_thres=0.05,
            search_thres=0.05,
            max_particles_per_cell=4,
            search_exclude_dir=5,
            ray_cast_dir=4,
            boundary=new_min_coords.tolist()+new_max_coords.tolist(),
            smooth=False,
        )
        print(new_pos.shape)
        materials = concat_materials_by_nn(init_pos, materials, new_pos)
        print(materials.shape)
        gaussians.append_gaussians_from_new_xyz_with_masked_sources(new_pos, selected_points)
        save_path = os.path.join(scene.model_path, "filled_gaussian.ply")
        gaussians.save_ply(save_path)
        # origin = torch.tensor([new_min_coords[0], new_min_coords[1], new_min_coords[2]]).cuda()
        # pos = points - origin
        # ti_pos = ti.Vector.field(n=3, dtype=float, shape=pos.shape[0])
        # ti_opacity = ti.field(dtype=float, shape=opacity.shape[0])
        # ti_cov = ti.Vector.field(n=6, dtype=float, shape=cov.shape[0])
        # ti_pos.from_torch(pos.reshape(-1, 3))
        # ti_opacity.from_torch(opacity.reshape(-1))
        # ti_cov.from_torch(cov.reshape(-1, 6))
        
        # grid_n = args.voxel_size
        # grid = ti.field(dtype=int, shape=(grid_n, grid_n, grid_n))
        # grid_density = ti.field(dtype=float, shape=(grid_n, grid_n, grid_n))
        # # particles = ti.Vector.field(n=3, dtype=float, shape=max_samples)
        # fill_num = 0
        
        # densify_grids(ti_pos, ti_opacity, ti_cov, grid, grid_density, grid_size[0])
        # print("grid densified")
        # dens_thres = 0.01
        # print(grid_density.to_numpy().min(), grid_density.to_numpy().max())
        # fill_dense_grids(grid, grid_density, grid_size[0], density_thres=dens_thres)
        # filled_grid = grid.to_numpy()
        # print("filled grid num: ", (filled_grid==1).sum())
        # internal_filling(grid, grid_density, grid_size[0], exclude_dir=5,  # 0: x, 1: -x, 2: y, 3: -y, 4: z, 5: -z direction
        #                                                 ray_cast_dir=4,  # 0: x, 1: -x, 2: y, 3: -y, 4: z, 5: -z direction
        #                                                 threshold=dens_thres,)
        
        filled_grid = occ_grid.to_numpy()
        print("filled grid num: ", (filled_grid>0).sum())
        
        voxel_grid_path = os.path.join(scene.model_path, "filled_voxel_grid.npy")
        np.save(voxel_grid_path,filled_grid)
        # import yaml
        # from pathlib import Path
        # cfg_path = Path(os.path.join(scene.model_path, "simulation_voxel.yaml"))
        # cfg_path.write_text(yaml.safe_dump(config, sort_keys=False))
        # print(f"Config saved to {cfg_path.resolve()}")
        new_mask = torch.ones(new_pos.shape[0], dtype=torch.bool).to(selected_points.device)
        selected_points = torch.cat([selected_points, new_mask], dim=0)
        voxel_grid, material_grid, indices_pts_in_grids = map_points_to_voxels_materials(gaussians.get_xyz[selected_points], torch.tensor(new_min_coords).cuda(), gaussians.get_opacity[selected_points], torch.tensor(grid_size).cuda(), voxel_grid, materials)
        
        # print_voxel_statistics(voxel_grid)
        
        # visualize_voxel_grid(voxel_grid=voxel_grid)
        
        point_cloud = voxel_grid_to_point_cloud(voxel_grid=torch.tensor(filled_grid), voxel_size=torch.tensor(grid_size), min_coords=torch.tensor(new_min_coords))
        voxel_path = os.path.join(scene.model_path, "voxel_filled_new.ply")
        o3d.io.write_point_cloud(voxel_path, point_cloud)
        point_cloud = voxel_grid_to_point_cloud_color(voxel_grid=material_grid, voxel_size=torch.from_numpy(grid_size), min_coords=torch.from_numpy(new_min_coords))
        material_path = os.path.join(scene.model_path, "material_voxel_filled.ply")
        o3d.io.write_point_cloud(material_path, point_cloud)
        # voxel_grid_path = os.path.join(scene.model_path, "voxel_grid.npy")
        # np.save(voxel_grid_path,voxel_grid)
        # material_grid_path = os.path.join(scene.model_path, "material_grid.npy")
        # np.save(material_grid_path,material_grid)
        # voxel = np.load(voxel_grid_path)
        # print(voxel.shape)
        # voxel = np.load(material_grid_path)
        # print(np.unique(voxel))
        # # assert voxel.shape == (128, 128, 128)
        
        print("save indices_pts_in_grids")  # Grid coordinates of each Gaussian point in the grid
        indices_pts_in_grids_path = os.path.join(scene.model_path, "indices_pts_in_grids_filled.pth")
        torch.save(indices_pts_in_grids, indices_pts_in_grids_path)

        # print("save indices_pts_ingrids_for_inall")   # Index of each Gaussian point in the grid among all Gaussians
        # indices_pts_ingrids_for_inall_path = os.path.join(scene.model_path, "indices_pts_ingrids_for_inall.pth")
        # indices_pts_ingrids_for_inall = torch.nonzero(selected_points, as_tuple=False).squeeze()
        # torch.save(indices_pts_ingrids_for_inall, indices_pts_ingrids_for_inall_path)

        print("save selected_points")  # Mask of each Gaussian point in the grid among all Gaussians
        mask_pts_in_grids_path = os.path.join(scene.model_path, "mask_pts_in_grids_filled.pth")
        torch.save(selected_points, mask_pts_in_grids_path)
        
                    
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
        # for idx, view in enumerate(tqdm(views, desc="Exracting foreground")):
        #     # print(gaussians.get_xyz.size()) # [N, 3]
        #     camera_coords = (view.world_view_transform.T @ points_h.T).T # [N, 4]
        #     # np.savetxt(os.path.join(scene.model_path, "train", "ours_30000", "pc_c", f"point_cloud_c_{idx}.txt"), camera_coords[:, 0:3].cpu().numpy())
        #     # np.savetxt(os.path.join(scene.model_path, "train", "ours_30000", "pc_c", f"point_cloud_w_{idx}.txt"), points.cpu().numpy())
        #     depths = camera_coords[:, 2:3] 
        #     # print(depths[:5])
            
        #     # ndc_coords = (view.projection_matrix @ camera_coords.T).T # [N, 4]
        #     # ndc_coords = (view.full_proj_transform.T @ points_h.T).T
        #     # # print(ndc_coords[:5, ...])
        #     # ndc_coords[:, :3] /= ndc_coords[:, 3:4]  # (N, 3)  -->  Normalize to [-1, 1] range

        #     # u = (ndc_coords[:, 0] + 1) * 0.5 * view.image_width   # Map to [0, width]
        #     # v = (1 - ndc_coords[:, 1]) * 0.5 * view.image_height  # Map to [0, height]

        #     # pixels = torch.stack([u, v], dim=1)  # (N, 2)
        #     # print(pixels[:5, ...])
            
        #     depth_path = os.path.join(scene.model_path, "train", "ours_30000", "depth", '{0:05d}'.format(idx) + ".npy")
        #     depth_map = torch.tensor(np.load(depth_path), dtype=torch.float32, device=points.device)
            
        #     # visualize depth map in 3d
        #     H, W = depth_map.shape
        #     fx = fov2focal(view.FoVx, W)
        #     fy = fov2focal(view.FoVy, H)
        #     cx = W / 2 - 0.5
        #     cy = H / 2 - 0.5
        #     K = torch.tensor([[fx, 0, cx],
        #                     [0, fy, cy],
        #                     [0, 0, 1]]).float().cuda()
        #     xyz_map, nonzero_depth_mask = depth_to_xyz_map(depth_map, H, W, K)  # [h,w,3]
        #     point_cloud_c = xyz_map.view(-1, 3)  # [h*w, 3]
        #     # np.savetxt(os.path.join(scene.model_path, "train", "ours_30000", "pc_c_d", f"point_cloud_c_depth_{idx}.txt"), point_cloud_c.cpu().numpy())
            
        #     u,v = xyz_to_uv(camera_coords[:, :3], K)
            
        #     mask_valid = (u >= 0) & (u < depth_map.shape[1]) & (v >= 0) & (v < depth_map.shape[0])  # (N,)

            
        #     depth_values =  depth_map[v.int()[mask_valid], u.int()[mask_valid]]
        #     depths_valid = depths[mask_valid].squeeze()
        #     visible_mask = (depths_valid <=  depth_values) &  (depths_valid >= 0)  # (N,)
        #     # print(torch.sum(visible_mask))
            
        #     visible_mask_all = torch.zeros(points.size()[0], dtype=torch.bool, device=points.device)  # [N]

        #     visible_mask_all[mask_valid] = visible_mask  # Update visibility of valid points into full_mask

        #     # print(f"Full visible mask size: {visible_mask_all.sum()}")
            
        #     mask_path = os.path.join(fg_mask_dir, view.image_name + ".JPG")
        #     mask_image = torch.tensor(imageio.imread(mask_path), dtype=torch.float32, device=points.device)  # [H, W]
            
        #     mask_values = mask_image[v.int()[visible_mask_all], u.int()[visible_mask_all]]
        #     fg_mask = mask_values > 128
            
        #     # print(fg_mask.sum())
            
        #     all_mask = torch.zeros(points.size()[0], dtype=torch.bool, device=points.device)
        #     all_mask[visible_mask_all] = fg_mask
        #     all_mask_ = torch.zeros(points.size()[0], dtype=torch.bool, device=points.device)
        #     all_mask_[visible_mask_all] = ~fg_mask
        #     # print(all_mask.size())
        #     points_fg_num[all_mask, 0] += 1
        #     # print(points_fg_num[all_mask, 0])
        #     points_fg_num[all_mask_, 1] +=1
            
            
        # # print(points_fg_num[:10, ...])  
        opacities = gaussians.get_opacity.squeeze()
        # selected_points = (points_fg_num[:, 0] > points_fg_num[:, 1])
        # print(selected_points.size())
        # print(selected_points.sum())
        # print(selected_points[:10])
        burnability = torch.from_numpy(np.load(os.path.join(scene.model_path, "points_burnability_knn.npy"))).cuda()
        materials = torch.from_numpy(np.load(os.path.join(scene.model_path, "points_materials_knn.npy"))).cuda()

        points = gaussians.get_xyz
        num_points = points.shape[0]
        print(f"Gaussians: {num_points}, burnability: {burnability.shape[0]}")

        # Read initial tight bounding box from YAML config, then expand
        voxel_cfg = args.extract_voxel
        min_coords = torch.tensor(voxel_cfg['bbox_min'], device=points.device)
        max_coords = torch.tensor(voxel_cfg['bbox_max'], device=points.device)
        scale_factor = voxel_cfg.get('scale_factor', args.scale_factor)
        print(f"Initial Bounding Box: min={min_coords}, max={max_coords}")
        new_min_coords, new_max_coords = expand_bounding_box(min_coords, max_coords, scale_factor=scale_factor)
        print(f"Expanded Bounding Box: min={new_min_coords}, max={new_max_coords}")

        in_voxel_mask = (
            (points[:, 0] >= new_min_coords[0]) & (points[:, 0] <= new_max_coords[0]) &
            (points[:, 1] >= new_min_coords[1]) & (points[:, 1] <= new_max_coords[1]) &
            (points[:, 2] >= new_min_coords[2]) & (points[:, 2] <= new_max_coords[2])
        )

        selected_points = (burnability == 0) & in_voxel_mask
        print(selected_points.sum())

        materials = materials[selected_points]
        print(materials.shape)
        
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
        print(f"Config saved to {cfg_path.resolve()}")
        
        voxel_grid, material_grid, indices_pts_in_grids = map_points_to_voxels_materials(points, new_min_coords, gaussians.get_opacity, grid_size, voxel_grid, materials)
        
        print_voxel_statistics(voxel_grid)

        visualize_voxel_grid(voxel_grid=voxel_grid)

        # Save core outputs (npy)
        voxel_grid_path = os.path.join(scene.model_path, "voxel_grid.npy")
        np.save(voxel_grid_path, voxel_grid)
        material_grid_path = os.path.join(scene.model_path, "material_grid.npy")
        np.save(material_grid_path, material_grid)

        # Save point cloud visualizations as PLY (using plyfile to avoid Open3D segfaults)
        from plyfile import PlyData, PlyElement

        def _save_voxel_ply(grid, path, voxel_size, min_coords, colors=None):
            x, y, z = np.where(grid.cpu().numpy() != 0)
            pts = np.vstack((x, y, z)).T * voxel_size.cpu().numpy() + min_coords.cpu().numpy()
            if colors is not None:
                dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
                arr = np.empty(len(pts), dtype=dtype)
                arr['x'], arr['y'], arr['z'] = pts[:, 0], pts[:, 1], pts[:, 2]
                c = (colors * 255).astype(np.uint8)
                arr['red'], arr['green'], arr['blue'] = c[:, 0], c[:, 1], c[:, 2]
            else:
                dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
                arr = np.empty(len(pts), dtype=dtype)
                arr['x'], arr['y'], arr['z'] = pts[:, 0], pts[:, 1], pts[:, 2]
            PlyData([PlyElement.describe(arr, 'vertex')]).write(path)
            print(f"Saved {path} ({len(pts)} points)")

        voxel_path = os.path.join(scene.model_path, "voxel.ply")
        _save_voxel_ply(voxel_grid, voxel_path, grid_size, new_min_coords)

        # Material voxel with colors
        x, y, z = np.where(material_grid.cpu().numpy() != 0)
        voxel_values = material_grid[x, y, z].cpu().numpy()
        colors = np.zeros((len(voxel_values), 3))
        colors[voxel_values == 1] = [1, 0, 0]
        colors[voxel_values == 6] = [0, 1, 0]
        colors[voxel_values == 12] = [0, 0, 1]
        material_path = os.path.join(scene.model_path, "material_voxel.ply")
        _save_voxel_ply(material_grid, material_path, grid_size, new_min_coords, colors=colors)
        voxel = np.load(voxel_grid_path)
        print(voxel.shape)
        voxel = np.load(material_grid_path)
        print(np.unique(voxel))
        # assert voxel.shape == (128, 128, 128)
        
        print("save indices_pts_in_grids")  # Grid coordinates of each Gaussian point in the grid
        indices_pts_in_grids_path = os.path.join(scene.model_path, "indices_pts_in_grids.pth")
        torch.save(indices_pts_in_grids, indices_pts_in_grids_path)

        # print("save indices_pts_ingrids_for_inall")   # Index of each Gaussian point in the grid among all Gaussians
        # indices_pts_ingrids_for_inall_path = os.path.join(scene.model_path, "indices_pts_ingrids_for_inall.pth")
        # indices_pts_ingrids_for_inall = torch.nonzero(selected_points, as_tuple=False).squeeze()
        # torch.save(indices_pts_ingrids_for_inall, indices_pts_ingrids_for_inall_path)

        print("save selected_points")  # Mask of each Gaussian point in the grid among all Gaussians
        mask_pts_in_grids_path = os.path.join(scene.model_path, "mask_pts_in_grids.pth")
        torch.save(selected_points, mask_pts_in_grids_path)


import open3d as o3d
import numpy as np

def extract_voxel_occupancy_from_tsdf(volume, bbox_min, bbox_max, voxel_dim, voxel_size):
    # Initialize occupancy grid
    occupancy_grid = np.zeros(voxel_dim, dtype=np.uint8)
    
    # Extract voxel grid
    voxel_grid = volume.extract_voxel_grid()
    voxels = voxel_grid.get_voxels()
    origin = np.array(voxel_grid.origin)
    voxel_tsdf_size = voxel_grid.voxel_size  # Note: this may differ from the voxel_size we set
    
    # Store point cloud of occupied voxels
    occupied_points = []
    
    # Iterate over all voxels
    for voxel in voxels:
        world_coord = origin + np.array(voxel.grid_index) * voxel_tsdf_size

        # Only process voxels within the given bbox range
        # if np.all(world_coord >= bbox_min) and np.all(world_coord <= bbox_max):
        #     local_index = ((world_coord - bbox_min) / voxel_size).astype(int)
        #     if np.all(local_index >= 0) and np.all(local_index < voxel_dim):
        if voxel.tsdf < 0:
                    # occupancy_grid[tuple(local_index)] = 1
            occupied_points.append(world_coord)  # Store the world coordinates of occupied voxels
    
    # Convert occupied point coordinates to Open3D point cloud
    occupied_points = np.array(occupied_points)
    print(occupied_points.shape)
    voxel_point_cloud = o3d.geometry.PointCloud()
    voxel_point_cloud.points = o3d.utility.Vector3dVector(occupied_points)
    
    print(f"Number of occupied voxels: {np.sum(occupancy_grid)}")
    o3d.io.write_point_cloud("occ.ply", voxel_point_cloud)
    
    return occupancy_grid



def occupancy_to_point_cloud(occupancy_grid, bbox_min, voxel_size):
    # Get the coordinates of all occupied voxels
    occupied = np.argwhere(occupancy_grid > 0)
    points = occupied * voxel_size + bbox_min  # Convert to world coordinates
    

    # Create point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([1, 0, 0])  # Red indicates occupied

    return pcd
    
        
def extract_fg_tsdf(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, voxel_size: float=0.01):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        # points_fg_num = torch.zeros((gaussians.get_xyz.size()[0], 2), dtype=torch.uint32, device=)
        points = gaussians.get_xyz
        points_fg_num = torch.zeros((gaussians.get_xyz.size()[0], 2), dtype=torch.int32, device=points.device)
        ones = torch.ones((points.size()[0], 1), device=points.device)
        points_h = torch.cat([points, ones], dim=1) # [N, 4]
        views = scene.getTrainCameras()
        bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=4.0*voxel_size,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        
        # volume = o3d.pipelines.integration.UniformTSDFVolume(
        #     length=3,  # e.g., 3.0 meter cube
        #     resolution=int(3/ voxel_size),
        #     sdf_trunc=4.0 * voxel_size,
        #     color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        # )
        
        depths_tsdf_fusion = []
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            out = render_PGSR(view, gaussians, pipeline, background, app_model=None)
            rendering = out["render"].clamp(0.0, 1.0)
            _, H, W = rendering.shape
            depth_path = os.path.join(scene.model_path, "train", "ours_30000", "depth", '{0:05d}'.format(idx) + ".npy")
            depth = torch.tensor(np.load(depth_path), dtype=torch.float32, device=points.device)
            depth_tsdf = depth.clone()
            # depth = depth.detach().cpu().numpy()
            # depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
            # depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
            # depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)

            # normal = out["rendered_normal"].permute(1,2,0)
            # normal = normal/(normal.norm(dim=-1, keepdim=True)+1.0e-8)
            # normal = normal.detach().cpu().numpy()
            # normal = ((normal+1) * 127.5).astype(np.uint8).clip(0, 255)

            # if use_depth_filter:
            #     view_dir = torch.nn.functional.normalize(view.get_rays(), p=2, dim=-1)
            #     depth_normal = out["depth_normal"].permute(1,2,0)
            #     depth_normal = torch.nn.functional.normalize(depth_normal, p=2, dim=-1)
            #     dot = torch.sum(view_dir*depth_normal, dim=-1).abs()
            #     angle = torch.acos(dot)
            #     mask = angle > (80.0 / 180 * 3.14159)
            #     depth_tsdf[mask] = 0
            depths_tsdf_fusion.append(depth_tsdf.squeeze().cpu())
            
        depths_tsdf_fusion = torch.stack(depths_tsdf_fusion, dim=0)
        for idx, view in enumerate(tqdm(views, desc="TSDF Fusion progress")):
            ref_depth = depths_tsdf_fusion[idx].cuda()

            # if view.mask is not None:
            #     ref_depth[view.mask.squeeze() < 0.5] = 0
            # ref_depth[ref_depth>max_depth] = 0
            ref_depth = ref_depth.detach().cpu().numpy()
            
            pose = np.identity(4)
            pose[:3,:3] = view.R.transpose(-1,-2)
            pose[:3, 3] = view.T
            assert not np.isnan(view.R).any(), f"Invalid R matrix for view {idx}"
            assert not np.isnan(view.T).any(), f"Invalid T matrix for view {idx}"
            color = o3d.io.read_image(os.path.join(scene.model_path, "train", "ours_30000", "renders", view.image_name + ".jpg"))
            color = np.asarray(color)  # Convert to numpy array

            # Check if there is an alpha channel
            if color.shape[-1] == 4:
                color = color[..., :3]  # Remove alpha channel

            # Ensure it is uint8 type
            assert color.dtype == np.uint8, "Color image must be uint8 type"
            depth = o3d.geometry.Image((ref_depth*1000).astype(np.uint16))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(color), depth, depth_scale=1000.0, depth_trunc=5, convert_rgb_to_intensity=False)
            volume.integrate(
                rgbd,
                o3d.camera.PinholeCameraIntrinsic(W, H, view.Fx, view.Fy, view.Cx, view.Cy),
                pose)
            
        # opacities = gaussians.get_opacity.squeeze()
        # burnability = torch.from_numpy(np.load(os.path.join(scene.model_path, "points_burnability_knn.npy"))).cuda()
        # materials = torch.from_numpy(np.load(os.path.join(scene.model_path, "points_materials_knn.npy"))).cuda()
        # # selected_points = selected_points & (burnability == 0)
        
        # selected_points = (burnability == 0)
        # print(selected_points.sum())
        
        # points = gaussians.get_xyz
        # print(points.shape[0])
        # points[..., -1] = -points[..., -1]
        # min_coords, max_coords = get_bounding_box(points)
        # #garden
        # max_coords = torch.tensor([1.3, 1.5, 0.0], device=points.device)
        # min_coords = torch.tensor([-2.3, -2.1, -3.5], device=points.device)
        # #garden_8 PGSR
        # max_coords = torch.tensor([1.5, 2.45, -0.15], device=points.device)
        # min_coords = torch.tensor([-2.1, -1.25, -3.38], device=points.device)
        # #firewood_sand PGSR
        # max_coords = torch.tensor([0.75, 1.25, -1.43], device=points.device)
        # min_coords = torch.tensor([-1.2, -1.05, -2.77], device=points.device)
        # print(f"Original Bounding Box: min={min_coords}, max={max_coords}")
        # # #chair PGSR
        # max_coords = torch.tensor([0.82, 0.82, -1.6], device=points.device)
        # min_coords = torch.tensor([-1.25, -1.65, -3.2], device=points.device)
        # print(f"Original Bounding Box: min={min_coords}, max={max_coords}")
        
        #  firewoods_sand PGSR
        # max_coords = torch.tensor([1.2, 0.8, -1.18], device=points.device)
        # min_coords = torch.tensor([-0.86, -1.65, -2.6], device=points.device)
        # print(f"Original Bounding Box: min={min_coords}, max={max_coords}")
        
        # firewoods_sand_dark
        # max_coords = torch.tensor([0.9, 0.9, -1.0], device=points.device)
        # min_coords = torch.tensor([-1.5, -1.5, -2.8], device=points.device)
        # print(f"Original Bounding Box: min={min_coords}, max={max_coords}")
        
        # chair_indoor
        # max_coords = torch.tensor([1.4, 1.55, -1.05], device=points.device)
        # min_coords = torch.tensor([-0.28, -0.4, -3.35], device=points.device)
        # print(f"Original Bounding Box: min={min_coords}, max={max_coords}")
        
        # kitchen
        # max_coords = torch.tensor([2, -2.3, -2.6], device=points.device)
        # min_coords = torch.tensor([-0.5, -6, -5.5], device=points.device)
        # print(f"Original Bounding Box: min={min_coords}, max={max_coords}")
        
        # playground_paper
        # max_coords = torch.tensor([0.66, 0.65, 0.3], device=points.device)
        # min_coords = torch.tensor([-0.36, -0.35, -0.5], device=points.device)
        # print(f"Original Bounding Box: min={min_coords}, max={max_coords}")
        
        # Playground
        max_coords = torch.tensor([1.1, 1.4, 0.38], device=points.device)
        min_coords = torch.tensor([-1.05, -0.81, -0.53], device=points.device)
        print(f"Original Bounding Box: min={min_coords}, max={max_coords}")

        # Expand bounding box
        new_min_coords, new_max_coords = expand_bounding_box(min_coords, max_coords, scale_factor=args.scale_factor)
        print(f"Expanded Bounding Box: min={new_min_coords}, max={new_max_coords}")
        
        # in_voxel_mask = (
        #     (points[:, 0] >= new_min_coords[0]) & (points[:, 0] <= new_max_coords[0]) &
        #     (points[:, 1] >= new_min_coords[1]) & (points[:, 1] <= new_max_coords[1]) &
        #     (points[:, 2] >= new_min_coords[2]) & (points[:, 2] <= new_max_coords[2])
        # )
        
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
        print(f"Config saved to {cfg_path.resolve()}")
        voxel_grid_occ = volume.extract_voxel_point_cloud()
        # print(voxel_grid.shape, voxel_grid.sum())
        num_points = len(voxel_grid_occ.points)  # This will give you the number of points in the point cloud
        print(f"Number of points in the point cloud: {num_points}")
        # o3d.io.write_point_cloud("occ_point_cloud.ply", voxel_grid)
        points = torch.from_numpy(np.asarray(voxel_grid_occ.points)).cuda()
        in_voxel_mask = (
            (points[:, 0] >= new_min_coords[0]) & (points[:, 0] <= new_max_coords[0]) &
            (points[:, 1] >= new_min_coords[1]) & (points[:, 1] <= new_max_coords[1]) &
            (points[:, 2] >= new_min_coords[2]) & (points[:, 2] <= new_max_coords[2])
        )
        print(f"{in_voxel_mask.sum()} points in the voxel")
        points = points[in_voxel_mask]
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points.cpu().numpy())
        # o3d.io.write_point_cloud("occ_pointcloud.ply", pcd)

        
        # occ_grid = extract_voxel_occupancy_from_tsdf(volume, np.array(config["bounding_box"]["min"]), np.array(config["bounding_box"]["max"]), config["voxel_grid"]["dims"],  np.array(config["voxel_grid"]["voxel_size"]))
        # del volume
        # occ_pointcloud = occupancy_to_point_cloud(occ_grid, np.array(config["bounding_box"]["min"]), np.array(config["voxel_grid"]["dims"]))
        # occ_path = os.path.join(scene.model_path, "voxel.ply")
        # o3d.io.write_point_cloud(occ_path, occ_pointcloud)
        voxel_grid, indices_pts_in_grids = map_points_to_voxels(points, new_min_coords, gaussians.get_opacity, grid_size, voxel_grid)
        
        # print_voxel_statistics(voxel_grid)
        
        # visualize_voxel_grid(voxel_grid=voxel_grid)
        
        point_cloud = voxel_grid_to_point_cloud(voxel_grid=voxel_grid, voxel_size=grid_size, min_coords=new_min_coords)
        voxel_path = os.path.join(scene.model_path, "occ_voxel.ply")
        o3d.io.write_point_cloud(voxel_path, point_cloud)
        # point_cloud = voxel_grid_to_point_cloud_color(voxel_grid=material_grid, voxel_size=grid_size, min_coords=new_min_coords)
        # material_path = os.path.join(scene.model_path, "material_voxel.ply")
        # o3d.io.write_point_cloud(material_path, point_cloud)
        # voxel_grid_path = os.path.join(scene.model_path, "voxel_grid.npy")
        # np.save(voxel_grid_path,voxel_grid)
        # material_grid_path = os.path.join(scene.model_path, "material_grid.npy")
        # np.save(material_grid_path,material_grid)
        # voxel = np.load(voxel_grid_path)
        # print(voxel.shape)
        # voxel = np.load(material_grid_path)
        # print(np.unique(voxel))
        # # assert voxel.shape == (128, 128, 128)
        
        # print("save indices_pts_in_grids")  # Grid coordinates of each Gaussian point in the grid
        # indices_pts_in_grids_path = os.path.join(scene.model_path, "indices_pts_in_grids.pth")
        # torch.save(indices_pts_in_grids, indices_pts_in_grids_path)

        # # print("save indices_pts_ingrids_for_inall")   # Index of each Gaussian point in the grid among all Gaussians
        # # indices_pts_ingrids_for_inall_path = os.path.join(scene.model_path, "indices_pts_ingrids_for_inall.pth")
        # # indices_pts_ingrids_for_inall = torch.nonzero(selected_points, as_tuple=False).squeeze()
        # # torch.save(indices_pts_ingrids_for_inall, indices_pts_ingrids_for_inall_path)

        # print("save selected_points")  # Mask of each Gaussian point in the grid among all Gaussians
        # mask_pts_in_grids_path = os.path.join(scene.model_path, "mask_pts_in_grids.pth")
        # torch.save(selected_points, mask_pts_in_grids_path)


def load_yaml_config(yaml_path):
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)

def update_args_with_yaml(args, yaml_config):
    for key, value in yaml_config.items():
        if not hasattr(args, key):
            setattr(args, key, value)


def extract_fg_depth(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        # points_fg_num = torch.zeros((gaussians.get_xyz.size()[0], 2), dtype=torch.uint32, device=)
        points = gaussians.get_xyz
        points_fg_num = torch.zeros((gaussians.get_xyz.size()[0], 2), dtype=torch.int32, device=points.device)
        ones = torch.ones((points.size()[0], 1), device=points.device)
        points_h = torch.cat([points, ones], dim=1) # [N, 4]
        views = scene.getTrainCameras()
        all_points = []
        for idx, view in enumerate(tqdm(views, desc="Exracting foreground")):
            # print(gaussians.get_xyz.size()) # [N, 3]
            camera_coords = (view.world_view_transform.T @ points_h.T).T # [N, 4]
            # np.savetxt(os.path.join(scene.model_path, "train", "ours_30000", "pc_c", f"point_cloud_c_{idx}.txt"), camera_coords[:, 0:3].cpu().numpy())
            # np.savetxt(os.path.join(scene.model_path, "train", "ours_30000", "pc_c", f"point_cloud_w_{idx}.txt"), points.cpu().numpy())
            depths = camera_coords[:, 2:3] 
            # print(depths[:5])
            
            # ndc_coords = (view.projection_matrix @ camera_coords.T).T # [N, 4]
            # ndc_coords = (view.full_proj_transform.T @ points_h.T).T
            # # print(ndc_coords[:5, ...])
            # ndc_coords[:, :3] /= ndc_coords[:, 3:4]  # (N, 3)  -->  Normalize to [-1, 1] range

            # u = (ndc_coords[:, 0] + 1) * 0.5 * view.image_width   # Map to [0, width]
            # v = (1 - ndc_coords[:, 1]) * 0.5 * view.image_height  # Map to [0, height]

            # pixels = torch.stack([u, v], dim=1)  # (N, 2)
            # print(pixels[:5, ...])
            
            depth_path = os.path.join(scene.model_path, "train", "ours_30000", "depth", '{0:05d}'.format(idx) + ".npy")
            depth_map = torch.tensor(np.load(depth_path), dtype=torch.float32, device=points.device)
            
            # visualize depth map in 3d
            H, W = depth_map.shape
            fx = fov2focal(view.FoVx, W)
            fy = fov2focal(view.FoVy, H)
            cx = W / 2 - 0.5
            cy = H / 2 - 0.5
            K = torch.tensor([[fx, 0, cx],
                            [0, fy, cy],
                            [0, 0, 1]]).float().cuda()
            xyz_map, nonzero_depth_mask = depth_to_xyz_map(depth_map, H, W, K)  # [h,w,3]
            point_cloud_c = xyz_map.view(-1, 3)  # [h*w, 3]
            R_cam = torch.tensor(view.R.T, dtype=torch.float32, device=points.device)
            T_cam = torch.tensor(view.T, dtype=torch.float32, device=points.device)
            point_cloud_w = (R_cam @ point_cloud_c.T).T + T_cam  # (N, 3)
            all_points.append(point_cloud_w)
            
        final_point_cloud = torch.cat(all_points, dim=0)  # (N_total, 3)
        print(final_point_cloud.shape)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(final_point_cloud.cpu().numpy())
        pcd_down = pcd.voxel_down_sample(voxel_size=0.001)

        downsampled_points = torch.tensor(np.asarray(pcd_down.points), device=final_point_cloud.device, dtype=torch.float32)

        print("Original point count:", final_point_cloud.shape[0])
        print("Point count after voxel deduplication:", downsampled_points.shape[0])
        o3d.io.write_point_cloud(os.path.join(scene.model_path, "depth_fused_pointcloud.ply"), pcd_down)
            # np.savetxt(os.path.join(scene.model_path, "train", "ours_30000", "pc_c_d", f"point_cloud_c_depth_{idx}.txt"), point_cloud_c.cpu().numpy())
            
        #     u,v = xyz_to_uv(camera_coords[:, :3], K)
            
        #     mask_valid = (u >= 0) & (u < depth_map.shape[1]) & (v >= 0) & (v < depth_map.shape[0])  # (N,)

            
        #     depth_values =  depth_map[v.int()[mask_valid], u.int()[mask_valid]]
        #     depths_valid = depths[mask_valid].squeeze()
        #     visible_mask = (depths_valid <=  depth_values) &  (depths_valid >= 0)  # (N,)
        #     # print(torch.sum(visible_mask))
            
        #     visible_mask_all = torch.zeros(points.size()[0], dtype=torch.bool, device=points.device)  # [N]

        #     visible_mask_all[mask_valid] = visible_mask  # Update visibility of valid points into full_mask

        #     # print(f"Full visible mask size: {visible_mask_all.sum()}")
            
        #     mask_path = os.path.join(fg_mask_dir, view.image_name + ".JPG")
        #     mask_image = torch.tensor(imageio.imread(mask_path), dtype=torch.float32, device=points.device)  # [H, W]
            
        #     mask_values = mask_image[v.int()[visible_mask_all], u.int()[visible_mask_all]]
        #     fg_mask = mask_values > 128
            
        #     # print(fg_mask.sum())
            
        #     all_mask = torch.zeros(points.size()[0], dtype=torch.bool, device=points.device)
        #     all_mask[visible_mask_all] = fg_mask
        #     all_mask_ = torch.zeros(points.size()[0], dtype=torch.bool, device=points.device)
        #     all_mask_[visible_mask_all] = ~fg_mask
        #     # print(all_mask.size())
        #     points_fg_num[all_mask, 0] += 1
        #     # print(points_fg_num[all_mask, 0])
        #     points_fg_num[all_mask_, 1] +=1
            
            
        # # # print(points_fg_num[:10, ...])  
        # opacities = gaussians.get_opacity.squeeze()
        # # selected_points = (points_fg_num[:, 0] > points_fg_num[:, 1])
        # # print(selected_points.size())
        # # print(selected_points.sum())
        # # print(selected_points[:10])
        # burnability = torch.from_numpy(np.load(os.path.join(scene.model_path, "points_burnability_knn.npy"))).cuda()
        # materials = torch.from_numpy(np.load(os.path.join(scene.model_path, "points_materials_knn.npy"))).cuda()
        # # selected_points = selected_points & (burnability == 0)
        
        # selected_points = (burnability == 0)
        # print(selected_points.sum())
        
        # points = gaussians.get_xyz
        # print(points.shape[0])
        # points[..., -1] = -points[..., -1]
        # min_coords, max_coords = get_bounding_box(points)
        # #garden
        # max_coords = torch.tensor([1.3, 1.5, 0.0], device=points.device)
        # min_coords = torch.tensor([-2.3, -2.1, -3.5], device=points.device)
        # # #garden_8 PGSR
        # max_coords = torch.tensor([1.5, 2.45, -0.15], device=points.device)
        # min_coords = torch.tensor([-2.1, -1.25, -3.38], device=points.device)
        # # #firewood_sand PGSR
        # # max_coords = torch.tensor([0.75, 1.25, -1.43], device=points.device)
        # # min_coords = torch.tensor([-1.2, -1.05, -2.77], device=points.device)
        # # print(f"Original Bounding Box: min={min_coords}, max={max_coords}")
        # # # #chair PGSR scale:1.2
        # max_coords = torch.tensor([0.82, 0.82, -1.6], device=points.device)
        # min_coords = torch.tensor([-1.25, -1.65, -3.2], device=points.device)
        # print(f"Original Bounding Box: min={min_coords}, max={max_coords}")
        
        # #  firewoods_sand PGSR
        # # max_coords = torch.tensor([1.2, 0.8, -1.18], device=points.device)
        # # min_coords = torch.tensor([-0.86, -1.65, -2.6], device=points.device)
        # # print(f"Original Bounding Box: min={min_coords}, max={max_coords}")
        
        # # firewoods_sand_dark
        # max_coords = torch.tensor([0.9, 0.9, -1.0], device=points.device)
        # min_coords = torch.tensor([-1.5, -1.5, -2.8], device=points.device)
        # print(f"Original Bounding Box: min={min_coords}, max={max_coords}")
        
        # # chair_indoor  scale:1.1
        # # max_coords = torch.tensor([1.4, 1.55, -1.05], device=points.device)
        # # min_coords = torch.tensor([-0.28, -0.4, -3.35], device=points.device)
        # # print(f"Original Bounding Box: min={min_coords}, max={max_coords}")
        
        # # kitchen
        # # max_coords = torch.tensor([2, -2.3, -2.6], device=points.device)
        # # min_coords = torch.tensor([-0.5, -6, -5.5], device=points.device)
        # # print(f"Original Bounding Box: min={min_coords}, max={max_coords}")
        
        # # playground scale=1.1
        # # max_coords = torch.tensor([0.66, 0.65, 0.3], device=points.device)
        # # min_coords = torch.tensor([-0.36, -0.35, -0.5], device=points.device)
        # # print(f"Original Bounding Box: min={min_coords}, max={max_coords}")
        
        # # kitchen_new
        # # max_coords = torch.tensor([3, -1.2, -2.4], device=points.device)
        # # min_coords = torch.tensor([-2, -5.4, -4.9], device=points.device)
        # # print(f"Original Bounding Box: min={min_coords}, max={max_coords}")
        
        # # kitchen_lego scale: 1.1
        # # max_coords = torch.tensor([2.15, -1.6, -2.0], device=points.device)
        # # min_coords = torch.tensor([-2.5, -5.0, -4.1], device=points.device)
        # # print(f"Original Bounding Box: min={min_coords}, max={max_coords}")

        # # Expand bounding box
        # new_min_coords, new_max_coords = expand_bounding_box(min_coords, max_coords, scale_factor=args.scale_factor)
        # print(f"Expanded Bounding Box: min={new_min_coords}, max={new_max_coords}")
        
        # in_voxel_mask = (
        #     (points[:, 0] >= new_min_coords[0]) & (points[:, 0] <= new_max_coords[0]) &
        #     (points[:, 1] >= new_min_coords[1]) & (points[:, 1] <= new_max_coords[1]) &
        #     (points[:, 2] >= new_min_coords[2]) & (points[:, 2] <= new_max_coords[2])
        # )
        # selected_points = selected_points & in_voxel_mask
        
        # materials = materials[selected_points]
        # print(materials.shape)
        
        # gaussians.crop_setup(selected_points)
        # save_path = os.path.join(scene.model_path, "fg", "fg_opac_0_burnable.ply")
        # gaussians.save_ply(save_path)
        
        # points = gaussians.get_xyz
        # print(points.shape[0])
        
        # voxel_grid, grid_size = create_voxel_grid(new_min_coords, new_max_coords, voxel_size=args.voxel_size)
        # print(f"Voxel Grid size: {voxel_grid.shape}, Grid size per voxel: {grid_size}")
        # config = {
        #     "bounding_box": {
        #         "min": new_min_coords.tolist(),   # ==> list[float]
        #         "max": new_max_coords.tolist(),
        #     },
        #     "voxel_grid": {
        #         "dims": list(voxel_grid.shape),    # ==> list[int]
        #         "voxel_size": grid_size[0].item()  # If isotropic, store a single number; if anisotropic, store a list
        #     },
        # }
        # import yaml
        # from pathlib import Path
        # cfg_path = Path(os.path.join(scene.model_path, "simulation_voxel.yaml"))
        # cfg_path.write_text(yaml.safe_dump(config, sort_keys=False))
        # print(f"Config saved to {cfg_path.resolve()}")
        
        # voxel_grid, material_grid, indices_pts_in_grids = map_points_to_voxels_materials(points, new_min_coords, gaussians.get_opacity, grid_size, voxel_grid, materials)
        
        # print_voxel_statistics(voxel_grid)
        
        # visualize_voxel_grid(voxel_grid=voxel_grid)
        
        # point_cloud = voxel_grid_to_point_cloud(voxel_grid=voxel_grid, voxel_size=grid_size, min_coords=new_min_coords)
        # voxel_path = os.path.join(scene.model_path, "voxel.ply")
        # o3d.io.write_point_cloud(voxel_path, point_cloud)
        # point_cloud = voxel_grid_to_point_cloud_color(voxel_grid=material_grid, voxel_size=grid_size, min_coords=new_min_coords)
        # material_path = os.path.join(scene.model_path, "material_voxel.ply")
        # o3d.io.write_point_cloud(material_path, point_cloud)
        # voxel_grid_path = os.path.join(scene.model_path, "voxel_grid.npy")
        # np.save(voxel_grid_path,voxel_grid)
        # material_grid_path = os.path.join(scene.model_path, "material_grid.npy")
        # np.save(material_grid_path,material_grid)
        # voxel = np.load(voxel_grid_path)
        # print(voxel.shape)
        # voxel = np.load(material_grid_path)
        # print(np.unique(voxel))
        # # assert voxel.shape == (128, 128, 128)
        
        # print("save indices_pts_in_grids")  # Grid coordinates of each Gaussian point in the grid
        # indices_pts_in_grids_path = os.path.join(scene.model_path, "indices_pts_in_grids.pth")
        # torch.save(indices_pts_in_grids, indices_pts_in_grids_path)

        # # print("save indices_pts_ingrids_for_inall")   # Index of each Gaussian point in the grid among all Gaussians
        # # indices_pts_ingrids_for_inall_path = os.path.join(scene.model_path, "indices_pts_ingrids_for_inall.pth")
        # # indices_pts_ingrids_for_inall = torch.nonzero(selected_points, as_tuple=False).squeeze()
        # # torch.save(indices_pts_ingrids_for_inall, indices_pts_ingrids_for_inall_path)

        # print("save selected_points")  # Mask of each Gaussian point in the grid among all Gaussians
        # mask_pts_in_grids_path = os.path.join(scene.model_path, "mask_pts_in_grids.pth")
        # torch.save(selected_points, mask_pts_in_grids_path)         

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
    parser.add_argument("-v", "--voxel_size", type=int, default=256)
    parser.add_argument("--scale_factor", type=float, default=1.1)
    parser.add_argument("--config_file", type=str, help="Path to the YAML configuration file")
    
    args = get_combined_args(parser)

    if args.config_file:
        yaml_config = load_yaml_config(args.config_file)
        update_args_with_yaml(args, yaml_config)

    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    print(f"multi_view_num {model.multi_view_num}")
    # render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.max_depth, args.voxel_size, args.num_cluster, args.use_depth_filter)
    extract_fg(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
    # extract_fg_filling(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
    # exp_depth(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
    # extract_fg_tsdf(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
    # extract_fg_depth(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)