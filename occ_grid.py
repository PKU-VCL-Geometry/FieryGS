import numpy as np
import plyfile
import torch
# from utils.graphics_utils import depth_pcd2normal
import torch.nn.functional as F
import math
import torch.nn as nn
import yaml
from scene import Scene, GaussianModel

def occ_grid_2_global_index(points: torch.tensor, occ_grids: torch.tensor, voxel_yaml_file):
    with open(voxel_yaml_file, 'r') as file:
        voxel_info = yaml.safe_load(file)
    
    min_bound = torch.tensor(voxel_info['bounding_box']['min'], dtype=torch.float32)
    max_bound = torch.tensor(voxel_info['bounding_box']['max'], dtype=torch.float32)
    voxel_size = voxel_info['voxel_grid']['voxel_size']
    dims = voxel_info['voxel_grid']['dims']
    assert dims == list(occ_grids.shape), occ_grids.shape
    
    grid_coords = (points - min_bound) / voxel_size
    grid_indices = torch.floor(grid_coords).long()

    valid_mask = (
        (grid_indices[:, 0] >= 0) & (grid_indices[:, 0] < dims[0]) &
        (grid_indices[:, 1] >= 0) & (grid_indices[:, 1] < dims[1]) &
        (grid_indices[:, 2] >= 0) & (grid_indices[:, 2] < dims[2])
    )
    
    print(valid_mask.sum())
    
    occ_mask = torch.zeros(points.shape[0], dtype=torch.bool)

    valid_indices = grid_indices[valid_mask]

    occ_values = occ_grids[
        valid_indices[:, 0],
        valid_indices[:, 1],
        valid_indices[:, 2]
    ]
    print(occ_values.unique())

    free_mask = (occ_values < 1e-7)
    print(free_mask.sum())

    occ_mask[valid_mask] = free_mask

    return occ_mask



occ_grid = np.load('utils/occupancy_398.npz')['arr_0']
occ_grid = occ_grid
print(np.unique(occ_grid))

# mask = occ_grid_2_global_index(points, occ_grids, 'output/garden/simulation_voxel.yaml')
# print(mask)
gaussians = GaussianModel(sh_degree=3)
gaussians.load_ply('./output/garden/fg/fg_opac_0_.ply')
print(gaussians.get_xyz.shape)

points  = gaussians.get_xyz.cpu()
occ_grid = torch.from_numpy(occ_grid)

mask = occ_grid_2_global_index(points, occ_grid, 'output/garden/simulation_voxel.yaml')
print(mask.sum())

import open3d as o3d
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points[~mask].detach().numpy())
o3d.io.write_point_cloud("output_normalized.ply", pcd)