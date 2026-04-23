import numpy as np
import plyfile
import torch
# from utils.graphics_utils import depth_pcd2normal
import torch.nn.functional as F
import math
import torch.nn as nn
import yaml


########################################
#
# save
#
########################################
def save_tensor_point_cloud_as_ply(tensor, filename):
    # Ensure the input tensor has shape [N, 3]
    assert tensor.ndim == 2 and tensor.shape[1] == 3, "Input tensor must have shape [N, 3]"

    # Convert the PyTorch tensor to a numpy array
    points = tensor.cpu().numpy()

    # Define the elements of the PLY file
    vertex = np.array(
        [(points[i, 0], points[i, 1], points[i, 2]) for i in range(points.shape[0])],
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    )

    # Create a list of elements for the PLY file
    element = plyfile.PlyElement.describe(vertex, 'vertex')

    # Save as a PLY file
    plyfile.PlyData([element], text=True).write(filename)
    print(f"Point cloud saved as {filename}")


########################################
#
# Others
#
########################################
def print_cuda(name):
    mem = torch.cuda.memory_allocated() / (1024**2)  # Convert to MB
    print(f"[CUDA] {name} GPU memory usage: {mem:.2f} MB")

class FourierScaleGate(nn.Module):
    def __init__(self, num_frequencies=10, hidden_dim=64, out_dim=32):
        super().__init__()
        self.num_frequencies = num_frequencies

        # Create frequency basis: [L, 1]
        freq_bands = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        self.register_buffer("freq_bands", freq_bands.unsqueeze(1))  # [L, 1]

        self.net = nn.Sequential(
            nn.Linear(num_frequencies * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.Sigmoid()  # Output constrained to [0, 1]
        )

    def fourier_encode(self, x):
        """
        x: [N, 1] scalar scale values
        Returns: [N, 2L] encoded features
        """
        x_proj = 2 * math.pi * x @ self.freq_bands.T  # [N, L]
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)  # [N, 2L]

    def forward(self, scale):
        if scale.dim() == 1:
            scale = scale.unsqueeze(-1)  # [N] -> [N,1]
        x_encoded = self.fourier_encode(scale)  # [N, 2L]
        return self.net(x_encoded)  # [N, out_dim]


class ContextAwareScaleGate(nn.Module):
    def __init__(self, embed_dim=32, num_heads=4, out_dim=32):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Linear(1, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.out_proj = nn.Linear(embed_dim, out_dim)
        self.activation = nn.Sigmoid()  # Keep original gate activation

    def forward(self, scales):  # scales: [N_s, 1]
        x = self.embedding(scales)        # [N_s, embed_dim]
        x_attn, _ = self.attn(x, x, x)    # [N_s, embed_dim]
        gate = self.out_proj(x_attn)      # [N_s, out_dim]
        return self.activation(gate)      # [N_s, out_dim]

def _logit(y):
    return torch.log(y / (1 - y))

def _log(y):
    return torch.log(y)

def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)

def length(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN

def safe_normalize(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return x / length(x, eps)

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


def transform_c2w(point_cloud, W2C):
    # 计算C2W矩阵
    C2W = torch.inverse(W2C)
    # 转换点云为齐次坐标
    ones = torch.ones(point_cloud.shape[0], 1, dtype=point_cloud.dtype, device=point_cloud.device)
    point_cloud_homogeneous = torch.cat([point_cloud, ones], dim=1)  # [N, 4]
    # 应用C2W变换
    point_cloud_world = (C2W @ point_cloud_homogeneous.T).T  # [N, 4]
    # 去掉齐次分量
    point_cloud_world = point_cloud_world[:, :3]  # [N, 3]

    return point_cloud_world


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


def interpolate_intensity(t: torch.Tensor, pos: torch.Tensor, pos_color: torch.Tensor):
    t = torch.tensor(t, dtype=torch.float32).unsqueeze(-1)  # Expand to (1, N)

    mask_0 = t < pos[0]
    mask_1 = (t >= pos[0]) & (t < pos[1])
    mask_2 = (t >= pos[1]) & (t < pos[2])

    t_norm1 = (t - pos[0]) / (pos[1] - pos[0])
    t_norm2 = (t - pos[1]) / (pos[2] - pos[1])

    # t_eased1 = 1 - (1 - t_norm1) ** 2
    # t_eased2 = 1 - (1 - t_norm2) ** 2
    t_eased1 = t_norm1**2
    t_eased2 = 1 - (1 - t_norm2) ** 2

    res_color = torch.where(mask_0, pos_color[0], 
                torch.where(mask_1, pos_color[0] + (pos_color[1] - pos_color[0]) * t_eased1,
                torch.where(mask_2, pos_color[1] + (pos_color[2] - pos_color[1]) * t_eased2, 
                pos_color[2])))
    # print(res_color.shape)

    return res_color


def interpolate_color(t: torch.Tensor, pos: torch.Tensor, pos_color: torch.Tensor):
    t = t.to(dtype=torch.float32).unsqueeze(-1) 
    t_norm = (t - pos[0]) / (pos[1] - pos[0])
    t_norm = torch.clamp(t_norm, 0.0, 1.0)  

    res_color = pos_color[0] + (pos_color[1] - pos_color[0]) * t_norm
    # print(res_color.shape)
    return res_color

def temp2color(t: torch.Tensor):
    # t = t.unsqueeze(-1) # [128, 128, 128, 1]
    pos_intensity_color = torch.tensor([
                                    [0.0, 0.0, 0.0],  # Black
                                    [1.0, 1.0, 1.0],  # White
                                    [0.019, 0.019, 0.019]  # Dark gray
                                ]).cuda()
    pos_intensity = torch.tensor([0.895, 0.959, 1.0]).cuda()
    # pos_intensity = torch.tensor([0.65, 0.895, 1.0]).cuda()
    pos_fire_color = torch.tensor([
                                    [0.859, 0.091, 0.004],  # Start color
                                    [1.000, 0.479, 0.063]   # End color
                                ]).cuda()
    pos_fire = torch.tensor([0.000, 1.000]).cuda()
    
    t_ = t * 0.5
    intensity = interpolate_intensity(t_, pos_intensity, pos_intensity_color)
    color = interpolate_color(t_, pos_fire, pos_fire_color)
    
    
    color = torch.clamp(color * intensity, 0.0, 1.0)
    return color, intensity



def generate_grid(min_coords, max_coords, resolution=128):
    """
    Generate a 128x128x128 uniform 3D grid and return (x, y, z) coordinates of all grid centers.

    Args:
        min_coords (torch.Tensor): Tensor of shape (3,) representing the minimum coordinates (x_min, y_min, z_min)
        max_coords (torch.Tensor): Tensor of shape (3,) representing the maximum coordinates (x_max, y_max, z_max)
        resolution (int): Number of grid cells per dimension, default is 128

    Returns:
        grid_coords (torch.Tensor): Tensor of shape (128, 128, 128, 3) representing center coordinates of each grid point
    """
    device = min_coords.device  # Ensure computation runs on GPU

    # Compute step size (size of each grid cell)
    step = (max_coords - min_coords) / resolution  # Compute step size

    # Compute indices (0, 1, ..., 127)
    indices = torch.arange(resolution, device=device, dtype=torch.float32)  # Shape (128,)

    # Compute center coordinates (min + (i + 0.5) * step)
    x = min_coords[0] + (indices + 0.5) * step[0]
    y = min_coords[1] + (indices + 0.5) * step[1]
    z = min_coords[2] + (indices + 0.5) * step[2]

    # Generate 3D grid coordinates
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')  # Shape (128, 128, 128)

    # Combine into a coordinate tensor of shape (128, 128, 128, 3)
    grid_coords = torch.stack((X, Y, Z), dim=-1)  # Shape (128, 128, 128, 3)

    return grid_coords, step

def voxel2grid(t: torch.tensor):
    t = t.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
    D, H, W = t.shape[2], t.shape[3], t.shape[4]
    
    # Use F.interpolate for trilinear interpolation, upsampling to (D+1, H+1, W+1)
    T_vertex = F.interpolate(t.double().cuda(),
                             size=(D+1, H+1, W+1),
                             mode='trilinear',
                             align_corners=True)
    return T_vertex.squeeze() 

def sample_temperature(points: torch.tensor, grid: torch.Tensor):
    # N = points.shape[0]
    # points = points.unsqueeze(0).unsqueeze(0)
    # points = (points  / (T_vertex.shape[0] - 1)) * 2 - 1
    # interpolated_values = F.grid_sample(
    #     T_vertex.unsqueeze(0).unsqueeze(0).cuda(),  # Expand batch and channel dimensions
    #     points.view(1, 1, N, 1, 3).double().cuda(), 
    #     mode='bilinear',  # 3D version uses trilinear
    #     align_corners=True
    # )
    # return interpolated_values.squeeze()
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    # Compute floor indices (x0, y0, z0) and ceil indices (x1, y1, z1)
    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    z0 = torch.floor(z).long()
    
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1
    
    # Get grid dimensions (D, H, W)
    D, H, W = grid.shape
    
    # Boundary handling: ensure indices stay within bounds
    x0 = x0.clamp(0, D - 1)
    y0 = y0.clamp(0, H - 1)
    z0 = z0.clamp(0, W - 1)
    x1 = x1.clamp(0, D - 1)
    y1 = y1.clamp(0, H - 1)
    z1 = z1.clamp(0, W - 1)
    
    # Compute fractional part (offset) for each point in x, y, z directions
    xd = (x - x0.float())
    yd = (y - y0.float())
    zd = (z - z0.float())
    
    # Retrieve values from 8 neighboring vertices
    # Using advanced indexing; all operations are vectorized
    v000 = grid[x0, y0, z0]
    v001 = grid[x0, y0, z1]
    v010 = grid[x0, y1, z0]
    v011 = grid[x0, y1, z1]
    v100 = grid[x1, y0, z0]
    v101 = grid[x1, y0, z1]
    v110 = grid[x1, y1, z0]
    v111 = grid[x1, y1, z1]
    
    # Interpolate along x direction
    c00 = v000 * (1 - xd) + v100 * xd  # x interpolation at y0, z0
    c01 = v001 * (1 - xd) + v101 * xd  # x interpolation at y0, z1
    c10 = v010 * (1 - xd) + v110 * xd  # x interpolation at y1, z0
    c11 = v011 * (1 - xd) + v111 * xd  # x interpolation at y1, z1

    # Interpolate along y direction
    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd

    # Finally interpolate along z direction
    interpolated = c0 * (1 - zd) + c1 * zd
    
    return interpolated


def simulation_to_world(simulation_file, voxel_yaml_file):
    '''
    input:
        simulation_file: simulation temperature file
        voxel_file: voxel YAML file
    return:
        numpy array of shape (resolution^3, 4), containing (x, y, z, temperature)
    '''
    sim_res = np.load(simulation_file)['arr_0']
    temp = sim_res[2:-2, 2:-2, 2:-2]
    print(temp.shape)
    with open(voxel_yaml_file, 'r') as file:
        voxel_info = yaml.safe_load(file)
    # grid_coords,  step= generate_grid(torch.tensor(voxel_info['bounding_box']['min']), 
    #                                   torch.tensor(voxel_info['bounding_box']['max']), 
    #                                   voxel_info['voxel_grid']['dims'][0])  # (128*128*128,)
    # points_with_values = np.concatenate([grid_coords.reshape(-1, 3), temp.flatten()[:, None]], axis=-1)  # (N, 4)
    min_bound = torch.tensor(voxel_info['bounding_box']['min'], dtype=torch.float32)
    max_bound = torch.tensor(voxel_info['bounding_box']['max'], dtype=torch.float32)

    grid_coords, step = generate_grid(min_bound, max_bound, voxel_info['voxel_grid']['dims'][0])  # (N, 3)

    # Normalize coordinates to [-1, 1]^3
    grid_centered = (grid_coords - min_bound) / (max_bound - min_bound)  # Normalize to [0, 1]
    grid_normalized = grid_centered * 2.0 - 1.0  # Map to [-1, 1]

    # Concatenate temperature values
    points_with_values = np.concatenate([grid_normalized.reshape(-1, 3), temp.flatten()[:, None]], axis=-1)
    return points_with_values

def occ_grid_2_global_index(points: torch.tensor, occ_grids: torch.tensor, voxel_yaml_file):
    with open(voxel_yaml_file, 'r') as file:
        voxel_info = yaml.safe_load(file)
    
    min_bound = torch.tensor(voxel_info['bounding_box']['min'], dtype=torch.float32)
    max_bound = torch.tensor(voxel_info['bounding_box']['max'], dtype=torch.float32)
    voxel_size = voxel_info['voxel_grid']['voxel_size']
    dims = voxel_info['voxel_grid']['dims']
    assert dims == list(occ_grids.shape)
    
    grid_coords = (points - min_bound) / voxel_size
    grid_indices = torch.floor(grid_coords).long()

    valid_mask = (
        (grid_indices[:, 0] >= 0) & (grid_indices[:, 0] < dims[0]) &
        (grid_indices[:, 1] >= 0) & (grid_indices[:, 1] < dims[1]) &
        (grid_indices[:, 2] >= 0) & (grid_indices[:, 2] < dims[2])
    )
    
    occ_mask = torch.zeros(points.shape[0], dtype=torch.bool)

    valid_indices = grid_indices[valid_mask]

    occ_values = occ_grids[
        valid_indices[:, 0],
        valid_indices[:, 1],
        valid_indices[:, 2]
    ]

    free_mask = (occ_values == 0)

    occ_mask[valid_mask] = free_mask

    return occ_mask

