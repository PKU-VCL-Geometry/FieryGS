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
"""Plane Gaussian rasterization with fire/smoke volume compositing (``render_fire``).

Uses ``diff_plane_rasterization`` for RGB + depth, then ray-marches a fuel grid via ``vis_npz``.
"""

import torch
import torch.nn.functional as F
import math
from diff_plane_rasterization import GaussianRasterizationSettings as PlaneGaussianRasterizationSettings
from diff_plane_rasterization import GaussianRasterizer as PlaneGaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from scene.app_model import AppModel
from utils.graphics_utils import normal_from_depth_image
from rendering.volume_utils import (
    render_coarse_image_XYZ,
    batched_importance_sample_nerf_vmap,
    batched_importance_sample_nerf_vmap_chunked,
    render_image_fire_smoke,
)
import numpy as np
import os
from utils.my_utils import depth_to_xyz_map, transform_c2w, safe_normalize
import time


def render_normal(viewpoint_cam, depth, offset=None, normal=None, scale=1):
    """Surface normals from a depth map using NERF-style intrinsics/extrinsics. Returns (3, H', W')."""
    intrinsic_matrix, extrinsic_matrix = viewpoint_cam.get_calib_matrix_nerf(scale=scale)
    st = max(int(scale/2)-1,0)
    if offset is not None:
        offset = offset[st::scale,st::scale]
    normal_ref = normal_from_depth_image(depth[st::scale,st::scale], 
                                            intrinsic_matrix.to(depth.device), 
                                            extrinsic_matrix.to(depth.device), offset)

    normal_ref = normal_ref.permute(2,0,1)
    return normal_ref


def phong_illumination(light_coords, XYZ_at_valid_indices, point_cloud_w, 
                                 rendered_global_normal_flat, wo_all, kd=0.8, ks=0.5, shininess=32,
                                 attenuation_type='inverse_square', attenuation_constant=1.0, min_distance=0.1):
    """
    Vectorized version of Phong illumination with distance attenuation.
    
    Args:
        light_coords: [M, 3] Light source coordinates
        XYZ_at_valid_indices: [M, 3] Light intensity in XYZ color space
        point_cloud_w: [HW, 3] Point cloud coordinates in world space
        rendered_global_normal_flat: [HW, 3] Surface normals
        wo_all: [HW, 3] View direction
        kd: Diffuse coefficient
        ks: Specular coefficient
        shininess: Specular shininess exponent
        attenuation_type: Type of distance attenuation ('inverse_square', 'linear', 'quadratic', 'physical')
        attenuation_constant: Constant for attenuation calculation
        min_distance: Minimum distance to avoid division by zero
    """
    M = light_coords.shape[0]
    normals = rendered_global_normal_flat
    light_coords = light_coords.unsqueeze(1)
    point_cloud_w = point_cloud_w.unsqueeze(0)
    normals = normals.unsqueeze(0)

    light_vec = light_coords - point_cloud_w
    distances = light_vec.norm(dim=2, keepdim=True)
    distances = torch.clamp(distances, min=min_distance)
    light_dir = light_vec / (distances + 1e-6)
    view_dir = wo_all

    if attenuation_type == 'inverse_square':
        attenuation = 1.0 / (distances.pow(2) + attenuation_constant)
    elif attenuation_type == 'linear':
        attenuation = 1.0 / (1.0 + attenuation_constant * distances)
    elif attenuation_type == 'quadratic':
        attenuation = 1.0 / (1.0 + attenuation_constant * distances.pow(2))
    elif attenuation_type == 'physical':
        attenuation = 1.0 / (1.0 + attenuation_constant * distances + attenuation_constant * distances.pow(2))
    else:
        attenuation = torch.ones_like(distances)

    diffuse = kd * torch.clamp((light_dir * normals).sum(dim=2, keepdim=True), 0, 1)
    reflect_dir = 2 * (light_dir * normals).sum(dim=2, keepdim=True) * normals - light_dir
    specular = ks * torch.clamp((view_dir * reflect_dir).sum(dim=2, keepdim=True), 0, 1).pow(shininess)

    light_intensity = XYZ_at_valid_indices.view(M, 1, 3)
    per_light_XYZ = (diffuse + specular) * light_intensity * attenuation
    XYZ = per_light_XYZ.sum(dim=0)
    
    return XYZ


def render_fire(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, 
           app_model: AppModel=None, return_plane = True, return_depth_normal = True, query_tensor = None, temp2rgb_converter = None, args = None, scale_ratio = 0.12, add_sepc_light=False, fuel2temp=None, color_field=None):
    """Composite plane-GS RGB with a volumetric fire pass. ``bg_color`` and tensors must be on CUDA."""
    start_GS = time.perf_counter()
    # Screen-space 2D means (retain grad for rasterizer backward)
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points_abs = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
        screenspace_points_abs.retain_grad()
    except:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    means3D = pc.get_xyz
    means2D = screenspace_points
    means2D_abs = screenspace_points_abs
    opacity = pc.get_opacity

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    shs = None
    colors_precomp = None

    carbonized_mask = pc.carbonized_mask
    if carbonized_mask is not None:
        shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
        dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        colors_precomp[carbonized_mask, :] = torch.clamp(colors_precomp[carbonized_mask, :] - 0.5 * (pc.carbonized_threshold - pc.occ_carbon[carbonized_mask, :]) / pc.carbonized_threshold, min=0.0)
    else:
        shs = pc.get_features
    
    return_dict = None
    raster_settings = PlaneGaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            render_geo=return_plane,
            debug=pipe.debug
        )

    rasterizer = PlaneGaussianRasterizer(raster_settings=raster_settings)

    if not return_plane:
        rendered_image, radii, out_observe, _, _ = rasterizer(
            means3D = means3D,
            means2D = means2D,
            means2D_abs = means2D_abs,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        
        return_dict =  {"render": rendered_image,
                        "viewspace_points": screenspace_points,
                        "viewspace_points_abs": screenspace_points_abs,
                        "visibility_filter" : radii > 0,
                        "radii": radii,
                        "out_observe": out_observe}
        if app_model is not None and pc.use_app:
            appear_ab = app_model.appear_ab[torch.tensor(viewpoint_camera.uid).cuda()]
            app_image = torch.exp(appear_ab[0]) * rendered_image + appear_ab[1]
            return_dict.update({"app_image": app_image})
        return return_dict

    global_normal = pc.get_normal(viewpoint_camera)
    local_normal = global_normal @ viewpoint_camera.world_view_transform[:3,:3]
    pts_in_cam = means3D @ viewpoint_camera.world_view_transform[:3,:3] + viewpoint_camera.world_view_transform[3,:3]
    depth_z = pts_in_cam[:, 2]
    local_distance = (local_normal * pts_in_cam).sum(-1).abs()
    input_all_map = torch.zeros((means3D.shape[0], 5)).cuda().float()
    input_all_map[:, :3] = local_normal
    input_all_map[:, 3] = 1.0
    input_all_map[:, 4] = local_distance

    rendered_image, radii, out_observe, out_all_map, plane_depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        means2D_abs = means2D_abs,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        all_map = input_all_map,
        cov3D_precomp = cov3D_precomp)

    end_GS = time.perf_counter()
    start_fire = time.perf_counter()

    # Fire rays: invert w2c and apply OpenGL-style Y/Z flip
    w2c = viewpoint_camera.world_view_transform.transpose(1, 0)
    c2w = torch.inverse(w2c)
    flip = torch.diag(torch.tensor([1., -1., -1., 1.], device=c2w.device, dtype=c2w.dtype))
    c2w = c2w @ flip

    rays_o, rays_d = viewpoint_camera.get_rays_fire(c2w)
    rays_o = rays_o.cuda()
    rays_d = rays_d.cuda()
    
    args.dual_smoke = getattr(args, 'dual_smoke', False)
    args.near = getattr(args, 'near', 3)
    args.far = getattr(args, 'far', 7)
    args.num_samples_coarse = getattr(args, 'num_samples_coarse', 128)
    args.num_samples_fine = getattr(args, 'num_samples_fine', 1024)
    args.spec_intensity = getattr(args, 'spec_intensity', 1e-8)
    near = args.near
    far = args.far
    num_samples_coarse = args.num_samples_coarse
    num_samples_fine = args.num_samples_fine

    H, W, _ = rays_d.shape
    fire_image = torch.zeros((3, H, W), device=rays_d.device)
    transmittance = torch.zeros((H, W, 1), device=rays_d.device)
    W_batch_size = 64

    args.strength = getattr(args, 'strength', 0.01)
    args.smoke_strength = getattr(args, 'smoke_strength', 0.5)

    # Ray-march fire in W strips to limit VRAM
    for start in range(0, W, W_batch_size):
        end = min(start + W_batch_size, W)
        rays_d_batch = rays_d[:, start:end, :]
        rays_o_batch = rays_o[:, start:end, :]

        t_vals_coarse = torch.linspace(near, far, num_samples_coarse, device=rays_o.device)
        t_vals_coarse_jittered = t_vals_coarse.view(1, 1, num_samples_coarse)
        fuel_sample = render_coarse_image_XYZ(rays_o_batch, rays_d_batch, num_samples_coarse, t_vals_coarse_jittered, query_tensor) + 1e-6

        smooth_factor = getattr(args, 'sampling_smooth_factor', 0.3)
        if num_samples_fine > 1024:
            chunk_size = getattr(args, 'sampling_chunk_size', 32)
            t_vals_fine = batched_importance_sample_nerf_vmap_chunked(
                t_vals_coarse_jittered, fuel_sample,
                num_samples=num_samples_fine, smooth_factor=smooth_factor, chunk_size=chunk_size)
        else:
            t_vals_fine = batched_importance_sample_nerf_vmap(
                t_vals_coarse_jittered, fuel_sample,
                num_samples=num_samples_fine, smooth_factor=smooth_factor)
        
        y_expanded = t_vals_coarse_jittered.expand(t_vals_fine.shape[0], t_vals_fine.shape[1], num_samples_coarse)
        combined = torch.cat([t_vals_fine, y_expanded], dim=2)
        t_vals_combine, _ = torch.sort(combined, dim=2)

        averaged_color, transmittance_batch = render_image_fire_smoke(
            rays_o_batch, rays_d_batch, num_samples_fine, t_vals_combine, query_tensor,
            temp2rgb_converter, batch_size=128, strength=args.strength, sigma_a=args.sigma_a,
            is_norm_samples=True, args=args, depth_map=plane_depth[..., start:end], debug_mask=False,
            scale_ratio=scale_ratio, smoke_strengh=args.smoke_strength, exposure=0.8, threshold=0.6,
            fuel2temp=fuel2temp, render_smoke=args.render_smoke, color_field=color_field,
        )

        fire_image[:, :, start:end] = averaged_color.permute(2, 0, 1)
        transmittance[:, start:end, :] = transmittance_batch

    # Front-to-back: fire over Gaussians scaled by remaining transmittance
    rendered_image = fire_image + rendered_image * transmittance.permute(2, 0, 1) * args.darkness

    args.ks = getattr(args, 'ks', 0.5)

    if add_sepc_light:
        # Optional specular from hot voxels (downsampled temperature grid)
        temp_query_tensor = fuel2temp(query_tensor)
        light_grids = F.interpolate(
            temp_query_tensor,
            scale_factor=0.25,
            mode='trilinear',
            align_corners=False
        )
        light_threshold = 0.0
        light_values = light_grids.squeeze()
        valid_indices = torch.nonzero(light_values > light_threshold, as_tuple=True)

        temp_values = light_grids.squeeze()
        temperature_at_valid_indices = temp_values[valid_indices]
        XYZ_at_valid_indices = temp2rgb_converter(temperature_at_valid_indices * scale_ratio)

        if len(valid_indices[0]) > 0:
            input_all_map_for_global_normal = torch.zeros((means3D.shape[0], 5)).cuda().float()
            input_all_map_for_global_normal[:, :3] = global_normal
            input_all_map_for_global_normal[:, 3] = 1.0
            input_all_map_for_global_normal[:, 4] = local_distance

            _, _, _, out_all_map_for_global_normal, _ = rasterizer(
                means3D = means3D,
                means2D = means2D,
                means2D_abs = means2D_abs,
                shs = shs,
                colors_precomp = colors_precomp,
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                all_map = input_all_map_for_global_normal,
                cov3D_precomp = cov3D_precomp)

            rendered_global_normal = out_all_map_for_global_normal[0:3]
            rendered_global_normal_flat = rendered_global_normal.permute(1, 2, 0).view(-1, 3)

            # Pixel centers in world space for view-dependent specular
            xyz_map, _ = depth_to_xyz_map(plane_depth[0], viewpoint_camera.image_height, viewpoint_camera.image_width, viewpoint_camera.K)
            point_cloud_c = xyz_map.view(-1, 3)

            w2c_mat = viewpoint_camera.world_view_transform.transpose(1, 0)
            point_cloud_w = transform_c2w(point_cloud_c, w2c_mat)
            view_pos_all = viewpoint_camera.camera_center.repeat(point_cloud_w.shape[0], 1)
            wo_all = safe_normalize(view_pos_all - point_cloud_w)

            grid_coords = torch.stack(valid_indices, dim=1)
            grid_coords = grid_coords[:, [2, 1, 0]]
            min_bound = torch.tensor(args.bounding_box['min'], dtype=torch.float32, device=light_grids.device)
            max_bound = torch.tensor(args.bounding_box['max'], dtype=torch.float32, device=light_grids.device)
            voxel_size = (max_bound - min_bound) / light_grids.shape[2]
            light_coords = min_bound + (grid_coords.float() + 0.5) * voxel_size
            # Specular: batched Phong over voxel lights (phong_illumination), not full N×M light–surface pairing.

            HW = point_cloud_w.shape[0]
            spec_intensity = float(args.spec_intensity)
            spec_batch_size = 1024
            device = point_cloud_w.device
            
            spec_XYZ = torch.zeros(HW, 3, device=device)
            
            for i in range(0, HW, spec_batch_size):
                start = i
                end = min(i + spec_batch_size, HW)
                
                batch_points = point_cloud_w[start:end]
                batch_normals = rendered_global_normal_flat[start:end]
                batch_wo = wo_all[start:end]
                
                attenuation_type = getattr(args, 'attenuation_type', 'inverse_square')
                attenuation_constant = getattr(args, 'attenuation_constant', 1.0)
                min_distance = getattr(args, 'min_distance', 0.1)
                
                batch_XYZ = phong_illumination(
                    light_coords,
                    XYZ_at_valid_indices,
                    batch_points,
                    batch_normals,
                    batch_wo,
                    kd=0.8,
                    ks=args.ks,
                    shininess=32,
                    attenuation_type=attenuation_type,
                    attenuation_constant=attenuation_constant,
                    min_distance=min_distance,
                )

                spec_XYZ[start:end] = batch_XYZ

            spec_color = temp2rgb_converter.XYZ2RGB(spec_XYZ * spec_intensity)
            phong_map = spec_color.view(viewpoint_camera.image_height, viewpoint_camera.image_width, 3)
            phong_map = phong_map.permute(2, 0, 1)
            rendered_image += phong_map * transmittance.permute(2, 0, 1)

    end_fire = time.perf_counter()

    rendered_normal = out_all_map[0:3]
    rendered_alpha = out_all_map[3:4, ]
    rendered_distance = out_all_map[4:5, ]
    
    return_dict =  {"render": rendered_image,
                    "viewspace_points": screenspace_points,
                    "viewspace_points_abs": screenspace_points_abs,
                    "visibility_filter" : radii > 0,
                    "radii": radii,
                    "out_observe": out_observe,
                    "rendered_normal": rendered_normal,
                    "plane_depth": plane_depth,
                    "rendered_distance": rendered_distance, 
                    "GS_render_timing": end_GS - start_GS,
                    "fire_render_timing": end_fire - start_fire,
                    }
    
    if app_model is not None and pc.use_app:
        appear_ab = app_model.appear_ab[torch.tensor(viewpoint_camera.uid).cuda()]
        app_image = torch.exp(appear_ab[0]) * rendered_image + appear_ab[1]
        return_dict.update({"app_image": app_image})

    if return_depth_normal:
        depth_normal = render_normal(viewpoint_camera, plane_depth.squeeze()) * (rendered_alpha).detach()
        return_dict.update({"depth_normal": depth_normal})

    return return_dict