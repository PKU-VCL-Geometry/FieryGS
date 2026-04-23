"""Volumetric rendering utilities used by gaussian renderer fire pass."""

import copy

import torch
from torch import vmap
from torch.nn import functional as F


def interp_data(query_points_normalized, query_tensor):
    """Trilinear sample of ``query_tensor`` [B,C,D,H,W] at points in [-1,1]^3."""
    original_shape = query_points_normalized.shape
    grid = query_points_normalized.view(1, -1, 1, 1, 3)
    output = F.grid_sample(
        query_tensor,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    output = output.squeeze().t()
    if query_tensor.size(1) == 1:
        return output.view(*original_shape[:-1])
    return output.view(*original_shape)


def render_coarse_image_XYZ(
    rays_o,
    rays_d,
    num_samples,
    t_vals,
    query_tensor,
    batch_size=1024,
    sigma_a=0.01,
    density=0.1,
    is_norm_samples=False,
    args=None,
    depth_map=None,
    debug_mask=False,
):
    del num_samples, batch_size, sigma_a, density, debug_mask
    if depth_map is not None:
        H, W = rays_o.shape[:2]
        depth = depth_map.squeeze().view(H, W, 1)
        valid_mask = t_vals <= depth
    else:
        valid_mask = torch.ones_like(t_vals, dtype=torch.bool)

    sample_points = rays_o.unsqueeze(2) + rays_d.unsqueeze(2) * t_vals.unsqueeze(-1)
    if is_norm_samples and args is not None:
        min_bound = torch.tensor(args.bounding_box["min"], dtype=torch.float32, device=sample_points.device)
        max_bound = torch.tensor(args.bounding_box["max"], dtype=torch.float32, device=sample_points.device)
        sample_points = ((sample_points - min_bound) / (max_bound - min_bound)) * 2.0 - 1.0
    temperature_batch = interp_data(sample_points, query_tensor)
    if depth_map is not None:
        temperature_batch = temperature_batch * valid_mask.unsqueeze(-1)
    return temperature_batch


def batched_importance_sample_nerf_vmap(t, temp, num_samples=256, smooth_factor=0.3):
    H, W, _ = temp.shape
    if t.dim() == 3 and t.shape[0] == 1 and t.shape[1] == 1:
        t = t.expand(H, W, -1)

    def sample_fn(t_row, temp_row):
        near = t_row[0]
        far = t_row[-1]
        temp_sum = temp_row.sum()
        uniform_samples = torch.linspace(near, far, num_samples, device=t.device)
        importance_pdf = temp_row / (temp_sum + 1e-8)
        importance_pdf = torch.clamp(importance_pdf, min=0.0)
        importance_pdf = importance_pdf / (importance_pdf.sum() + 1e-8)
        uniform_pdf = torch.ones_like(temp_row) / len(temp_row)
        mixed_pdf = (1.0 - smooth_factor) * importance_pdf + smooth_factor * uniform_pdf
        mixed_pdf = torch.clamp(mixed_pdf, min=1e-8)
        mixed_pdf = mixed_pdf / (mixed_pdf.sum() + 1e-8)
        cdf = torch.cumsum(mixed_pdf, dim=0)
        cdf = torch.cat([torch.zeros(1, device=cdf.device), cdf])
        cdf = torch.clamp(cdf, min=0.0, max=1.0)
        u = torch.linspace(0, 1, num_samples, device=t.device)
        idx = torch.searchsorted(cdf, u) - 1
        idx = torch.clamp(idx, 0, len(t_row) - 2)
        t_low = t_row[idx]
        t_high = t_row[idx + 1]
        cdf_low = cdf[idx]
        cdf_high = cdf[idx + 1]
        cdf_diff = cdf_high - cdf_low
        z = torch.where(cdf_diff > 1e-8, (u - cdf_low) / cdf_diff, torch.zeros_like(u))
        importance_samples = t_low + z * (t_high - t_low)
        importance_samples = torch.clamp(importance_samples, near, far)
        use_uniform = temp_sum < 1e-8
        return torch.where(use_uniform.unsqueeze(-1), uniform_samples, importance_samples)

    return vmap(vmap(sample_fn, in_dims=(0, 0)), in_dims=(0, 0))(t, temp)


def sample_fn_chunked(t_row, temp_row, num_samples, smooth_factor):
    near = t_row[0]
    far = t_row[-1]
    temp_sum = temp_row.sum()
    uniform_samples = torch.linspace(near, far, num_samples, device=t_row.device)
    importance_pdf = temp_row / (temp_sum + 1e-8)
    importance_pdf = torch.clamp(importance_pdf, min=0.0)
    importance_pdf = importance_pdf / (importance_pdf.sum() + 1e-8)
    uniform_pdf = torch.ones_like(temp_row) / len(temp_row)
    mixed_pdf = (1.0 - smooth_factor) * importance_pdf + smooth_factor * uniform_pdf
    mixed_pdf = torch.clamp(mixed_pdf, min=1e-8)
    mixed_pdf = mixed_pdf / (mixed_pdf.sum() + 1e-8)
    cdf = torch.cumsum(mixed_pdf, dim=0)
    cdf = torch.cat([torch.zeros(1, device=cdf.device), cdf])
    cdf = torch.clamp(cdf, min=0.0, max=1.0)
    u = torch.linspace(0, 1, num_samples, device=t_row.device)
    idx = torch.searchsorted(cdf, u) - 1
    idx = torch.clamp(idx, 0, len(t_row) - 2)
    t_low = t_row[idx]
    t_high = t_row[idx + 1]
    cdf_low = cdf[idx]
    cdf_high = cdf[idx + 1]
    cdf_diff = cdf_high - cdf_low
    z = torch.where(cdf_diff > 1e-8, (u - cdf_low) / cdf_diff, torch.zeros_like(u))
    importance_samples = t_low + z * (t_high - t_low)
    importance_samples = torch.clamp(importance_samples, near, far)
    use_uniform = temp_sum < 1e-8
    return torch.where(use_uniform.unsqueeze(-1), uniform_samples, importance_samples)


def batched_importance_sample_nerf_vmap_chunked(t, temp, num_samples=256, smooth_factor=0.3, chunk_size=64):
    H, W, _ = temp.shape
    if t.dim() == 3 and t.shape[0] == 1 and t.shape[1] == 1:
        t = t.expand(H, W, -1)
    sampled_positions = torch.zeros((H, W, num_samples), device=t.device, dtype=t.dtype)
    for h_start in range(0, H, chunk_size):
        h_end = min(h_start + chunk_size, H)
        t_chunk = t[h_start:h_end]
        temp_chunk = temp[h_start:h_end]

        def sample_fn_wrapper(t_row, temp_row):
            return sample_fn_chunked(t_row, temp_row, num_samples, smooth_factor)

        chunk_result = vmap(vmap(sample_fn_wrapper, in_dims=(0, 0)), in_dims=(0, 0))(t_chunk, temp_chunk)
        sampled_positions[h_start:h_end] = chunk_result
    return sampled_positions


def update_transmittance(T, batch_mask, transmittance):
    H, W, B = batch_mask.shape
    device = T.device
    batch_mask = batch_mask.to(device)
    transmittance = transmittance.to(device)
    indices = torch.arange(B, device=device).view(1, 1, B).expand(H, W, B)
    masked_indices = torch.where(batch_mask, indices, -torch.inf)
    last_true_idx = torch.max(masked_indices, dim=-1, keepdim=True)[0]
    need_update = last_true_idx != -torch.inf
    safe_idx = torch.where(need_update, last_true_idx, 0).long()
    selected_T = torch.gather(T.unsqueeze(-1), dim=2, index=safe_idx.unsqueeze(-1)).squeeze(-1)
    return torch.where(need_update, selected_T, transmittance)


def render_image_fire_smoke(
    rays_o,
    rays_d,
    num_samples,
    t_vals,
    query_tensor,
    temp_converter,
    batch_size=1024,
    strength=0.01,
    sigma_a=0.1,
    is_norm_samples=False,
    args=None,
    depth_map=None,
    debug_mask=False,
    scale_ratio=0.12,
    smoke_color=[1.0, 1.0, 1.0],
    threshold=0.4,
    smoke_strengh=1.0,
    exposure=1.0,
    render_smoke=True,
    fuel2temp=None,
    color_field=None,
):
    num_batches = (num_samples + batch_size - 1) // batch_size
    sum_XYZ = torch.zeros_like(rays_o)
    sum_smoke = torch.zeros_like(rays_o)
    transmittance = torch.ones_like(rays_o[..., :1])
    delta_t = t_vals[..., 1:] - t_vals[..., :-1]
    delta_t = torch.cat([delta_t, delta_t[..., -1:]], dim=-1)
    H, W = rays_o.shape[:2]
    base_color = torch.tensor(smoke_color).to(rays_o.device)
    temp_query_tensor = fuel2temp(query_tensor)

    if depth_map is not None:
        depth = depth_map.squeeze().view(H, W, 1)
        valid_mask = t_vals <= depth
        if debug_mask:
            total_points = valid_mask.numel()
            invalid_points = total_points - valid_mask.sum().item()
            invalid_percentage = (invalid_points / total_points) * 100
            print(f"[DEBUG] Mask statistics: {invalid_points}/{total_points} points are invalid ({invalid_percentage:.2f}%)")
    else:
        valid_mask = torch.ones_like(t_vals, dtype=torch.bool)
        if debug_mask:
            print("[DEBUG] No depth map provided, all points are valid")

    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, num_samples)
        batch_mask = valid_mask[..., start:end]
        batch_points = rays_o.unsqueeze(2) + rays_d.unsqueeze(2) * t_vals[..., start:end].unsqueeze(-1)

        if is_norm_samples:
            min_bound = torch.tensor(args.bounding_box["min"], dtype=torch.float32, device=batch_points.device)
            max_bound = torch.tensor(args.bounding_box["max"], dtype=torch.float32, device=batch_points.device)
            batch_points = ((batch_points - min_bound) / (max_bound - min_bound)) * 2.0 - 1.0

        fuel_batch = interp_data(batch_points, query_tensor)
        temperature_batch = interp_data(batch_points, temp_query_tensor)
        color_batch = temp_converter(temperature_batch * scale_ratio)

        density_batch = copy.deepcopy(fuel_batch)
        density_batch[density_batch > 0] = sigma_a
        sigma_a_dt = density_batch * delta_t[..., start:end]
        one_minus_sigma = 1 - sigma_a_dt
        T = transmittance * torch.cumprod(
            torch.cat([torch.ones((H, W, 1), device=rays_o.device), one_minus_sigma[..., :-1]], dim=-1),
            dim=-1,
        )

        if depth_map is not None:
            color_batch = color_batch * batch_mask.unsqueeze(-1)

        sum_XYZ += torch.sum(T.unsqueeze(-1) * color_batch * strength * delta_t[..., start:end].unsqueeze(-1), dim=2)

        smoke_batch = copy.deepcopy(fuel_batch)
        smoke_batch[smoke_batch > threshold] = 0.0
        smoke_batch[smoke_batch > 1e-3] = sigma_a
        smoke_sigma_a_dt = smoke_batch * delta_t[..., start:end]
        if color_field is not None:
            base_color = interp_data(batch_points, color_field)
        sum_smoke += torch.sum(
            base_color * T.unsqueeze(-1) * smoke_sigma_a_dt.unsqueeze(-1) * smoke_strengh * batch_mask.unsqueeze(-1),
            dim=-2,
        )
        transmittance = update_transmittance(T, batch_mask, transmittance)

    if render_smoke:
        sum_colors = temp_converter.XYZ2RGB(sum_XYZ) + sum_smoke

        def aces_tonemap(hdr, exposure=1.0):
            hdr = hdr * exposure
            a = 2.51
            b = 0.03
            c = 2.43
            d = 0.59
            e = 0.14
            return torch.clip((hdr * (a * hdr + b)) / (hdr * (c * hdr + d) + e), 0, 1)

        sum_colors = aces_tonemap(sum_colors, exposure=exposure)
    else:
        sum_colors = temp_converter.XYZ2RGB(sum_XYZ)

    return sum_colors, transmittance


__all__ = [
    "interp_data",
    "render_coarse_image_XYZ",
    "batched_importance_sample_nerf_vmap",
    "batched_importance_sample_nerf_vmap_chunked",
    "update_transmittance",
    "render_image_fire_smoke",
]
