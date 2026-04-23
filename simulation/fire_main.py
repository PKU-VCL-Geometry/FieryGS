"""3D fire simulation on a staggered Eulerian grid (Taichi).
"""

import json
import math
import os
import sys
from argparse import ArgumentParser, Namespace

import copy
import matplotlib.pyplot as plt
import noise
import numpy as np
import taichi as ti
import torch
import vtk
import yaml
from torch.nn import functional as F
from tqdm import tqdm
from vtk.util import numpy_support
import imageio

from simulation.color_mapping import TemperatureToRGB

# -----------------------------------------------------------------------------
# Taichi backend & grid-wide constants
# -----------------------------------------------------------------------------
ti.init(arch=ti.gpu)
wi = 1.0 / 6
real = ti.f32


# -----------------------------------------------------------------------------
# Double-buffer helper (velocity, pressure, scalars)
# -----------------------------------------------------------------------------
class TexPair:
    def __init__(self, cur, nxt):
        self.cur = cur
        self.nxt = nxt

    def swap(self):
        self.cur, self.nxt = self.nxt, self.cur


# -----------------------------------------------------------------------------
# Taichi: NaN checks & value noise (used inside kernels / @ti.func)
# -----------------------------------------------------------------------------
@ti.func
def isnan(x):
    return not (x < 0 or 0 < x or x == 0)

@ti.func
def isnan_vector(vec):
    result = False
    for i in ti.static(range(vec.n)):
        if isnan(vec[i]):
            result = True
    return result


# -----------------------------------------------------------------------------
# NumPy / PyTorch helpers (host-side; not Taichi kernels)
# -----------------------------------------------------------------------------
def generate_3d_perlin_noise_vectorized(shape, scale=10.0, octaves=6,
                                        persistence=0.5, lacunarity=2.0, seed=42):
    """3D Perlin noise via numpy.vectorize over noise.pnoise3."""
    x = np.arange(shape[0]) / scale
    y = np.arange(shape[1]) / scale
    z = np.arange(shape[2]) / scale
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    print("正在生成3D Perlin噪声...")
    pnoise3_vec = np.vectorize(
        noise.pnoise3,
        otypes=[np.float32],
        excluded=['octaves', 'persistence', 'lacunarity', 'repeatx', 'repeaty', 'repeatz', 'base'],
    )
    return pnoise3_vec(
        X, Y, Z,
        octaves=octaves,
        persistence=persistence,
        lacunarity=lacunarity,
        repeatx=1024,
        repeaty=1024,
        repeatz=1024,
        base=seed,
    )


def interp_data(query_points_normalized, query_tensor):
    """grid_sample on normalized coords; supports 1- or 3-channel [C,D,H,W] volumes."""
    original_shape = query_points_normalized.shape
    grid = query_points_normalized.view(1, -1, 1, 1, 3)
    output = F.grid_sample(
        query_tensor,
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True,
    )
    output = output.squeeze().t()
    if query_tensor.size(1) == 1:
        return output.view(*original_shape[:-1])
    return output.view(*original_shape)


# -----------------------------------------------------------------------------
# Taichi ↔ NumPy bridge for 2D RGB slice upload
# -----------------------------------------------------------------------------
@ti.kernel
def assign_field(field: ti.template(), arr: ti.types.ndarray(dtype=ti.f32, ndim=3)):
    for i, j in field:
        for k in ti.static(range(3)):
            field[i, j][k] = arr[i, j, k]


# -----------------------------------------------------------------------------
# Taichi: value noise (2D/3D) for procedural masks inside kernels
# -----------------------------------------------------------------------------
@ti.func
def random_gradient(seed: ti.i32) -> ti.f32:
    """生成伪随机梯度值（-1 到 1）"""
    x = (seed * 127 + 123456) % 1234567
    x = (x * 1103515245 + 12345) & 0x7fffffff
    return (x / 0x7fffffff) * 2.0 - 1.0  # [-1, 1]

@ti.func
def lerp(a, b, t):
    """线性插值"""
    return a + t * (b - a)

@ti.func
def smoothstep(t):
    """平滑过渡（3t² - 2t³）"""
    return t * t * (3.0 - 2.0 * t)

@ti.func
def noise_2d(x, y):
    """2D Value Noise"""
    xi = ti.floor(x, ti.i32)
    yi = ti.floor(y, ti.i32)
    
    xf = x - xi
    yf = y - yi
    
    # 四个角点的随机值
    n00 = random_gradient(xi + yi * 57)
    n01 = random_gradient(xi + (yi + 1) * 57)
    n10 = random_gradient((xi + 1) + yi * 57)
    n11 = random_gradient((xi + 1) + (yi + 1) * 57)
    
    # 插值
    ix0 = lerp(n00, n10, smoothstep(xf))
    ix1 = lerp(n01, n11, smoothstep(xf))
    return lerp(ix0, ix1, smoothstep(yf))

@ti.func
def noise_3d(x, y, z):
    """3D Value Noise（类似 2D 但扩展到 3D）"""
    xi = ti.floor(x, ti.i32)
    yi = ti.floor(y, ti.i32)
    zi = ti.floor(z, ti.i32)
    
    xf = x - xi
    yf = y - yi
    zf = z - zi
    
    # 8 个角点的随机值
    n000 = random_gradient(xi + yi * 57 + zi * 131)
    n001 = random_gradient(xi + yi * 57 + (zi + 1) * 131)
    n010 = random_gradient(xi + (yi + 1) * 57 + zi * 131)
    n011 = random_gradient(xi + (yi + 1) * 57 + (zi + 1) * 131)
    n100 = random_gradient((xi + 1) + yi * 57 + zi * 131)
    n101 = random_gradient((xi + 1) + yi * 57 + (zi + 1) * 131)
    n110 = random_gradient((xi + 1) + (yi + 1) * 57 + zi * 131)
    n111 = random_gradient((xi + 1) + (yi + 1) * 57 + (zi + 1) * 131)
    
    # 三线性插值
    ix00 = lerp(n000, n100, smoothstep(xf))
    ix01 = lerp(n001, n101, smoothstep(xf))
    ix10 = lerp(n010, n110, smoothstep(xf))
    ix11 = lerp(n011, n111, smoothstep(xf))
    
    iy0 = lerp(ix00, ix10, smoothstep(yf))
    iy1 = lerp(ix01, ix11, smoothstep(yf))
    
    return lerp(iy0, iy1, smoothstep(zf))


@ti.data_oriented
class FireSolver:
    """Grid-based fire/fluid solver: velocity, pressure, density, fuel coefficient, temperature; optional wood burning and dual smoke."""

    def __init__(self, args, x, y, z, scene, T_white=12000, T_max=1440, T_air=273, k=0.5, burn_rate=0.1, beta=15, alpha=50, epsilon=5, sim_frames=400, nu=0.1, load_path=None):
        self.args = args
        self.scene = scene
        self.T_white_phy = T_white
        self.T_max_phy = T_max
        self.T_air_phy = T_air
        self.ratio = self.T_max_phy / self.T_white_phy

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.converter = TemperatureToRGB(T_max=self.T_white_phy, device=self.device)

        self._setup_grid_and_time(x, y, z, sim_frames)
        self._setup_combustion_params(k, alpha, epsilon, nu, burn_rate, beta)
        self._setup_taichi_fluid_fields()
        self._setup_occupancy_from_file(load_path)
        self._setup_wood_plane_and_display()
        self._setup_io_flags()
        self._setup_dual_smoke_if_needed()
        self._setup_scene_noise_fields()

    # --- Construction helpers -------------------------------------------------

    def _setup_grid_and_time(self, x: int, y: int, z: int, sim_frames: int) -> None:
        self.dim = 3
        self.res = [x, y, z]
        self.resx, self.resy, self.resz = self.res
        self.dx = 1.0
        self.dt = 0.04
        self.inv_dx = 1.0 / self.dx
        self.half_inv_dx = 0.5 * self.inv_dx
        self.p_alpha = -self.dx * self.dx
        self.max_iter = 60
        self.sim_frames = sim_frames

    def _setup_combustion_params(self, k, alpha, epsilon, nu, burn_rate, beta) -> None:
        self.EPS = 0.001
        self.k = k
        self.S = 0.15
        self.T_air = 0
        self.T_ign = 0.4
        self.T_max = 1
        self.Y_c = 0.9
        self.c_T = 2.0
        self.alpha = alpha
        self.epsilon = epsilon
        self.sigma = 6.67e-14
        self.nu = nu

        self.T_air_wood = 0
        self.T_ign_wood = 1.0
        self.T_max_wood = 1.2
        self.burn_rate = burn_rate
        self.c_m = 1.0
        self.beta = beta

    def _setup_taichi_fluid_fields(self) -> None:
        self._velocities = ti.Vector.field(3, dtype=real, shape=self.res)
        self._new_velocities = ti.Vector.field(3, dtype=real, shape=self.res)
        self._vel_temp = ti.Vector.field(3, dtype=real, shape=self.res)
        self.velocity_divs = ti.field(dtype=real, shape=self.res)
        self._pressures = ti.field(dtype=real, shape=self.res)
        self._new_pressures = ti.field(dtype=real, shape=self.res)
        self._dens_buffer = ti.field(dtype=real, shape=self.res)
        self._new_dens_buffer = ti.field(dtype=real, shape=self.res)

        self._coef_buffer = ti.field(dtype=real, shape=self.res)
        self._new_coef_buffer = ti.field(dtype=real, shape=self.res)
        self._temperature_buffer = ti.field(dtype=real, shape=self.res)
        self._new_temperature_buffer = ti.field(dtype=real, shape=self.res)

        self.velocities_pair = TexPair(self._velocities, self._new_velocities)
        self.pressures_pair = TexPair(self._pressures, self._new_pressures)
        self.dens_pair = TexPair(self._dens_buffer, self._new_dens_buffer)

        self.coef_pair = TexPair(self._coef_buffer, self._new_coef_buffer)
        self.temperature_pair = TexPair(self._temperature_buffer, self._new_temperature_buffer)

        self.N = ti.Vector.field(3, dtype=real, shape=self.res)
        self.curl = ti.Vector.field(3, dtype=real, shape=self.res)

    def _setup_occupancy_from_file(self, load_path) -> None:
        self.occupancy = ti.field(dtype=real, shape=self.res)
        occupancy_np = np.load(load_path)
        occupancy_np[occupancy_np > 0] = 1
        self.occupancy.from_numpy(occupancy_np)

        self.occupancy_static = ti.field(dtype=real, shape=self.res)
        self.occupancy_static.from_numpy(occupancy_np)

    def _setup_wood_plane_and_display(self) -> None:
        self.wood_temperature = ti.field(dtype=real, shape=self.res)
        self.new_wood_temperature = ti.field(dtype=real, shape=self.res)
        self.wood_temperature_pair = TexPair(self.wood_temperature, self.new_wood_temperature)

        self._img = ti.Vector.field(3, dtype=ti.f32, shape=(self.resx + self.resx, self.resy))

    def _setup_io_flags(self) -> None:
        self._data_dir = self.args.fire_sim_root
        self.sim_wood = False

        self.need_save_vti = self.args.save_vti
        self.need_save_npz = self.args.save_npz

    def _setup_dual_smoke_if_needed(self) -> None:
        if not self.args.dual_smoke:
            return
        self._color_grid = ti.Vector.field(3, dtype=ti.f32, shape=self.res)
        self._new_color_grid = ti.Vector.field(3, dtype=ti.f32, shape=self.res)
        self.color_buffer = TexPair(self._color_grid, self._new_color_grid)
        color_np = np.load(self.args.material_path)
        self.assign_smoke_color(color_np, self.color_buffer.cur)

    def _setup_scene_noise_fields(self) -> None:
        shape = (64, 64, 64)
        scale, octaves, persistence, lacunarity = 20.0, 6, 0.5, 2.0
        noise_ignite = generate_3d_perlin_noise_vectorized(shape, scale, octaves, persistence, lacunarity, seed=42)
        noise_burn = generate_3d_perlin_noise_vectorized(shape, scale, octaves, persistence, lacunarity, seed=24)
        noise_ignite = (noise_ignite - noise_ignite.min()) / (noise_ignite.max() - noise_ignite.min())
        noise_burn = (noise_burn - noise_burn.min()) / (noise_burn.max() - noise_burn.min())
        self._noise_ignite_field = ti.field(dtype=real, shape=(64, 64, 64))
        self._noise_burn_field = ti.field(dtype=real, shape=(64, 64, 64))
        self._noise_ignite_field.from_numpy(noise_ignite)
        self._noise_burn_field.from_numpy(noise_burn)

    # --- Dual smoke, debug plots, PyTorch volume rendering --------------------

    @ti.kernel
    def assign_smoke_color(self, mask: ti.types.ndarray(), vec_field: ti.template()):
        for i, j, k in ti.ndrange(self.resx, self.resy, self.resz):
            if mask[i, j, k] == 4:
                vec_field[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            else:
                vec_field[i, j, k] = ti.Vector([1.0, 1.0, 1.0])

    # --- NaN diagnostics ------------------------------------------------------

    @ti.kernel
    def isnan_scalar_field(self, vf: ti.template()) -> bool:
        found = False
        for i, j, k in vf:
            if isnan(vf[i, j, k]):
                found = True
                print(f"found nan at: ({i}, {j}, {k})")
        return found
    
    @ti.kernel
    def isnan_vector_field(self, vf: ti.template()) -> bool:
        found = False
        for i, j, k in vf:
            if isnan(vf[i, j, k][0]) or isnan(vf[i, j, k][1]) or isnan(vf[i, j, k][2]):
                found = True
                print(f"found nan at: ({i}, {j}, {k})")
        return found

    # --- Interpolation & semi-Lagrangian advection ------------------------------

    @ti.func
    def sample(self, qf, u, v, w):
        i, j, k = int(u), int(v), int(w)
        i = max(0, min(self.resx - 1, i))
        j = max(0, min(self.resy - 1, j))
        k = max(0, min(self.resz - 1, k))
        return qf[i, j, k]

    @ti.func
    def lerp(self, vl, vr, frac):
        # frac: [0.0, 1.0]
        return vl + frac * (vr - vl)

    @ti.func
    def trilerp(self, vf, u, v, w):
        s, t, n = u - 0.5, v - 0.5, w - 0.5
        iu, iv, iw = ti.floor(s), ti.floor(t), ti.floor(n)
        fu, fv, fw = s - iu, t - iv, n - iw
        a = self.sample(vf, iu + 0.5, iv + 0.5, iw + 0.5)
        b = self.sample(vf, iu + 1.5, iv + 0.5, iw + 0.5)
        c = self.sample(vf, iu + 0.5, iv + 1.5, iw + 0.5)
        d = self.sample(vf, iu + 1.5, iv + 1.5, iw + 0.5)
        e = self.sample(vf, iu + 0.5, iv + 0.5, iw + 1.5)
        f = self.sample(vf, iu + 1.5, iv + 0.5, iw + 1.5)
        g = self.sample(vf, iu + 0.5, iv + 1.5, iw + 1.5)
        h = self.sample(vf, iu + 1.5, iv + 1.5, iw + 1.5)

        bilerp1 = self.lerp(self.lerp(a, b, fu), self.lerp(c, d, fu), fv)
        bilerp2 = self.lerp(self.lerp(e, f, fu), self.lerp(g, h, fu), fv)
        return self.lerp(bilerp1, bilerp2, fw)

    @ti.func
    def mult_const(self, qf: ti.template(), vec3):
        for i, j, k in qf:
            qf[i, j, k] *= vec3

    @ti.func
    def add_scaled(self, qf: ti.template(), bf: ti.template(), vec3):
        for i, j, k in qf:
            qf[i, j, k] += bf[i, j, k] * vec3

    @ti.func
    def back_trace_rk2(self, vf: ti.template(), pos, delta_t):
        mid = pos - 0.5 * delta_t * self.trilerp(vf, pos[0], pos[1], pos[2])
        coord = pos - delta_t * self.trilerp(vf, mid[0], mid[1], mid[2])
        return coord

    @ti.kernel
    def advect_semi_l(self, vf: ti.template(), qf: ti.template(), new_qf: ti.template()):
        for i, j, k in vf:
            pos = ti.Vector([i, j, k]) + 0.5 * self.dx
            coord = self.back_trace_rk2(vf, pos, self.dt)
            new_qf[i, j, k] = self.trilerp(qf, coord[0], coord[1], coord[2])
    
    @ti.kernel
    def advect_semi_l_scalar(self, vf: ti.template(), qf: ti.template(), new_qf: ti.template()):
        for i, j, k in vf:
            pos = ti.Vector([i, j, k]) + 0.5 * self.dx
            coord = self.back_trace_rk2(vf, pos, self.dt)
            new_qf[i, j, k] = self.trilerp(qf, coord[0], coord[1], coord[2])

    # --- Divergence, Poisson (Gauss–Seidel), pressure gradient ----------------

    @ti.kernel
    def divergence(self, vf: ti.template()):
        outflow = 0.0
        for i, j, k in vf:
            vl = self.sample(vf, i - 1, j, k)[0]
            vr = self.sample(vf, i + 1, j, k)[0]
            vb = self.sample(vf, i, j - 1, k)[1]
            vt = self.sample(vf, i, j + 1, k)[1]
            vh = self.sample(vf, i, j, k - 1)[2]
            vq = self.sample(vf, i, j, k + 1)[2]
            vc = self.sample(vf, i, j, k)
            if self.sample(self.occupancy_static, i-1, j, k) > 0:
                vl = -(vc[0] - outflow) if vc[0] < 0 else vc[0]
            if self.sample(self.occupancy_static, i+1, j, k) > 0:
                vr = -(vc[0] + outflow) if vc[0] > 0 else vc[0]
            if self.sample(self.occupancy_static, i, j-1, k) > 0:
                vb = -(vc[1] - outflow) if vc[1] < 0 else vc[1]
            if self.sample(self.occupancy_static, i, j+1, k) > 0:
                vt = -(vc[1] + outflow) if vc[1] > 0 else vc[1]
            if self.sample(self.occupancy_static, i, j, k-1) > 0:
                vh = -(vc[2] - outflow) if vc[2] < 0 else vc[2]
            if self.sample(self.occupancy_static, i, j, k+1) > 0:
                vq = -(vc[2] + outflow) if vc[2] > 0 else vc[2]
            self.velocity_divs[i, j, k] = (vr - vl + vt - vb + vq - vh) * self.half_inv_dx

    @ti.kernel
    def Gauss_Seidel(self, pf: ti.template(), new_pf: ti.template()) -> float:
        for i, j, k in pf:
            if (i + j + k) % 2 == 0:
                pl = self.sample(pf, i - 1, j, k)
                pr = self.sample(pf, i + 1, j, k)
                pb = self.sample(pf, i, j - 1, k)
                pt = self.sample(pf, i, j + 1, k)
                ph = self.sample(pf, i, j, k - 1)
                pq = self.sample(pf, i, j, k + 1)
                div = self.velocity_divs[i, j, k]
                new_pf[i, j, k] = (pl + pr + pb + pt + ph + pq + self.p_alpha * div) * wi
        for i, j, k in pf:
            if (i + j + k) % 2 == 1:
                pl = self.sample(new_pf, i - 1, j, k)
                pr = self.sample(new_pf, i + 1, j, k)
                pb = self.sample(new_pf, i, j - 1, k)
                pt = self.sample(new_pf, i, j + 1, k)
                ph = self.sample(new_pf, i, j, k - 1)
                pq = self.sample(new_pf, i, j, k + 1)
                div = self.velocity_divs[i, j, k]
                new_pf[i, j, k] = (pl + pr + pb + pt + ph + pq + self.p_alpha * div) * wi
        residual = 0.0
        cnt = 0
        for i, j, k in pf:
            residual += ti.abs(new_pf[i, j, k] - pf[i, j, k])
            cnt += 1
        return residual / cnt

    @ti.kernel
    def subtract_gradient(self, vf: ti.template(), pf: ti.template()):
        for i, j, k in vf:
            pl = self.sample(pf, i - 1, j, k)
            pr = self.sample(pf, i + 1, j, k)
            pb = self.sample(pf, i, j - 1, k)
            pt = self.sample(pf, i, j + 1, k)
            ph = self.sample(pf, i, j, k - 1)
            pq = self.sample(pf, i, j, k + 1)
            v = self.sample(vf, i, j, k)
            v = v - self.half_inv_dx * ti.Vector([pr - pl, pt - pb, pq - ph])
            vf[i, j, k] = v

    @ti.kernel
    def my_copy_from(self, af: ti.template(), bf: ti.template()):
        for i, j, k in af:
            af[i, j, k] = bf[i, j, k]

    # --- Buoyancy source, slice buffers, timestep helpers ---------------------

    @ti.kernel
    def source(self):
        for i, j, k in self._velocities:
            self._velocities[i, j, k][2] = 50 if self.inside_fire(i, j, k) else 0

    def to_image_color(self):
        temperature_field = self._temperature_buffer.to_torch(device=self.device).float()
        xyz_field = self.converter(temperature_field * self.ratio)
        rgb_field = self.converter.XYZ2RGB(xyz_field)
        rgb_np = rgb_field.detach().cpu().numpy()
        slice1 = rgb_np[:, self.resy // 2, :, :]  # (128, 128, 3)
        slice2 = rgb_np[self.resx // 2, :, :, :]  # (128, 128, 3)
        result = np.concatenate([slice1, slice2], axis=0)
        assign_field(self._img, result)
        result = np.round(result * 255).astype(np.uint8).swapaxes(0, 1)
        result_flip = result[::-1, :, :]
        return result_flip

    def set_max_iter(self, val):
        self.max_iter = val

    @ti.kernel
    def average(self, sf: ti.template()) -> real:
        sum = 0.0
        cnt = 0
        for i, j, k in sf:
            sum += sf[i, j, k]
            cnt += 1
        return sum / cnt

    # --- Scene masks: ignite regions & geometry tests -------------------------
    
    @ti.func
    def inside_fire_firewoods_sand(self, u, v, w) -> bool:
        return w >= 0.5 * self.resz and not self.is_boundary(u, v, w)

    @ti.func
    def inside_fire_playground(self, u, v, w) -> bool:
        norm_u = u / self.resx
        norm_v = v / self.resy
        norm_w = w / self.resz
        noise_val = self.trilerp(self._noise_ignite_field, norm_u * 64.0, norm_v * 64.0, norm_w * 64.0)
        return noise_val > 0.7 and not self.is_boundary(u, v, w)

    # --- Wood temperature, occupancy burn, fuel coef, gas temperature --------

    @ti.kernel
    def update_wood_temperature(self, tf: ti.template(), new_tf: ti.template(), sub_time_step: real):
        for i, j, k in tf:
            if self.occupancy[i, j, k] > 0:
                if tf[i, j, k] >= self.T_ign_wood:
                    new_tf[i, j, k] = self.T_max_wood
                else:
                    laplacian = 0.0
                    center = tf[i, j, k]
                    laplacian += (self.sample(tf, i+1, j, k) - center) * self.inv_dx if self.sample(self.occupancy, i+1, j, k) > 0 else 0
                    laplacian += (self.sample(tf, i-1, j, k) - center) * self.inv_dx if self.sample(self.occupancy, i-1, j, k) > 0 else 0
                    laplacian += (self.sample(tf, i, j+1, k) - center) * self.inv_dx if self.sample(self.occupancy, i, j+1, k) > 0 else 0
                    laplacian += (self.sample(tf, i, j-1, k) - center) * self.inv_dx if self.sample(self.occupancy, i, j-1, k) > 0 else 0
                    laplacian += (self.sample(tf, i, j, k+1) - center) * self.inv_dx if self.sample(self.occupancy, i, j, k+1) > 0 else 0
                    laplacian += (self.sample(tf, i, j, k-1) - center) * self.inv_dx if self.sample(self.occupancy, i, j, k-1) > 0 else 0
                    laplacian *= self.inv_dx
                    new_tf[i, j, k] = tf[i, j, k] + self.beta * laplacian * sub_time_step
            else:
                new_tf[i, j, k] = self.T_air_wood
        
        for i, j, k in new_tf:
            if new_tf[i, j, k] < self.T_ign_wood:
                new_tf[i, j, k] += self.c_m * (ti.pow(self.T_air_wood, 4) - ti.pow(new_tf[i, j, k], 4)) * sub_time_step
    

    @ti.kernel
    def update_occupancy(self, qf: ti.template(), coef: ti.template()):
        for i, j, k in qf:
            if qf[i, j, k] > 0 and self.wood_temperature[i, j, k] >= self.T_ign_wood:
                norm_u = i / self.resx
                norm_v = j / self.resy
                norm_w = k / self.resz
                noise_val = self.trilerp(self._noise_burn_field, norm_u * 64.0, norm_v * 64.0, norm_w * 64.0)
                noise_val = (noise_val - 0.5) * 2.0  # [-1,1]
                
                ratio = 1.0 + noise_val * 0.5
                qf[i, j, k] -= ratio * self.burn_rate * self.dt
                if qf[i, j, k] <= 1e-3:
                    qf[i, j, k] = 0
                    coef[i, j, k] = 0

    
    @ti.kernel
    def update_coef(self, vf: ti.template()):
        for i, j, k in vf:
            if self.wood_temperature[i, j, k] >= self.T_ign_wood and self.occupancy[i, j, k] > 0:
                vf[i, j, k] = 1.0
            else:
                vf[i, j, k] = vf[i, j, k] - self.k * self.dt if vf[i, j, k] > 0 else 0
    
    @ti.kernel
    def update_temperature(self, vf: ti.template()): 
        for i, j, k in vf:
            coef = self.coef_pair.cur[i, j, k]
            vf[i, j, k] = max(self.T_ign - (self.T_max - self.T_ign) / ((0.3) ** 2) * (coef - 1.0) * (coef - 0.4), 0.0)

    @ti.func
    def is_boundary(self, u, v, w) -> bool:
        return u < 5 or u >= self.resx - 5 or v < 5 or v >= self.resy - 5 or w < 5 or w >= self.resz - 5

    # --- Velocity: buoyancy, curl / vorticity confinement, viscosity ----------

    @ti.kernel
    def update_velocity(self, vf: ti.template(), vf_new: ti.template()):
        for i, j, k in vf:
            vf_new[i, j, k] = vf[i, j, k]
            if not self.is_boundary(i, j, k):
                vf_new[i, j, k][2] += self.alpha * (self.temperature_pair.cur[i, j, k] - self.T_air) * self.dt
        
        for i, j, k in vf:
            vr = self.sample(vf, i, j+1, k)[2]
            vl = self.sample(vf, i, j-1, k)[2]
            vt = self.sample(vf, i, j, k+1)[1]
            vb = self.sample(vf, i, j, k-1)[1]
            self.curl[i, j, k][0] = (vr - vl - (vt - vb)) * self.inv_dx * 0.5
            
            vr = self.sample(vf, i, j, k+1)[0]
            vl = self.sample(vf, i, j, k-1)[0]
            vt = self.sample(vf, i+1, j, k)[2]
            vb = self.sample(vf, i-1, j, k)[2]
            self.curl[i, j, k][1] = (vr - vl - (vt - vb)) * self.inv_dx * 0.5
            
            vr = self.sample(vf, i+1, j, k)[1]
            vl = self.sample(vf, i-1, j, k)[1]
            vt = self.sample(vf, i, j+1, k)[0]
            vb = self.sample(vf, i, j-1, k)[0]
            self.curl[i, j, k][2] = (vr - vl - (vt - vb)) * self.inv_dx * 0.5
        
        for i, j, k in self.curl:
            curl_r = self.sample(self.curl, i+1, j, k)[0]
            curl_l = self.sample(self.curl, i-1, j, k)[0]
            curl_t = self.sample(self.curl, i, j+1, k)[1]
            curl_b = self.sample(self.curl, i, j-1, k)[1]
            curl_q = self.sample(self.curl, i, j, k-1)[2]
            curl_h = self.sample(self.curl, i, j, k+1)[2]
            grad_x = curl_r - curl_l
            grad_y = curl_t - curl_b
            grad_z = curl_h - curl_q
            grad_norm = ti.sqrt(grad_x**2 + grad_y**2 + grad_z**2 + 1e-6) 
            self.N[i, j, k][0] = grad_x / grad_norm
            self.N[i, j, k][1] = grad_y / grad_norm
            self.N[i, j, k][2] = grad_z / grad_norm
        
        for i, j, k in vf:
            vf_new[i, j, k][0] += self.epsilon * self.dx * (self.N[i, j, k][1] * self.curl[i, j, k][2] - self.N[i, j, k][2] * self.curl[i, j, k][1]) * self.dt
            vf_new[i, j, k][1] += self.epsilon * self.dx * (self.N[i, j, k][2] * self.curl[i, j, k][0] - self.N[i, j, k][0] * self.curl[i, j, k][2]) * self.dt
            vf_new[i, j, k][2] += self.epsilon * self.dx * (self.N[i, j, k][0] * self.curl[i, j, k][1] - self.N[i, j, k][1] * self.curl[i, j, k][0]) * self.dt
        
        for i, j, k in vf:
            if not self.is_boundary(i, j, k):
                vr = self.sample(vf, i+1, j, k)
                vl = self.sample(vf, i-1, j, k)
                vu = self.sample(vf, i, j+1, k)
                vb = self.sample(vf, i, j-1, k)
                vq = self.sample(vf, i, j, k+1)
                vh = self.sample(vf, i, j, k-1)
                vm = self.sample(vf, i, j, k)
                vf_new[i, j, k] += self.nu * (vr + vl + vu + vb + vq + vh - 6 * vm) * self.inv_dx ** 2 * self.dt

    # --- Domain boundaries: solids -----------------

    @ti.kernel
    def enforce_boundary(self, vf: ti.template()):
        for i, j, k in vf:
            if self.occupancy_static[i, j, k] > 0:
                vf[i, j, k] = ti.Vector([0.0, 0.0, 0.0])

    # --- One timestep & field reset ------------------------------------------

    def step(self):
        # Advect → wood/occupancy → combustion scalars → pressure projection (Gauss–Seidel) → gradient subtract.
        self.advect_semi_l(self.velocities_pair.cur, self.velocities_pair.cur, self.velocities_pair.nxt)
        self.advect_semi_l(self.velocities_pair.cur, self.dens_pair.cur, self.dens_pair.nxt)
        self.velocities_pair.swap()
        self.dens_pair.swap()

        if self.args.dual_smoke:
            self.advect_semi_l(self.velocities_pair.cur, self.color_buffer.cur, self.color_buffer.nxt)
            self.color_buffer.swap()

        self.advect_semi_l(self.velocities_pair.cur, self.temperature_pair.cur, self.temperature_pair.nxt)
        self.advect_semi_l(self.velocities_pair.cur, self.coef_pair.cur, self.coef_pair.nxt)
        self.temperature_pair.swap()
        self.coef_pair.swap()

        if self.sim_wood:
            for _ in range(100):
                self.update_wood_temperature(self.wood_temperature_pair.cur, self.wood_temperature_pair.nxt, self.dt / 100)
                self.wood_temperature_pair.swap()
            self.update_occupancy(self.occupancy, self.coef_pair.cur)
        self.update_coef(self.coef_pair.cur)
        self.update_temperature(self.temperature_pair.cur)
        self.update_velocity(self.velocities_pair.cur, self.velocities_pair.nxt)
        self.velocities_pair.swap()
        self.divergence(self.velocities_pair.cur)

        for cur_iter in range(self.max_iter):
            residual = self.Gauss_Seidel(self.pressures_pair.cur, self.pressures_pair.nxt)
            self.pressures_pair.swap()
            if ti.abs(residual) < 1e-6:
                break
        if math.isnan(residual):
            exit(1)
        else:
            print("residual: ", residual)
        self.epsilon *= self.args.epilon_decay
        self.subtract_gradient(self.velocities_pair.cur, self.pressures_pair.cur)
        self.enforce_boundary(self.velocities_pair.cur)


    def reset(self):
        self._dens_buffer.fill(0.0)
        self._velocities.fill(0.0)
        self._pressures.fill(0.0)
        self._coef_buffer.fill(0.0)
        self._temperature_buffer.fill(0.0)

    # --- init_wood, output dirs, camera / rays for offline render --------------

    @ti.kernel
    def init_wood(self):
        for i, j, k in self.occupancy:
            if self.is_boundary(i, j, k):
                self.occupancy_static[i, j, k] = 0.0
        if self.scene == "firewoods_sand_dark":
            for i, j, k in self.wood_temperature:
                self.wood_temperature[i, j, k] = self.T_max_wood if self.inside_fire_firewoods_sand(i, j, k) else self.T_air_wood
        elif self.scene == "playground":
            for i, j, k in self.wood_temperature:
                self.wood_temperature[i, j, k] = self.T_max_wood if self.inside_fire_playground(i, j, k) else self.T_air_wood
        else:
            print("Unknown scene: ", self.scene)
        
    def init_output(self):
        os.makedirs(self._data_dir, exist_ok=True)
        os.makedirs(os.path.join(self._data_dir, "slice"), exist_ok=True)
        os.makedirs(os.path.join(self._data_dir, "occupancy"), exist_ok=True)
        os.makedirs(os.path.join(self._data_dir, "temperature"), exist_ok=True)
        os.makedirs(os.path.join(self._data_dir, "fuel"), exist_ok=True)
        os.makedirs(os.path.join(self._data_dir, "vti"), exist_ok=True)
        if self.args.dual_smoke:
            os.makedirs(os.path.join(self._data_dir, "color"), exist_ok=True)

    # --- Export (NPZ, VTI) & offline volume render helpers --------------------

    def save_npz(self, frame):
        np.savez_compressed(f"{self._data_dir}/temperature/temperature_{frame}.npz", self._temperature_buffer.to_numpy())
        np.savez_compressed(f"{self._data_dir}/fuel/fuel_{frame}.npz", self._coef_buffer.to_numpy())
        np.savez_compressed(f"{self._data_dir}/occupancy/occupancy_{frame}.npz", self.occupancy.to_numpy())
        if self.args.dual_smoke:
            np.savez_compressed(f"{self._data_dir}/color/color_{frame}.npz", self.color_buffer.cur.to_numpy())
    
    def save_vti(self, frame):
        """Write float32 vtkImageData (.vti) for fuel and wood temperature (ParaView)."""
        def _write_scalar_vti(path: str, volume: np.ndarray, corner_value: float):
            data = np.asarray(volume, dtype=np.float32).copy()
            data[0, 0, 0] = np.float32(corner_value)
            vtk_data = vtk.vtkImageData()
            vtk_data.SetDimensions(data.shape)
            vtk_data.SetSpacing(1, 1, 1)
            vtk_array = numpy_support.numpy_to_vtk(data.ravel(order='F'), deep=True, array_type=vtk.VTK_FLOAT)
            vtk_array.SetName('scalar_data')
            vtk_data.GetPointData().SetScalars(vtk_array)
            writer = vtk.vtkXMLImageDataWriter()
            writer.SetFileName(path)
            writer.SetInputData(vtk_data)
            writer.Write()

        _write_scalar_vti(f'{self._data_dir}/vti/fuel_{frame}.vti', self._coef_buffer.to_numpy(), 1.0)
        _write_scalar_vti(
            f'{self._data_dir}/vti/wood_temperature_{frame}.vti',
            self.wood_temperature_pair.cur.to_numpy(),
            float(self.T_max_wood),
        )

    # --- Main simulation loop -------------------------------------------------

    def run(self):
        """Main loop: timestep, slice PNGs, optional NPZ/VTI dumps"""
        self.init_output()
        self.reset()
        self.init_wood()
        slice_imgs = []

        for count in tqdm(range(self.sim_frames), ncols=60):
            self.step()
            result_flip = self.to_image_color()
            imageio.imwrite(os.path.join(self._data_dir, "slice", f"slice_{count}.png"), result_flip)
            slice_imgs.append(result_flip)

            if self.need_save_npz:
                self.save_npz(count)
            if self.need_save_vti and count % 10 == 0:
                self.save_vti(count//10)

        imageio.mimwrite(os.path.join(self._data_dir, "slice.mp4"), slice_imgs, fps=25)

def main(args):
    args.nu = getattr(args, 'nu', 0.1)
    args.dual_smoke = getattr(args, 'dual_smoke', False)
    args.epilon_decay = getattr(args, 'epilon_decay', 1.0)
    # Unify simulation outputs under output/<scene>/sim_output.
    args.fire_sim_root = os.path.join("output", args.scene, "sim_output")
    dims = args.voxel_grid['dims']
    fire_solver = FireSolver(
        args, dims[0], dims[1], dims[2], args.scene,
        T_white=args.T_white, T_max=args.T_max, T_air=args.T_air,
        k=args.k, burn_rate=args.burn_rate, beta=args.beta, alpha=args.alpha, epsilon=args.epsilon,
        sim_frames=args.sim_frames, load_path=args.load_path, nu=args.nu,
    )
    fire_solver.set_max_iter(30)
    fire_solver.sim_wood = True
    fire_solver.run()


def load_yaml_config(yaml_path):
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)

def update_args_with_yaml(args, yaml_config):
    for key, value in yaml_config.items():
        if not hasattr(args, key):
            setattr(args, key, value)


if __name__ == '__main__':
    parser = ArgumentParser(description="loading simulation parameters")
    parser.add_argument("--config_file", type=str, help="Path to the YAML configuration file")

    args = parser.parse_args(sys.argv[1:])

    if args.config_file:
        yaml_config = load_yaml_config(args.config_file)
        update_args_with_yaml(args, yaml_config)

    main(args)