"""Simulation-side temperature/color conversion."""

import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from scipy.constants import c, h, k


class TemperatureToRGB:
    """Convert normalized temperature to XYZ/RGB colors via blackbody + CIE CMF."""

    def __init__(self, T_max=12000, device="cpu"):
        self.T_max = T_max
        self.device = device

        # Reuse the shared CIE table under Gaussian_Fire.
        project_root = os.path.dirname(os.path.dirname(__file__))
        cie_cmf_file_path = os.path.join(project_root, "simulation", "cie-cmf.txt")
        cmf_np = np.loadtxt(cie_cmf_file_path, usecols=(1, 2, 3))
        self.cmf = torch.tensor(cmf_np, dtype=torch.float32, device=device)

        self.wavelengths = torch.linspace(380, 780, 81, dtype=torch.float32, device=device)
        self.M_CAT02 = torch.tensor(
            [[0.7328, 0.4296, -0.1624], [-0.7036, 1.6975, 0.0061], [0.0030, 0.0136, 0.9834]],
            dtype=torch.float32,
            device=device,
        )
        self.M_CAT02_inv = torch.linalg.inv(self.M_CAT02)
        self.M_XYZ_to_sRGB = torch.tensor(
            [[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]],
            dtype=torch.float32,
            device=device,
        )

        spectrum_white = self._blackbody_spectrum(self.wavelengths, 273 + T_max)
        spectrum_white *= 1e-9
        XYZ_white = self._spectrum_to_xyz(spectrum_white.unsqueeze(0))
        self.LMS_white = torch.matmul(XYZ_white, self.M_CAT02.T)
        self._precompute_color_table()

    def _blackbody_spectrum(self, wavelength_nm, T):
        wavelength_m = wavelength_nm * 1e-9
        numerator = 2 * h * c**2 / wavelength_m**5
        denominator = torch.exp(h * c / (wavelength_m * k * T)) - 1
        return numerator / denominator

    def _spectrum_to_xyz(self, spectra):
        return torch.einsum("bw,wc->bc", spectra, self.cmf)

    def _precompute_color_table(self):
        self.T_table = torch.linspace(0, 1, 101, device=self.device)
        T_kelvin = 273 + self.T_table * self.T_max
        wavelengths_expanded = self.wavelengths.unsqueeze(0).expand(len(self.T_table), -1)
        T_expanded = T_kelvin.unsqueeze(-1).expand(-1, len(self.wavelengths))
        spectra = self._blackbody_spectrum(wavelengths_expanded, T_expanded)
        self.XYZ_table = self._spectrum_to_xyz(spectra)

    def __call__(self, temperature_norm):
        if isinstance(temperature_norm, np.ndarray):
            temperature_norm = torch.from_numpy(temperature_norm).to(self.device)

        original_shape = temperature_norm.shape
        T_flat = torch.clamp(temperature_norm.reshape(-1), 0, 1)
        t_scaled = T_flat * 100
        lower_idx = torch.clamp(t_scaled.floor().long(), 0, 99)
        upper_idx = torch.clamp(lower_idx + 1, 0, 100)
        weight = (t_scaled - lower_idx).unsqueeze(-1)
        lower_xyz = self.XYZ_table[lower_idx]
        upper_xyz = self.XYZ_table[upper_idx]
        xyz = lower_xyz + weight * (upper_xyz - lower_xyz)
        return xyz.reshape(original_shape + (3,))

    def XYZ2RGB(self, XYZ):
        if isinstance(XYZ, np.ndarray):
            XYZ = torch.from_numpy(XYZ).to(self.device)

        original_shape = XYZ.shape
        LMS = torch.matmul(XYZ.reshape(-1, 3), self.M_CAT02.T)
        LMS_adapted = LMS / self.LMS_white
        XYZ_adapted = torch.matmul(LMS_adapted, self.M_CAT02_inv.T)
        RGB_linear = torch.matmul(XYZ_adapted, self.M_XYZ_to_sRGB.T)
        RGB = torch.where(
            RGB_linear <= 0.0031308,
            12.92 * RGB_linear,
            1.055 * (RGB_linear ** (1 / 2.4)) - 0.055,
        )
        return torch.clamp(RGB, 0, 1).reshape(original_shape)


def plot_temperature_color_bar(temp_min=0, temp_max=1, n_samples=10, T_max=12000, device="cpu"):
    """Draw a simple temperature color bar for qualitative inspection."""
    del temp_min, temp_max, n_samples
    fig, ax = plt.subplots()
    converter = TemperatureToRGB(T_max=T_max, device=device)

    for i in range(24):
        ratio = i / 24.0
        T = int(273 + ratio * T_max)
        ratio_tensor = torch.tensor([ratio], device=device)
        html_rgb = converter.XYZ2RGB(converter(ratio_tensor)[0]).cpu().numpy()
        x, y = i % 6, -(i // 6)
        circle = Circle(xy=(x, y * 1.2), radius=0.4, fc=html_rgb)
        ax.add_patch(circle)
        ax.annotate(f"{T:4d} K", xy=(x, y * 1.2 - 0.5), va="center", ha="center", color=html_rgb)

    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-4.35, 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("k")
    ax.set_aspect("equal")
    plt.show()


__all__ = ["TemperatureToRGB", "plot_temperature_color_bar"]
