"""Simulation-side fuel to temperature mapping."""

import torch


class FuelToTemperatureLUT:
    """Closed-form mapping from fuel coefficient to normalized gas temperature."""

    def __init__(
        self,
        resolution=0.001,
        T_ign=0.4,
        T_max=1.0,
        Y_c=0.9,
        EPS=0.001,
        c_T=1.0,
        k=0.2,
        device="cpu",
    ):
        self.resolution = resolution
        self.T_ign = T_ign
        self.T_max = T_max
        self.Y_c = Y_c
        self.EPS = EPS
        self.c_T = c_T
        self.k = k
        self.device = device

    def __call__(self, coef):
        return torch.clamp(
            self.T_ign - (self.T_max - self.T_ign) / (0.3**2) * (coef - 1.0) * (coef - 0.4),
            min=0.0,
        )


__all__ = ["FuelToTemperatureLUT"]
