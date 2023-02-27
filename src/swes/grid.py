# -*- coding: utf-8 -*-
from __future__ import annotations
import math
import numpy as np
from typing import TYPE_CHECKING

from swes.utils import to_gt4py

if TYPE_CHECKING:
    from swes.config import Config


class Grid:
    def __init__(self, config: Config) -> None:
        # longitude
        self.nx = config.nx
        self.hx = config.hx
        self.dphi = 2 * math.pi / self.nx
        self.phic = np.linspace(
            -self.hx * self.dphi, 2 * math.pi + self.hx * self.dphi, self.nx + 1 + 2 * self.hx
        )

        # latitude
        # note: the number of grid points must be even to prevent F to vanish
        # (important for computing initial height and velocity in geostrophic balance)
        self.ny = config.ny if config.ny % 2 == 0 else config.ny + 1
        self.hy = config.hy
        self.dtheta = math.pi / (self.ny - 1)
        self.thetac = np.zeros(self.ny + 2 * self.hy)
        self.thetac[self.hy : -self.hy] = np.linspace(-0.5 * math.pi, 0.5 * math.pi, self.ny)
        self.thetac[: self.hy] = self.thetac[self.hy + 1 : 2 * self.hy + 1][::-1]
        self.thetac[-self.hy :] = self.thetac[-2 * self.hy - 1 : -self.hy - 1][::-1]

        # storage shape
        self.ni = self.nx + 1 + 2 * self.hx
        self.nj = self.ny + 2 * self.hy

        # grid
        phi, theta = np.meshgrid(self.phic, self.thetac, indexing="ij")

        # $\cos(\theta)$
        c = np.cos(theta)
        c_y = np.zeros((self.ni, self.nj))
        c_y[:, :-1] = np.cos(0.5 * (theta[:, :-1] + theta[:, 1:]))

        # $\tan(\theta)$
        tg = np.tan(theta)
        tg_x = np.zeros((self.ni, self.nj))
        tg_x[:-1, :] = np.tan(0.5 * (theta[:-1, :] + theta[1:, :]))
        tg_y = np.zeros((self.ni, self.nj))
        tg_y[:, :-1] = np.tan(0.5 * (theta[:, :-1] + theta[:, 1:]))

        # Cartesian coordinates
        self.a = config.planet_constants.a
        x = self.a * np.cos(theta) * phi
        y = self.a * theta
        y1 = self.a * np.sin(theta)

        # Cartesian increments
        dx = np.zeros((self.ni, self.nj))
        dx[:-1, :] = np.abs(x[1:, :] - x[:-1, :])
        dy = np.zeros((self.ni, self.nj))
        dy[:, :-1] = np.abs(y[:, 1:] - y[:, :-1])
        dy1 = np.zeros((self.ni, self.nj))
        dy1[:, :-1] = np.abs(y1[:, 1:] - y1[:, :-1])

        # Coriolis term
        self.omega = config.planet_constants.omega
        f = 2.0 * self.omega * np.sin(theta)

        # convert arrays to gt4py storages
        self.phi = to_gt4py(phi)
        self.theta = to_gt4py(theta)
        self.c = to_gt4py(c)
        self.c_y = to_gt4py(c_y)
        self.tg = to_gt4py(tg)
        self.tg_x = to_gt4py(tg_x)
        self.tg_y = to_gt4py(tg_y)
        self.x = to_gt4py(x)
        self.y = to_gt4py(y)
        self.y1 = to_gt4py(y1)
        self.dx = to_gt4py(dx)
        self.dy = to_gt4py(dy)
        self.dy1 = to_gt4py(dy1)
        self.f = to_gt4py(f)
