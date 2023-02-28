# -*- coding: utf-8 -*-
#
# py-swes: A Python solver for the Shallow Water Equations over a Sphere (SWES)
# Copyright (C) 2016-2023, ETH Zurich
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations
import math
import numpy as np
from typing import TYPE_CHECKING

from swes.utils import to_gt4py

if TYPE_CHECKING:
    from swes.build_config import float_type
    from swes.config import Config


class Grid:
    a: float_type
    c: np.ndarray
    c_y: np.ndarray
    dphi: float
    dtheta: float
    dx: np.ndarray
    dy1: np.ndarray
    dy: np.ndarray
    f: np.ndarray
    hx: int
    hy: int
    ni: int
    nj: int
    nx: int
    ny: int
    omega: float_type
    phi: np.ndarray
    phic: np.ndarray
    tg: np.ndarray
    tg_x: np.ndarray
    tg_y: np.ndarray
    theta: np.ndarray
    thetac: np.ndarray
    x: np.ndarray
    y1: np.ndarray
    y: np.ndarray

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
        self.thetac[self.hy : self.hy + self.ny] = np.linspace(
            -0.5 * math.pi, 0.5 * math.pi, self.ny
        )
        self.thetac[: self.hy] = self.thetac[self.hy + 1 : 2 * self.hy + 1][::-1]
        self.thetac[self.hy + self.ny :] = self.thetac[-2 * self.hy - 1 : -self.hy - 1][::-1]

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
