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
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from swes.grid import Grid
    from swes.state import State


def update_halo_points_field(grid: Grid, field: np.ndarray) -> None:
    # periodicity along x
    for i in range(grid.hx + 1):
        field[-(i + 1), :] = field[-(i + 1) - grid.nx, :]
        field[i, :] = field[i + grid.nx, :]

    # cross-pole conditions along y
    for j in range(grid.hy):
        field[:, j] = field[:, 2 * grid.hy - j]
        field[:, -(j + 1)] = field[:, -2 * grid.hy - 1 + j]


def update_halo_points(state: State) -> None:
    update_halo_points_field(state.grid, state.h)
    update_halo_points_field(state.grid, state.u)
    update_halo_points_field(state.grid, state.v)
