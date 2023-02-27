# -*- coding: utf-8 -*-
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
        field[:, -(j + 1)] = field[:,  -2 * grid.hy - 1 + j]


def update_halo_points(state: State) -> None:
    update_halo_points_field(state.grid, state.h)
    update_halo_points_field(state.grid, state.u)
    update_halo_points_field(state.grid, state.v)
