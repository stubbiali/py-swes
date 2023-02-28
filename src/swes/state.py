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

from swes.utils import zeros

if TYPE_CHECKING:
    from swes.grid import Grid


def diagnostics(grid: Grid, field: np.ndarray, field_name: str) -> str:
    i = slice(grid.hx, grid.hx + grid.nx)
    j = slice(grid.hy, grid.hy + grid.ny)
    return (
        f"max({field_name})={np.max(field[i, j]):.10f}, "
        f"min({field_name})={np.min(field[i, j]):.10f}, "
        f"avg({field_name})={np.sum(field[i, j]) / field[i, j].size: .10f}"
    )


class State:
    grid: Grid
    h_new: np.ndarray
    u: np.ndarray
    u_new: np.ndarray
    u_x: np.ndarray
    u_y: np.ndarray
    v: np.ndarray
    v_new: np.ndarray
    v_x: np.ndarray
    v_y: np.ndarray

    def __init__(self, grid: Grid) -> None:
        self.grid = grid

        # solution at current time level
        self.h = zeros(grid.ni, grid.nj)
        self.u = zeros(grid.ni, grid.nj)
        self.u_x = zeros(grid.ni, grid.nj)
        self.u_y = zeros(grid.ni, grid.nj)
        self.v = zeros(grid.ni, grid.nj)
        self.v_x = zeros(grid.ni, grid.nj)
        self.v_y = zeros(grid.ni, grid.nj)

        # solution at next time level
        self.h_new = zeros(grid.ni, grid.nj)
        self.u_new = zeros(grid.ni, grid.nj)
        self.v_new = zeros(grid.ni, grid.nj)

    def update_solution(self) -> None:
        self.h, self.h_new = self.h_new, self.h
        self.u, self.u_new = self.u_new, self.u
        self.v, self.v_new = self.v_new, self.v

    def __repr__(self) -> str:
        return (
            f"   {diagnostics(self.grid, self.h, 'h')}\n"
            f"   {diagnostics(self.grid, self.u, 'u')}\n"
            f"   {diagnostics(self.grid, self.v, 'v')}"
        )
