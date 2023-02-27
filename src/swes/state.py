# -*- coding: utf-8 -*-
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
