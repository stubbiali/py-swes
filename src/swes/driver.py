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

from swes.build_config import float_type
from swes.diffusion import get_diffusion
from swes.grid import Grid
from swes.halo import update_halo_points
from swes.initialization import get_initial_state
from swes.orography import Orography
from swes.output import NetCDFWriter
from swes.solver import get_solver
from swes.utils import get_time_string

if TYPE_CHECKING:
    from typing import Optional, Union

    from swes.config import Config
    from swes.diffusion import Diffusion
    from swes.solver import LaxWendroff, LaxWendroffAdvectionOnly
    from swes.state import State


class Driver:
    check_points: list[float]
    config: Config
    diffusion: Optional[Diffusion]
    dx_min: float
    dy_min: float
    grid: Grid
    orography: Orography
    solver: Union[LaxWendroff, LaxWendroffAdvectionOnly]
    writer: NetCDFWriter

    def __init__(self, config: Config) -> None:
        self.config = config

        self.grid = Grid(config)
        self.orography = Orography(self.grid)
        self.solver = get_solver(config, self.grid, self.orography)
        self.diffusion = get_diffusion(config, self.grid)

        hx, hy = self.grid.hx, self.grid.hy
        self.dx_min = np.min(self.grid.dx[hx : -hx - 1, hy + 1 : -hy - 1])
        self.dy_min = np.min(self.grid.dy[hx : -hx - 1, hy + 1 : -hy - 1])

        self.check_points = [0]
        if config.print_interval > 0:
            n = 1
            while n * config.print_interval < config.final_time:
                self.check_points.append(n * config.print_interval)
                n += 1
        self.check_points.append(config.final_time)

        self.writer = NetCDFWriter(config)

    def get_timestep(self, state: State, t: float_type) -> tuple[float_type, float_type]:
        g = self.config.planet_constants.g
        i = slice(self.grid.hx, self.grid.hx + self.grid.nx)
        j = slice(self.grid.hy + 1, self.grid.hy + self.grid.ny - 2)
        gh = np.sqrt(g * np.abs(state.h[i, j]))
        eigen_x = np.max(
            np.maximum(
                np.abs(state.u[i, j] - gh),
                np.maximum(np.abs(state.u[i, j]), np.abs(state.u[i, j] + gh)),
            )
        )
        eigen_y = np.max(
            np.maximum(
                np.abs(state.v[i, j] - gh),
                np.maximum(np.abs(state.v[i, j]), np.abs(state.v[i, j] + gh)),
            )
        )
        dt_max = np.minimum(self.dx_min / eigen_x, self.dy_min / eigen_y)
        dt = self.config.cfl * dt_max

        check_point = [cp for cp in self.check_points if t < cp <= t + dt]
        if check_point:
            dt = check_point[0] - t
            t_new = check_point[0]
        else:
            t_new = t + dt

        return float_type(dt), float_type(t_new)

    def run(self) -> State:
        state, pole_treatment = get_initial_state(self.config, self.grid, self.orography)
        t = float_type(0)

        print(f"Starting simulation '{self.config.use_case}'")
        print(f"Simulation time: {get_time_string(t)}:\n{state}")
        self.writer(state, t)

        while t < self.config.final_time:
            dt, t_new = self.get_timestep(state, t)
            self.solver(state, dt)
            if self.diffusion is not None:
                self.diffusion(state, dt)
            state.update_solution()
            pole_treatment(state, dt)
            update_halo_points(state)

            t = t_new
            if t in self.check_points:
                print(f"Simulation time: {get_time_string(t)}:\n{state}")
                self.writer(state, t)

        return state
