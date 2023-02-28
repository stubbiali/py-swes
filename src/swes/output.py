# -*- coding: utf-8 -*-
from __future__ import annotations
import netCDF4 as nc
import os
from typing import TYPE_CHECKING

from swes.build_config import float_type

if TYPE_CHECKING:
    from typing import Optional

    from swes.config import Config
    from swes.state import State


class NetCDFWriter:
    counter: int
    output_directory: Optional[str]

    def __init__(self, config: Config) -> None:
        self.output_directory = config.output_directory
        self.counter = 0

    def __call__(self, state: State, t: float_type) -> None:
        if self.output_directory is not None:
            nx, ny = state.grid.nx, state.grid.ny
            hx, hy = state.grid.hx, state.grid.hy

            filename = os.path.join(self.output_directory, f"data_{self.counter:05d}.nc")
            self.counter += 1

            with nc.Dataset(filename, mode="w") as ds:
                # time
                _ = ds.createDimension("t", 1)
                tv = ds.createVariable("t", float, ["t"])
                tv[...] = t

                # dimensions
                _ = ds.createDimension("x", nx + 1)
                _ = ds.createDimension("y", ny)

                # grid
                phi = ds.createVariable("phi", float_type, ["x", "y"])
                phi[...] = state.grid.phi[hx : hx + nx + 1, hy : hy + ny]
                theta = ds.createVariable("theta", float_type, ["x", "y"])
                theta[...] = state.grid.theta[hx : hx + nx + 1, hy : hy + ny]

                # variables
                h = ds.createVariable("h", float_type, ["x", "y"])
                h[...] = state.h[hx : hx + nx + 1, hy : hy + ny]
                u = ds.createVariable("u", float_type, ["x", "y"])
                u[...] = state.u[hx : hx + nx + 1, hy : hy + ny]
                v = ds.createVariable("v", float_type, ["x", "y"])
                v[...] = state.v[hx : hx + nx + 1, hy : hy + ny]
