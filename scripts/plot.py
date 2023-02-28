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
from matplotlib import rcParams
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
from typing import TYPE_CHECKING

from swes.utils import get_time_string

if TYPE_CHECKING:
    from typing import Literal, Optional


EXTENDED_NAMES = {
    "h": "Fluid height [m]",
    "u": "Longitudinal velocity [m/s]",
    "v": "Latitudinal velocity [m/s]",
}


def main(
    field_name: Literal["h", "u", "v"],
    dataset_name: str,
    dataset_ref_name: Optional[str] = None,
    *,
    ax: Optional[plt.Axes] = None,
    draw_colorbar: bool = True,
    color_map: str = "Spectral",
    figsize: tuple[int, int] = (9, 6),
    num_color_levels: int = 19,
) -> plt.Axes:
    rcParams["font.size"] = 16

    dataset_ref_name = dataset_ref_name or dataset_name
    with nc.Dataset(dataset_ref_name, mode="r") as ds:
        field = ds.variables[field_name][...]
        field_min = field.min()
        field_max = field.max()
        color_levels = np.linspace(field_min, field_max, num_color_levels)

    with nc.Dataset(dataset_name, mode="r") as ds:
        t = ds.variables["t"][...]
        phi = ds.variables["phi"][...] * 180 / math.pi
        theta = ds.variables["theta"][...] * 180 / math.pi
        field = ds.variables[field_name][...]

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
    else:
        ax.cla()
    mappable = ax.contourf(phi, theta, field, color_levels, cmap=color_map, extend="both")
    if draw_colorbar:
        plt.colorbar(mappable, extend="both")

    ax.set_title(
        f"{EXTENDED_NAMES[field_name]} @ {get_time_string(t)}",
        loc="center",
        fontsize=rcParams["font.size"],
    )
    ax.set_xlim([0, 360])
    ax.get_xaxis().set_ticks([0, 60, 120, 180, 240, 300, 360])
    ax.get_xaxis().set_ticklabels(["0°", "60°E", "120°E", "180°", "120°W", "60°W", "0°"])
    ax.set_ylim([-90, 90])
    ax.get_yaxis().set_ticks([-90, -60, -30, 0, 30, 60, 90])
    ax.get_yaxis().set_ticklabels(["90°S", "60°S", "30°S", "0°", "30°N", "60°N", "90°N"])
    ax.grid(True, linestyle=":")

    return ax


if __name__ == "__main__":
    main(
        field_name="h",
        dataset_name="../output/williamson-1/data_0.nc",
        dataset_ref_name="../output/williamson-1/data_0.nc",
        color_map="Reds",
    )
    plt.show()
