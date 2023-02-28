# -*- coding: utf-8 -*-
import math
from matplotlib import rcParams
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np

from swes.utils import get_time_string

color_levels_number = 19
colormap = "Greens"  # "Spectral"
field_name = "h"
figsize = (9, 6)
filename = "../output/williamson-1/data_24.nc"
filename_ref = "../output/williamson-1/data_0.nc"


EXTENDED_NAMES = {
    "h": "Fluid height [m]",
    "u": "Longitudinal velocity [m/s]",
    "v": "Latitudinal velocity [m/s]",
}


def main():
    rcParams["font.size"] = 16

    with nc.Dataset(filename_ref, mode="r") as ds:
        field = ds.variables[field_name][...]
        field_min = field.min()
        field_max = field.max()
        color_levels = np.linspace(field_min, field_max, color_levels_number)

    with nc.Dataset(filename, mode="r") as ds:
        t = ds.variables["t"][...]
        phi = ds.variables["phi"][...] * 180 / math.pi
        theta = ds.variables["theta"][...] * 180 / math.pi
        field = ds.variables[field_name][...]

        fig, ax = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
        mappable = ax.contourf(phi, theta, field, color_levels, cmap=colormap, extend="both")
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

        plt.show()


if __name__ == "__main__":
    main()
