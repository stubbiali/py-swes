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
from typing import TYPE_CHECKING

from swes.halo import update_halo_points_field
from swes.utils import zeros

if TYPE_CHECKING:
    import numpy as np

    from swes.grid import Grid


class Orography:
    hs: np.ndarray

    def __init__(self, grid: Grid) -> None:
        self.hs = zeros(grid.ni, grid.nj)

        # non-flat terrain definition might happen here in the future

        update_halo_points_field(grid, self.hs)
