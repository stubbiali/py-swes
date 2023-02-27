# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import TYPE_CHECKING

from swes.halo import update_halo_points_field
from swes.utils import zeros

if TYPE_CHECKING:
    from swes.grid import Grid


class Orography:
    def __init__(self, grid: Grid) -> None:
        self.hs = zeros(grid.ni, grid.nj)

        # non-flat terrain definition might happen here in the future

        update_halo_points_field(grid, self.hs)
