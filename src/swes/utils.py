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
from gt4py.cartesian import gtscript
from gt4py.storage import zeros as gt_zeros
import pydantic
from typing import TYPE_CHECKING

from swes.build_config import backend, float_type, int_type

if TYPE_CHECKING:
    import numpy as np


GT_FIELD = gtscript.Field[gtscript.IJ, float_type]


def stencil(definition=None, **kwargs):
    def core(_definition):
        extra_args = {"dtypes": {float: float_type, int: int_type}, **kwargs}
        if backend != "numpy":
            extra_args["verbose"] = True
        return gtscript.stencil(backend, _definition, **extra_args)

    return core(definition) if definition is not None else core


def zeros(ni: int, nj: int) -> np.ndarray:
    return gt_zeros((ni, nj), float_type, backend=backend, aligned_index=(0, 0))


def to_gt4py(buffer: np.ndarray) -> np.ndarray:
    out = zeros(*buffer.shape)
    out[...] = buffer
    return out


def to_numpy(buffer) -> np.ndarray:
    try:
        # cupy arrays
        return buffer.get()
    except AttributeError:
        return buffer


def get_time_string(seconds: float_type) -> str:
    days = int(seconds // 86400)
    seconds -= days * 86400
    hours = int(seconds // 3600)
    seconds -= hours * 3600
    minutes = int(seconds // 60)
    seconds -= minutes * 60
    seconds = int(seconds)
    return f"{days:3d}d {hours:2d}h {minutes:2d}m {seconds:2d}s"


class float_type_(float_type):
    """Pydantic validator for `float_type`."""

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        return float_type(v)


class BaseModel(pydantic.BaseModel):
    """Custom BaseModel."""

    class Config:
        arbitrary_types_allowed = True
        extra = pydantic.Extra.allow
        validate_all = True
