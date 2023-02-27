# -*- coding: utf-8 -*-
from __future__ import annotations

import os.path
from enum import Enum
import math
from pydantic import Field, validator
from ruamel.yaml import YAML
from typing import Any, Literal, Optional

from swes.utils import BaseModel, float_type_


class PlanetConstants(BaseModel):
    # average radius [m]
    a: float_type_
    # gravity [m/s^2]
    g: float_type_
    # viscosity	[m2/s]
    nu: float_type_
    # rotation rate [Hz]
    omega: float_type_
    # atmosphere scale height [m]
    scale_height: float_type_


class UseCase(Enum):
    IDEALIZED_JET = "idealized_jet"
    WILLIAMSON_1 = "williamnson_1"
    WILLIAMSON_2 = "williamnson_2"
    WILLIAMSON_6 = "williamnson_6"


class Config(BaseModel):
    # number of grid points
    nx: int
    ny: int

    # number of halo points
    hx: int = 2
    hy: int = 1

    # planet
    planet: Literal["earth", "saturn"]
    planet_constants: PlanetConstants = Field(None)

    # initial conditions
    use_case: Literal["idealized_jet", "williamson_1", "williamson_2", "williamson_6"]
    init_args: dict[str, Any] = Field(default_factory=dict)
    initially_geostrophic: bool

    # solver settings
    enable_diffusion: bool

    # CFL number and integration time
    cfl: float_type_
    final_time: float_type_

    # output
    print_interval: float_type_ = 86400  # 1 day
    output_directory: Optional[str] = None

    # YAML file from which the object is initialized
    config_file: str = Field(None)

    @validator("planet_constants", pre=False)
    @classmethod
    def set_planet_constants(cls, v, values):
        # source: https://nssdc.gsfc.nasa.gov/planetary/factsheet/saturnfact.html
        if values["planet"] == "earth":
            return PlanetConstants(a=6.37122e6, g=9.80616, nu=5e5, omega=7.292e-5, scale_height=8e3)
        elif values["planet"] == "saturn":
            return PlanetConstants(
                a=5.8232e7,
                g=10.44,
                nu=5e6,
                omega=2.0 * math.pi / (10.656 * 3600.0),
                scale_height=60e3,
            )
        else:
            raise ValueError("Planet should be either 'earth' or 'saturn'.")

    @validator("cfl", pre=False)
    @classmethod
    def limit_cfl(cls, v):
        return min(1, max(0, v))

    @validator("output_directory", pre=False)
    @classmethod
    def set_output_directory(cls, v):
        if v is not None:
            if not os.path.exists(v) or not os.path.isdir(v) or not os.access(v, os.W_OK):
                print(
                    f"The directory {v} either does not exist or is not writeable. "
                    f"Disabling serialization."
                )
                return None
        return v

    @classmethod
    def from_yaml(cls, file: str, **kwargs) -> Config:
        with open(file) as f:
            raw_config = YAML(typ="safe").load(f)
            config = {**raw_config, "config_file": file, **kwargs}
            return cls(**config)
