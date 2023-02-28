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
import numpy as np
from typing import TYPE_CHECKING

from swes.halo import update_halo_points
from swes.pole import get_pole_treatment
from swes.state import State
from swes.utils import to_numpy

if TYPE_CHECKING:
    from swes.config import Config
    from swes.grid import Grid
    from swes.orography import Orography


def idealized_jet(config: Config, state: State, hs: np.ndarray) -> None:
    nx, ny = state.grid.nx, state.grid.ny

    # draw u from a normal
    u = np.random.normal(30 * math.pi / 180, 10 * math.pi / 180, (nx + 3, ny))
    u[...] *= 100 / state.u.max()

    # set v to rest
    state.v[...] = 0

    # prevent wind from flowing too fast
    wmax = 2000
    u[...] = np.where(np.abs(u) > wmax, wmax * np.sign(u), u)
    state.u[...] = u

    # exploit geostrophic balance to retrieve the fluid height
    # approximate the derivative through forward Euler
    f, hs = to_numpy(state.grid.f), to_numpy(hs)
    ht = np.zeros((state.grid.ni, state.grid.nj))
    ht[:, 0] = config.planet_constants.scale_height
    for j in range(state.grid.ny - 1):
        ht[:, j + 1] = (
            ht[:, j]
            - 0.5
            * (f[:, j + 1] + f[:, j])
            * 0.5
            * (u[:, j + 1] + u[:, j])
            * (config.planet_constants.a + 0.5 * (hs[:, j + 1] + hs[:, j]))
            / config.planet_constants.g
            * state.grid.dtheta
        )
    state.h[...] = ht - hs


def williamson_1(config: Config, state: State) -> None:
    # extract geometric fields
    phi = to_numpy(state.grid.phi)
    theta = to_numpy(state.grid.theta)

    # define coefficients
    alpha = config.init_args.get("alpha", 0)
    u0 = 2 * math.pi * config.planet_constants.a / (12 * 24 * 3600)
    h0 = 1000
    phi_c = 1.5 * math.pi
    theta_c = 0
    rmax = config.planet_constants.a / 3

    # compute advecting wind
    state.u[...] = u0 * (
        np.cos(theta) * np.cos(alpha) + np.sin(theta) * np.cos(phi) * np.sin(alpha)
    )
    state.u_new[...] = state.u
    state.u_x[:-1, :] = u0 * (
        np.cos(0.5 * (theta[:-1, :] + theta[1:, :])) * np.cos(alpha)
        + np.sin(0.5 * (theta[:-1, :] + theta[1:, :]))
        * np.cos(0.5 * (phi[:-1, :] + phi[1:, :]))
        * np.sin(alpha)
    )
    state.u_y[:, :-1] = u0 * (
        np.cos(0.5 * (theta[:, :-1] + theta[:, 1:])) * np.cos(alpha)
        + np.sin(0.5 * (theta[:, :-1] + theta[:, 1:]))
        * np.cos(0.5 * (phi[:, :-1] + phi[:, 1:]))
        * np.sin(alpha)
    )

    state.v[...] = -u0 * np.sin(phi) * np.sin(alpha)
    state.v_new[...] = state.v
    state.v_x[:-1, :] = -u0 * np.sin(0.5 * (phi[:-1, :] + phi[1:, :])) * np.sin(alpha)
    state.v_y[:, :-1] = -u0 * np.sin(0.5 * (phi[:, :-1] + phi[:, 1:])) * np.sin(alpha)

    # compute initial fluid height
    r = config.planet_constants.a * np.arccos(
        np.sin(theta_c) * np.sin(theta) + np.cos(theta_c) * np.cos(theta) * np.cos(phi - phi_c)
    )
    state.h[...] = np.where(r < rmax, 0.5 * h0 * (1 + np.cos(math.pi * r / rmax)), 0)


def williamson_2(config: Config, state: State) -> None:
    # extract geometric fields
    phi = to_numpy(state.grid.phi)
    theta = to_numpy(state.grid.theta)

    # define constants
    alpha = config.init_args.get("alpha", 0)
    h0 = 2.94e4 / config.planet_constants.g
    u0 = 2 * math.pi * config.planet_constants.a / (12 * 24 * 3600)

    # make Coriolis parameter dependent on longitude and latitude
    state.grid.f[...] = (
        2
        * config.planet_constants.omega
        * (-np.cos(phi) * np.cos(theta) * np.sin(alpha) + np.sin(theta) * np.cos(alpha))
    )

    # compute initial height
    state.h[...] = h0 - (
        config.planet_constants.a * config.planet_constants.omega * u0 + 0.5 * (u0**2)
    ) * ((-np.cos(phi) * np.cos(theta) * np.sin(alpha) + np.sin(theta) * np.cos(alpha)) ** 2)

    # compute initial wind
    state.u[...] = u0 * (
        np.cos(theta) * np.cos(alpha) + np.cos(phi) * np.sin(theta) * np.sin(alpha)
    )
    state.v[...] = -u0 * np.sin(phi) * np.sin(alpha)


def williamson_6(config: Config, state: State) -> None:
    # extract geometric fields
    phi = to_numpy(state.grid.phi)
    theta = to_numpy(state.grid.theta)

    # constants
    a, g = config.planet_constants.a, config.planet_constants.g
    w = 7.848e-6
    k = 7.848e-6
    h0 = 8e3
    r = 4.0

    # Compute initial fluid height
    aa = 0.5 * w * (2 * config.planet_constants.omega + w) * (
        (np.cos(theta) ** 2)
    ) + 0.25 * k**2 * (np.cos(theta) ** (2 * r)) * (
        (r + 1) * (np.cos(theta) ** 2) + (2 * r**2 - r - 2) - 2 * r**2 * (np.cos(theta) ** (-2))
    )
    bb = (
        (2 * (config.planet_constants.omega + w) * k)
        / ((r + 1) * (r + 2))
        * (np.cos(theta) ** r)
        * ((r**2 + 2 * r + 2) - ((r + 1) ** 2) * (np.cos(theta) ** 2))
    )
    cc = 0.25 * k**2 * (np.cos(theta) ** (2 * r)) * ((r + 1) * (np.cos(theta) ** 2) - (r + 2))
    state.h[...] = (
        h0 + (a**2 * aa + a**2 * bb * np.cos(r * phi) + a**2 * cc * np.cos(2 * r * phi)) / g
    )

    # compute initial wind
    state.u[...] = a * w * np.cos(theta) + a * k * (np.cos(theta) ** (r - 1)) * (
        r * (np.sin(theta) ** 2) - (np.cos(theta) ** 2)
    ) * np.cos(r * phi)
    state.v[...] = -a * k * r * (np.cos(theta) ** (r - 1)) * np.sin(theta) * np.sin(r * phi)


def get_initial_state(
    config: Config, grid: Grid, orography: Orography
) -> tuple[State, PoleTreatment]:
    state = State(grid)

    if config.use_case == "idealized_jet":
        idealized_jet(config, state, orography.hs)
    elif config.use_case == "williamson_1":
        williamson_1(config, state)
    elif config.use_case == "williamson_2":
        williamson_2(config, state)
    elif config.use_case == "williamson_6":
        williamson_6(config, state)

    if config.initially_geostrophic:
        ht = orography.hs + state.h

        # \[ d(ht) = -dfrac{F u (R + hs)}{g} d\theta \]
        # approximate the derivative through centred finite difference
        state.u[:, 1:-1] = (
            -0.5
            * config.planet_constants.g
            * (ht[:, 2:] - ht[:, :-2])
            / (
                (config.planet_constants.a + orography.hs[:, 1:-1])
                * state.grid.dtheta
                * state.grid.f[:, 1:-1]
            )
        )

        # \[ d(ht) = dfrac{F v (R + hs) \cos(\theta)}{g} d\phi \]
        # approximate the derivative through centred finite difference
        state.v[1:-1, :] = (
            0.5
            * config.planet_constants.g
            * (ht[2:, :] - ht[:-2, :])
            / (
                (config.planet_constants.a + orography.hs[1:-1, :])
                * state.grid.dphi
                * state.grid.c[1:-1, :]
                * state.grid.f[1:-1, :]
            )
        )

        # zero latitudinal derivative at the equator (not sure...)
        j = grid.nj // 2
        state.u[:, j] = state.u[:, j + 1]
        state.v[:, j] = state.v[:, j + 1]

        if config.use_case == "idealized_jet":
            # prevent wind from flowing too fast
            u, v = to_numpy(state.u), to_numpy(state.v)
            wmax = 2000
            state.u[...] = np.where(np.abs(u) > wmax, wmax * np.sign(u), u)
            state.v[...] = np.where(np.abs(v) > wmax, wmax * np.sign(v), v)

    pole_treatment = get_pole_treatment(config, state)
    update_halo_points(state)

    return state, pole_treatment
