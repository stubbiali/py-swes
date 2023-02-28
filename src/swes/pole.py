# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

from swes.utils import to_numpy

if TYPE_CHECKING:
    from typing import Optional, Union

    from swes.build_config import float_type
    from swes.config import Config
    from swes.state import State


class PoleTreatmentAdvectionOnly:
    dt_old: Optional[float_type]
    dxp: float
    h_north: float_type
    h_north_old: float_type
    h_south: float_type
    h_south_old: float_type
    m_north: float
    m_south: float

    def __init__(self, config: Config, state: State) -> None:
        # compute longitudinal increment
        self.dxp = (
            2
            * config.planet_constants.a
            * np.sin(state.grid.dtheta)
            / (1 + np.cos(state.grid.dtheta))
        )

        # compute map factor at the poles
        js, jn = state.grid.hy, state.grid.hy + state.grid.ny - 1
        self.m_north = 2 / (1 + np.sin(state.grid.thetac[jn]))
        self.m_south = 2 / (1 - np.sin(state.grid.thetac[js]))

        # set height at the poles
        ib, ie = state.grid.hx, state.grid.hx + state.grid.nx
        state.h[:, jn] = np.sum(state.h[ib:ie, jn]) / state.grid.nx
        state.h[:, js] = np.sum(state.h[ib:ie, js]) / state.grid.nx
        self.h_north = state.h[0, jn]
        self.h_south = state.h[0, js]

        # auxiliary variables
        self.dt_old = None
        self.h_north_old = self.h_north
        self.h_south_old = self.h_south

    def __call__(self, state: State, dt: float_type) -> None:
        if self.dt_old is None:
            self.dt_old = dt

        # shortcuts
        ib, ie = state.grid.hx, state.grid.hx + state.grid.nx
        js, jn = state.grid.hy, state.grid.hy + state.grid.ny - 1

        # north pole treatment
        h_north_new = self.h_north_old + (dt + self.dt_old) * 2 / (
            self.dxp * self.m_north * state.grid.nx
        ) * np.sum(state.h[ib:ie, jn - 1] * state.v[ib:ie, jn - 1])

        # south pole treatment
        h_south_new = self.h_south_old - (dt + self.dt_old) * 2 / (
            self.dxp * self.m_south * state.grid.nx
        ) * np.sum(state.h[ib:ie, js + 1] * state.v[ib:ie, js + 1])

        # set solution at the poles
        state.h[:, jn] = h_north_new
        state.h[:, js] = h_south_new

        # update auxiliary variables representing "old" solution
        self.dt_old = dt
        self.h_north_old = self.h_north
        self.h_south_old = self.h_south

        # update auxiliary variables representing "latest" solution
        self.h_north = h_north_new
        self.h_south = h_south_new


class PoleTreatment:
    dt_old: Optional[float_type]
    dxp: float
    g: float_type
    h_north: float
    h_north_old: float
    h_south: float
    h_south_old: float
    hu_north: float
    hu_north_old: float
    hu_south: float
    hu_south_old: float
    hv_north: float
    hv_north_old: float
    hv_south: float
    hv_south_old: float
    m_north: float
    m_south: float
    u_north: float
    u_south: float
    v_north: float
    v_south: float

    def __init__(self, config: Config, state: State) -> None:
        # compute longitudinal increment
        self.dxp = (
            2
            * config.planet_constants.a
            * np.sin(state.grid.dtheta)
            / (1 + np.cos(state.grid.dtheta))
        )

        # compute map factor at the poles
        self.m_north = 2 / (1 + np.sin(state.grid.thetac[-2]))
        self.m_south = 2 / (1 - np.sin(state.grid.thetac[1]))

        # set height at the poles
        ib, ie = state.grid.hx, state.grid.hx + state.grid.nx
        js, jn = state.grid.hy, state.grid.hy + state.grid.ny - 1
        state.h[:, jn] = np.sum(state.h[ib:ie, jn]) / state.grid.nx
        state.h[:, js] = np.sum(state.h[ib:ie, js]) / state.grid.nx
        self.h_north = state.h[0, jn]
        self.h_south = state.h[0, js]

        # compute stereographic components at north pole
        u, v = to_numpy(state.u), to_numpy(state.v)
        u_north = -u[ib:ie, jn] * np.sin(state.grid.phic[ib:ie]) - v[ib:ie, jn] * np.cos(
            state.grid.phic[ib:ie]
        )
        self.u_north = np.sum(u_north) / state.grid.nx
        v_north = u[ib:ie, jn] * np.cos(state.grid.phic[ib:ie]) - v[ib:ie, jn] * np.sin(
            state.grid.phic[ib:ie]
        )
        self.v_north = np.sum(v_north) / state.grid.nx

        # compute stereographic components at south pole
        # @ 01-09-2016: use the same formula as for north pole
        u_south = -u[ib:ie, js] * np.sin(state.grid.phic[ib:ie]) + v[ib:ie, js] * np.cos(
            state.grid.phic[ib:ie]
        )
        self.u_south = np.sum(u_south) / state.grid.nx
        v_south = u[ib:ie, js] * np.cos(state.grid.phic[ib:ie]) + v[ib:ie, js] * np.sin(
            state.grid.phic[ib:ie]
        )
        self.v_south = np.sum(v_south) / state.grid.nx

        # compute momentum at the poles
        self.hu_north = self.h_north * self.u_north
        self.hu_south = self.h_south * self.u_south
        self.hv_north = self.h_north * self.v_north
        self.hv_south = self.h_south * self.v_south

        # auxiliary variables
        self.dt_old = None
        self.h_north_old = self.h_north
        self.h_south_old = self.h_south
        self.hu_north_old = self.hu_north
        self.hu_south_old = self.hu_south
        self.hv_north_old = self.hv_north
        self.hv_south_old = self.hv_south

        # useful planet constants
        self.g = config.planet_constants.g

    def __call__(self, state: State, dt: float_type) -> None:
        if self.dt_old is None:
            self.dt_old = dt

        # shortcuts
        ib, ie = state.grid.hx, state.grid.hx + state.grid.nx
        js, jn = state.grid.hy, state.grid.hy + state.grid.ny - 1
        nx, phic = state.grid.nx, state.grid.phic
        h, u, v = state.h, state.u, state.v
        dt_old, dxp, m_north, m_south = self.dt_old, self.dxp, self.m_north, self.m_south

        # north pole treatment
        h_north_new = self.h_north_old + (dt + dt_old) * 2 / (dxp * m_north * nx) * np.sum(
            h[ib:ie, jn - 1] * v[ib:ie, jn - 1]
        )
        au = (
            -2
            / (dxp * m_north * nx)
            * np.sum(
                h[ib:ie, jn - 1]
                * (u[ib:ie, jn - 1] * np.sin(phic[ib:ie]) + v[ib:ie, jn - 1] * np.cos(phic[ib:ie]))
                * v[ib:ie, jn - 1]
            )
        )
        bu = -self.g / (dxp * nx) * np.sum((h[ib:ie, jn - 1] ** 2) * np.cos(phic[ib:ie]))
        av = (
            -2
            / (dxp * m_north * nx)
            * np.sum(
                h[ib:ie, jn - 1]
                * (-u[ib:ie, jn - 1] * np.cos(phic[ib:ie]) + v[ib:ie, jn - 1] * np.sin(phic[ib:ie]))
                * v[ib:ie, jn - 1]
            )
        )
        bv = -self.g / (dxp * nx) * np.sum((h[ib:ie, jn - 1] ** 2) * np.sin(phic[ib:ie]))
        fp = state.grid.f[0, jn]
        hu_north_new = (1 / (1 + 0.25 * (dt_old + dt) ** 2) * (fp**2)) * (
            (1 - 0.25 * ((dt_old + dt) ** 2) * (fp**2)) * self.hu_north_old
            + (dt_old + dt) * (au + bu)
            + (dt_old + dt) * fp * self.hv_north_old
            + 0.5 * ((dt_old + dt) ** 2) * fp * (av + bv)
        )
        u_north_new = hu_north_new / h_north_new
        hv_north_new = (
            self.hv_north_old
            + (dt_old + dt) * (av + bv)
            - 0.5 * (dt_old + dt) * fp * (self.hu_north_old + hu_north_new)
        )
        v_north_new = hv_north_new / h_north_new

        # south pole treatment
        h_south_new = self.h_south_old - (dt + dt_old) * 2 / (dxp * m_south * nx) * np.sum(
            h[ib:ie, js + 1] * v[ib:ie, js + 1]
        )
        au = (
            -2
            / (dxp * m_south * nx)
            * np.sum(
                h[ib:ie, js + 1]
                * (-u[ib:ie, js + 1] * np.sin(phic[ib:ie]) + v[ib:ie, js + 1] * np.cos(phic[ib:ie]))
                * v[ib:ie, js + 1]
            )
        )
        bu = -self.g / (dxp * nx) * np.sum((h[ib:ie, js + 1] ** 2) * np.cos(phic[ib:ie]))
        av = (
            -2
            / (dxp * m_south * nx)
            * np.sum(
                h[ib:ie, js + 1]
                * (u[ib:ie, js + 1] * np.cos(phic[ib:ie]) + v[ib:ie, js + 1] * np.sin(phic[ib:ie]))
                * v[ib:ie, js + 1]
            )
        )
        bv = -self.g / (dxp * nx) * np.sum((h[ib:ie, js + 1] ** 2) * np.sin(phic[ib:ie]))
        fp = state.grid.f[0, js]
        hu_south_new = (
            1
            / (1 + 0.25 * ((dt_old + dt) ** 2) * (fp**2))
            * (
                (1 - 0.25 * ((dt_old + dt) ** 2) * (fp**2)) * self.hu_south_old
                + (dt_old + dt) * (au + bu)
                - (dt_old + dt) * fp * self.hv_south_old
                - 0.5 * ((dt_old + dt) ** 2) * fp * (av + bv)
            )
        )
        u_south_new = hu_south_new / h_south_new
        hv_south_new = (
            self.hv_south_old
            + (dt_old + dt) * (av + bv)
            + 0.5 * (dt_old + dt) * fp * (self.hu_south_old + hu_south_new)
        )
        v_south_new = hv_south_new / h_south_new

        # set solution at the poles
        h[:, jn] = h_north_new
        h[:, js] = h_south_new
        u[:, jn] = -u_north_new * np.sin(phic) + v_north_new * np.cos(phic)
        u[:, js] = -u_south_new * np.sin(phic) + v_south_new * np.cos(phic)
        v[:, jn] = -u_north_new * np.cos(phic) - v_north_new * np.sin(phic)
        v[:, js] = u_south_new * np.cos(phic) + v_south_new * np.sin(phic)

        # update auxiliary variables representing "old" solution
        self.dt_old = dt
        self.h_north_old = self.h_north
        self.h_south_old = self.h_south
        self.hu_north_old = self.hu_north
        self.hu_south_old = self.hu_south
        self.hv_north_old = self.hv_north
        self.hv_south_old = self.hv_south

        # update auxiliary variables representing "latest" solution
        self.h_north = h_north_new
        self.h_south = h_south_new
        self.hu_north = hu_north_new
        self.hu_south = hu_south_new
        self.hv_north = hv_north_new
        self.hv_south = hv_south_new


def get_pole_treatment(
    config: Config, state: State
) -> Union[PoleTreatment, PoleTreatmentAdvectionOnly]:
    if config.advection_only:
        return PoleTreatmentAdvectionOnly(config, state)
    else:
        return PoleTreatment(config, state)
