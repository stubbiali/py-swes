# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import TYPE_CHECKING

from swes.build_config import float_type
from swes.utils import GT_FIELD, stencil, zeros

if TYPE_CHECKING:
    from swes.config import Config
    from swes.grid import Grid
    from swes.orography import Orography
    from swes.state import State


@stencil
def advection(
    in_c: GT_FIELD,
    in_c_y: GT_FIELD,
    in_dx: GT_FIELD,
    in_dxc: GT_FIELD,
    in_dy1: GT_FIELD,
    in_dy1c: GT_FIELD,
    in_h: GT_FIELD,
    in_u: GT_FIELD,
    in_u_x: GT_FIELD,
    in_v: GT_FIELD,
    in_v_y: GT_FIELD,
    out_h: GT_FIELD,
    *,
    dt: float_type,
):
    with computation(FORWARD), interval(...):
        h_x = 0.5 * (in_h[0, 0] + in_h[1, 0]) - 0.5 * dt / in_dx[0, 0] * (
            in_h[1, 0] * in_u[1, 0] - in_h[0, 0] * in_u[0, 0]
        )
        h_y = 0.5 * (in_h[0, 0] + in_h[0, 1]) - 0.5 * dt / in_dy1[0, 0] * (
            in_h[0, 1] * in_v[0, 1] * in_c[0, 1] - in_h[0, 0] * in_v[0, 0] * in_c[0, 0]
        )
        out_h[0, 0] = (
            in_h[0, 0]
            - dt / in_dxc[0, 0] * (h_x[0, 0, 0] * in_u_x[0, 0] - h_x[-1, 0, 0] * in_u_x[-1, 0])
            - dt
            / in_dy1c[0, 0]
            * (
                h_y[0, 0, 0] * in_v_y[0, 0] * in_c_y[0, 0]
                - h_y[0, -1, 0] * in_v_y[0, -1] * in_c_y[0, -1]
            )
        )


class LaxWendroffAdvectionOnly:
    def __init__(self, grid: Grid) -> None:
        # "centred" Cartesian increments
        self.dxc = zeros(grid.ni, grid.nj)
        self.dxc[1:-1, :] = 0.5 * (grid.dx[:-2, :] + grid.dx[1:-1, :])
        self.dy1c = zeros(grid.ni, grid.nj)
        self.dy1c[:, 1:-1] = 0.5 * (grid.dy1[:, :-2] + grid.dy1[:, 1:-1])

    def __call__(self, state: State, dt: float_type) -> None:
        advection(
            in_c=state.grid.c,
            in_c_y=state.grid.c_y,
            in_dx=state.grid.dx,
            in_dxc=self.dxc,
            in_dy1=state.grid.dy1,
            in_dy1c=self.dy1c,
            in_h=state.h,
            in_u=state.u,
            in_u_x=state.u_x,
            in_v=state.v,
            in_v_y=state.v_y,
            out_h=state.h_new,
            dt=dt,
            origin=(state.grid.hx, state.grid.hy + 1, 0),
            domain=(state.grid.nx, state.grid.ny - 2, 1),
        )


@stencil
def lax_wendroff(
    in_c: GT_FIELD,
    in_c_y: GT_FIELD,
    in_dx: GT_FIELD,
    in_dxc: GT_FIELD,
    in_dy: GT_FIELD,
    in_dyc: GT_FIELD,
    in_dy1: GT_FIELD,
    in_dy1c: GT_FIELD,
    in_f: GT_FIELD,
    in_h: GT_FIELD,
    in_hs: GT_FIELD,
    in_tg: GT_FIELD,
    in_tg_x: GT_FIELD,
    in_tg_y: GT_FIELD,
    in_u: GT_FIELD,
    in_v: GT_FIELD,
    out_h: GT_FIELD,
    out_u: GT_FIELD,
    out_v: GT_FIELD,
    *,
    dt: float_type,
    a: float_type,
    g: float_type,
):
    with computation(FORWARD), interval(...):
        # auxiliary variables
        v1 = in_v[0, 0] * in_c[0, 0]
        hu = in_h[0, 0] * in_u[0, 0]
        hv = in_h[0, 0] * in_v[0, 0]
        hv1 = in_h[0, 0] * v1

        # longitudinal mid-points
        h_x = 0.5 * (in_h[0, 0] + in_h[1, 0]) - 0.5 * dt / in_dx[0, 0] * (hu[1, 0, 0] - hu[0, 0, 0])
        hu_x = (
            0.5 * (hu[0, 0, 0] + hu[1, 0, 0])
            - 0.5
            * dt
            / in_dx[0, 0]
            * (
                (hu[1, 0, 0] * in_u[1, 0] + 0.5 * g * in_h[1, 0] ** 2)
                - (hu[0, 0, 0] * in_u[0, 0] + 0.5 * g * in_h[0, 0] ** 2)
            )
            + 0.5
            * dt
            * (
                0.5 * (in_f[0, 0] + in_f[1, 0])
                + 0.5 * (in_u[0, 0] + in_u[1, 0]) * in_tg_x[0, 0] / a
            )
            * 0.5
            * (hv[0, 0, 0] + hv[1, 0, 0])
        )
        hv_x = (
            0.5 * (hv[0, 0, 0] + hv[1, 0, 0])
            - 0.5 * dt / in_dx[0, 0] * (hu[1, 0, 0] * in_v[1, 0] - hu[0, 0, 0] * in_v[0, 0])
            - 0.5
            * dt
            * (
                0.5 * (in_f[0, 0] + in_f[1, 0])
                + 0.5 * (in_u[0, 0] + in_u[1, 0]) * in_tg_x[0, 0] / a
            )
            * 0.5
            * (hu[0, 0, 0] + hu[1, 0, 0])
        )

        # latitudinal mid-points
        h_y = 0.5 * (in_h[0, 0] + in_h[0, 1]) - 0.5 * dt / in_dy1[0, 0] * (
            hv1[0, 1, 0] - hv1[0, 0, 0]
        )
        hu_y = (
            0.5 * (hu[0, 0, 0] + hu[0, 1, 0])
            - 0.5 * dt / in_dy1[0, 0] * (hu[0, 1, 0] * v1[0, 1, 0] - hu[0, 0, 0] * v1[0, 0, 0])
            + 0.5
            * dt
            * (
                0.5 * (in_f[0, 0] + in_f[0, 1])
                + 0.5 * (in_u[0, 0] + in_u[0, 1]) * in_tg_y[0, 0] / a
            )
            * (0.5 * (hv[0, 0, 0] + hv[0, 1, 0]))
        )
        hv_y = (
            0.5 * (hv[0, 0, 0] + hv[0, 1, 0])
            - 0.5 * dt / in_dy1[0, 0] * (hv[0, 1, 0] * v1[0, 1, 0] - hv[0, 0, 0] * v1[0, 0, 0])
            - 0.5 * dt / in_dy[0, 0] * (0.5 * g * in_h[0, 1] ** 2 - 0.5 * g * in_h[0, 0] ** 2)
            - 0.5
            * dt
            * (
                0.5 * (in_f[0, 0] + in_f[0, 1])
                + 0.5 * (in_u[0, 0] + in_u[0, 1]) * in_tg_y[0, 0] / a
            )
            * 0.5
            * (hu[0, 0, 0] + hu[0, 1, 0])
        )

        # advance solution
        out_h[0, 0] = (
            in_h[0, 0]
            - dt / in_dxc[0, 0] * (hu_x[0, 0, 0] - hu_x[-1, 0, 0])
            - dt / in_dy1c[0, 0] * (hv_y[0, 0, 0] * in_c_y[0, 0] - hv_y[0, -1, 0] * in_c_y[0, -1])
        )
        hu_new = (
            hu[0, 0, 0]
            - dt
            / in_dxc[0, 0]
            * (
                (hu_x[0, 0, 0] ** 2 / h_x[0, 0, 0] + 0.5 * g * h_x[0, 0, 0] ** 2)
                - (hu_x[-1, 0, 0] ** 2 / h_x[-1, 0, 0] + 0.5 * g * h_x[-1, 0, 0] ** 2)
            )
            - dt
            / in_dy1c[0, 0]
            * (
                hv_y[0, 0, 0] * in_c_y[0, 0] * hu_y[0, 0, 0] / h_y[0, 0, 0]
                - hv_y[0, -1, 0] * in_c_y[0, -1] * hu_y[0, -1, 0] / h_y[0, -1, 0]
            )
            + dt
            * (
                in_f[0, 0]
                + 0.25
                * (
                    hu_x[-1, 0, 0] / h_x[-1, 0, 0]
                    + hu_x[0, 0, 0] / h_x[0, 0, 0]
                    + hu_y[0, -1, 0] / h_y[0, -1, 0]
                    + hu_y[0, 0, 0] / h_y[0, 0, 0]
                )
                * in_tg[0, 0]
                / a
            )
            * 0.25
            * (hv_x[0, 0, 0] + hv_x[-1, 0, 0] + hv_y[0, 0, 0] + hv_y[0, -1, 0])
            - dt
            * g
            * 0.25
            * (h_x[-1, 0, 0] + h_x[0, 0, 0] + h_y[0, -1, 0] + h_y[0, 0, 0])
            * (in_hs[1, 0] - in_hs[-1, 0])
            / (in_dx[-1, 0] + in_dx[0, 0])
        )
        hv_new = (
            hv[0, 0, 0]
            - dt
            / in_dxc[0, 0]
            * (
                hv_x[0, 0, 0] * hu_x[0, 0, 0] / h_x[0, 0, 0]
                - hv_x[-1, 0, 0] * hu_x[-1, 0, 0] / h_x[-1, 0, 0]
            )
            - dt
            / in_dy1c[0, 0]
            * (
                hv_y[0, 0, 0] ** 2 / h_y[0, 0, 0] * in_c_y[0, 0]
                - hv_y[0, -1, 0] ** 2 / h_y[0, -1, 0] * in_c_y[0, -1]
            )
            - dt / in_dyc[0, 0] * (0.5 * g * h_y[0, 0, 0] ** 2 - 0.5 * g * h_y[0, -1, 0] ** 2)
            - dt
            * (
                in_f[0, 0]
                + 0.25
                * (
                    hu_x[-1, 0, 0] / h_x[-1, 0, 0]
                    + hu_x[0, 0, 0] / h_x[0, 0, 0]
                    + hu_y[0, -1, 0] / h_y[0, -1, 0]
                    + hu_y[0, 0, 0] / h_y[0, 0, 0]
                )
                * in_tg[0, 0]
                / a
            )
            * 0.25
            * (hu_x[-1, 0, 0] + hu_x[0, 0, 0] + hu_y[0, -1, 0] + hu_y[0, 0, 0])
            - dt
            * g
            * 0.25
            * (h_x[-1, 0, 0] + h_x[0, 0, 0] + h_y[0, -1, 0] + h_y[0, 0, 0])
            * (in_hs[0, 1] - in_hs[0, -1])
            / (in_dy1[0, 0] + in_dy1[0, -1])
        )

        # retrieve velocity components
        out_u[0, 0] = hu_new[0, 0, 0] / out_h[0, 0]
        out_v[0, 0] = hv_new[0, 0, 0] / out_h[0, 0]


class LaxWendroff:
    def __init__(self, config: Config, grid: Grid, orography: Orography) -> None:
        # "centred" Cartesian increments
        self.dxc = zeros(grid.ni, grid.nj)
        self.dxc[1:-1, :] = 0.5 * (grid.dx[:-2, :] + grid.dx[1:-1, :])
        self.dyc = zeros(grid.ni, grid.nj)
        self.dyc[:, 1:-1] = 0.5 * (grid.dy[:, :-2] + grid.dy[:, 1:-1])
        self.dy1c = zeros(grid.ni, grid.nj)
        self.dy1c[:, 1:-1] = 0.5 * (grid.dy1[:, :-2] + grid.dy1[:, 1:-1])

        # terrain height
        self.hs = orography.hs

        # useful planet constants
        self.a, self.g = config.planet_constants.a, config.planet_constants.g

    def __call__(self, state: State, dt: float_type) -> None:
        lax_wendroff(
            in_c=state.grid.c,
            in_c_y=state.grid.c_y,
            in_dx=state.grid.dx,
            in_dxc=self.dxc,
            in_dy=state.grid.dy,
            in_dyc=self.dyc,
            in_dy1=state.grid.dy1,
            in_dy1c=self.dy1c,
            in_f=state.grid.f,
            in_h=state.h,
            in_hs=self.hs,
            in_tg=state.grid.tg,
            in_tg_x=state.grid.tg_x,
            in_tg_y=state.grid.tg_y,
            in_u=state.u,
            in_v=state.v,
            out_h=state.h_new,
            out_u=state.u_new,
            out_v=state.v_new,
            dt=dt,
            a=self.a,
            g=self.g,
            origin=(state.grid.hx, state.grid.hy + 1, 0),
            domain=(state.grid.nx, state.grid.ny - 2, 1),
        )


def get_solver(config: Config, grid: Grid, orography: Orography):
    if config.advection_only:
        return LaxWendroffAdvectionOnly(grid)
    else:
        return LaxWendroff(config, grid, orography)
