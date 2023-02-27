# -*- coding: utf-8 -*-
from __future__ import annotations
from gt4py.cartesian import gtscript
from typing import TYPE_CHECKING

from swes.build_config import float_type
from swes.utils import GT_FIELD, stencil, zeros

if TYPE_CHECKING:
    from swes.config import Config
    from swes.grid import Grid
    from swes.state import State


@gtscript.function
def diffusion_kernel(
    ax: GT_FIELD,
    ay: GT_FIELD,
    bx: GT_FIELD,
    by: GT_FIELD,
    cx: GT_FIELD,
    cy: GT_FIELD,
    phi: GT_FIELD,
    phi_tmp: GT_FIELD,
    *,
    dt: float_type,
    nu: float_type,
):
    return phi_tmp[0, 0] + dt * nu * (
        (
            ax[0, 0] * (ax[0, 0] * phi[0, 0] + bx[0, 0] * phi[1, 0] + cx[0, 0] * phi[-1, 0])
            + bx[0, 0] * (ax[1, 0] * phi[1, 0] + bx[1, 0] * phi[2, 0] + cx[1, 0] * phi[0, 0])
            + cx[0, 0] * (ax[-1, 0] * phi[-1, 0] + bx[-1, 0] * phi[0, 0] + cx[-1, 0] * phi[-2, 0])
            + ay[0, 0] * (ay[0, 0] * phi[0, 0] + by[0, 0] * phi[0, 1] + cy[0, 0] * phi[0, -1])
            + by[0, 0] * (ay[0, 1] * phi[0, 1] + by[0, 1] * phi[0, 2] + cy[0, 1] * phi[0, 0])
            + cy[0, 0] * (ay[0, -1] * phi[0, -1] + by[0, -1] * phi[0, 0] + cy[0, -1] * phi[0, -2])
        )
    )


@stencil
def diffusion(
    in_ax: GT_FIELD,
    in_ay: GT_FIELD,
    in_bx: GT_FIELD,
    in_by: GT_FIELD,
    in_cx: GT_FIELD,
    in_cy: GT_FIELD,
    in_h: GT_FIELD,
    in_u: GT_FIELD,
    in_v: GT_FIELD,
    inout_h: GT_FIELD,
    inout_u: GT_FIELD,
    inout_v: GT_FIELD,
    *,
    dt: float_type,
    nu: float_type,
):
    with computation(FORWARD), interval(...):
        inout_h[0, 0] = diffusion_kernel(
            in_ax, in_ay, in_bx, in_by, in_cx, in_cy, in_h, inout_h, dt=dt, nu=nu
        )
        inout_u[0, 0] = diffusion_kernel(
            in_ax, in_ay, in_bx, in_by, in_cx, in_cy, in_u, inout_u, dt=dt, nu=nu
        )
        inout_v[0, 0] = diffusion_kernel(
            in_ax, in_ay, in_bx, in_by, in_cx, in_cy, in_v, inout_v, dt=dt, nu=nu
        )


class Diffusion:
    def __init__(self, config: Config, grid: Grid) -> None:
        if grid.hx < 2 or grid.hy < 1:
            raise RuntimeError(
                f"Diffusion requires hx >= 2 (got {config.hx}) and hy >= 1 (got {config.hy})."
            )
        self.nu = config.planet_constants.nu

        # pre-compute coefficients for centred finite difference along longitude
        # ax, bx and cx denote the coefficients associated with the centred, upwind and downwind point
        self.ax = zeros(grid.ni, grid.nj)
        self.ax[1:-1, 1:-1] = (grid.dx[1:-1, 1:-1] - grid.dx[:-2, 1:-1]) / (
            grid.dx[:-2, 1:-1] * grid.dx[1:-1, 1:-1]
        )
        self.bx = zeros(grid.ni, grid.nj)
        self.bx[1:-1, 1:-1] = grid.dx[:-2, 1:-1] / (
            grid.dx[1:-1, 1:-1] * (grid.dx[:-2, 1:-1] + grid.dx[1:-1, 1:-1])
        )
        self.cx = zeros(grid.ni, grid.nj)
        self.cx[1:-1, 1:-1] = -grid.dx[1:-1, 1:-1] / (
            grid.dx[:-2, 1:-1] * (grid.dx[:-2, 1:-1] + grid.dx[1:-1, 1:-1])
        )

        # pre-compute coefficients for centred finite difference along latitude
        # ay, by and cy denote the coefficients associated with the centred, upwind and downwind point
        self.ay = zeros(grid.ni, grid.nj)
        self.ay[1:-1, 1:-1] = (grid.dy[1:-1, 1:-1] - grid.dy[1:-1, :-2]) / (
            grid.dy[1:-1, :-2] * grid.dy[1:-1, 1:-1]
        )
        self.by = zeros(grid.ni, grid.nj)
        self.by[1:-1, 1:-1] = grid.dy[1:-1, :-2] / (
            grid.dy[1:-1, 1:-1] * (grid.dy[1:-1, :-2] + grid.dy[1:-1, 1:-1])
        )
        self.cy = zeros(grid.ni, grid.nj)
        self.cy[1:-1, 1:-1] = -grid.dy[1:-1, 1:-1] / (
            grid.dy[1:-1, :-2] * (grid.dy[1:-1, :-2] + grid.dy[1:-1, 1:-1])
        )

    def __call__(self, state: State, dt: float_type) -> None:
        diffusion(
            in_ax=self.ax,
            in_ay=self.ay,
            in_bx=self.bx,
            in_by=self.by,
            in_cx=self.cx,
            in_cy=self.cy,
            in_h=state.h,
            in_u=state.u,
            in_v=state.v,
            inout_h=state.h_new,
            inout_u=state.u_new,
            inout_v=state.v_new,
            dt=dt,
            nu=self.nu,
            origin=(state.grid.hx, state.grid.hy + 1, 0),
            domain=(state.grid.nx, state.grid.ny - 2, 1)
        )
