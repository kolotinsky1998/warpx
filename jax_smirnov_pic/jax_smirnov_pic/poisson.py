from __future__ import annotations

import jax
import jax.numpy as jnp

from .geometry import circular_domain_mask
from .state import GeometryState

EPSILON_0 = 8.854187817620389e-12


def weighted_jacobi_step(phi, rho, inside_mask, dx, dy, omega):
    dx2 = dx * dx
    dy2 = dy * dy
    laplace_rhs = (
        rho / EPSILON_0
        + (jnp.roll(phi, 1, axis=0) + jnp.roll(phi, -1, axis=0)) / dx2
        + (jnp.roll(phi, 1, axis=1) + jnp.roll(phi, -1, axis=1)) / dy2
    ) / (2.0 / dx2 + 2.0 / dy2)
    updated = (1.0 - omega) * phi + omega * laplace_rhs
    return jnp.where(inside_mask, updated, 0.0)


def solve_poisson_weighted_jacobi(phi0, rho, geometry: GeometryState, omega: float, n_iter: int):
    inside_mask = circular_domain_mask(geometry)

    def body(_, phi):
        return weighted_jacobi_step(phi, rho, inside_mask, geometry.dx, geometry.dy, omega)

    return jax.lax.fori_loop(0, n_iter, body, phi0)


def compute_electric_field(phi, geometry: GeometryState):
    ex = (jnp.roll(phi, 1, axis=0) - jnp.roll(phi, -1, axis=0)) / (2.0 * geometry.dx)
    ey = (jnp.roll(phi, 1, axis=1) - jnp.roll(phi, -1, axis=1)) / (2.0 * geometry.dy)
    ex = ex.at[0, :].set((phi[0, :] - phi[1, :]) / geometry.dx)
    ex = ex.at[-1, :].set((phi[-2, :] - phi[-1, :]) / geometry.dx)
    ey = ey.at[:, 0].set((phi[:, 0] - phi[:, 1]) / geometry.dy)
    ey = ey.at[:, -1].set((phi[:, -2] - phi[:, -1]) / geometry.dy)
    return ex.astype(jnp.float32), ey.astype(jnp.float32)
