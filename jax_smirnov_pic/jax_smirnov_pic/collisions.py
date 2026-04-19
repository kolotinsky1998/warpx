from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import jax
import jax.numpy as jnp
import numpy as np

from .config import EV
from .particles import first_free_slots, write_particles
from .state import ParticlePool

K_B = 1.380649e-23
C = 299792458.0
E_M = 9.10938356e-31


@dataclass(frozen=True)
class CrossSectionTable:
    energy_ev: jax.Array
    sigma_m2: jax.Array


def load_cross_section(path: str | Path) -> CrossSectionTable:
    data = np.loadtxt(path)
    return CrossSectionTable(
        energy_ev=jnp.asarray(data[:, 0], dtype=jnp.float32),
        sigma_m2=jnp.asarray(data[:, 1], dtype=jnp.float32),
    )


def cross_section_value(table: CrossSectionTable, energy_joule):
    energy_ev = energy_joule / EV
    return jnp.interp(energy_ev, table.energy_ev, table.sigma_m2, left=table.sigma_m2[0], right=table.sigma_m2[-1])


def velocity_norm(vx, vy, vz):
    return jnp.sqrt(vx * vx + vy * vy + vz * vz + 1.0e-30)


def electron_elastic_probability(pool: ParticlePool, table: CrossSectionTable, gas_density: float, dt: float, particle_mass: float):
    speed = velocity_norm(pool.vx, pool.vy, pool.vz)
    energy = 0.5 * particle_mass * speed * speed
    sigma = cross_section_value(table, energy)
    return jnp.where(pool.alive, sigma * gas_density * speed * dt, 0.0)


def ion_elastic_probability(pool: ParticlePool, table: CrossSectionTable, gas_density: float, dt: float, particle_mass: float):
    speed = velocity_norm(pool.vx, pool.vy, pool.vz)
    energy = 0.5 * particle_mass * speed * speed
    sigma = cross_section_value(table, energy)
    return jnp.where(pool.alive, sigma * gas_density * dt, 0.0)


def ionization_probability(pool: ParticlePool, table: CrossSectionTable, gas_density: float, dt: float, particle_mass: float):
    speed = velocity_norm(pool.vx, pool.vy, pool.vz)
    energy = 0.5 * particle_mass * speed * speed
    sigma = cross_section_value(table, energy)
    return jnp.where(pool.alive, sigma * gas_density * speed * dt, 0.0)


def apply_electron_elastic(pool: ParticlePool, collision_mask: jax.Array, key, gas_mass: float, particle_mass: float):
    key_phi, key_theta = jax.random.split(key)
    phi = 2.0 * jnp.pi * jax.random.uniform(key_phi, pool.vx.shape, dtype=jnp.float32)
    speed = velocity_norm(pool.vx, pool.vy, pool.vz)
    energy = 0.5 * particle_mass * speed * speed
    energy_ev = energy / EV
    rand = jax.random.uniform(key_theta, pool.vx.shape, dtype=jnp.float32)
    theta = jnp.where(
        energy_ev > 0.0,
        jnp.arccos(jnp.clip((2.0 + energy_ev - 2.0 * jnp.power(1.0 + energy_ev, rand)) / jnp.maximum(energy_ev, 1.0e-6), -1.0, 1.0)),
        0.0,
    )
    factor = jnp.sqrt(jnp.maximum(1.0 - 2.0 * particle_mass / gas_mass * (1.0 - jnp.cos(theta)), 0.0))
    new_speed = speed * factor
    vx = jnp.where(collision_mask, new_speed * jnp.cos(phi), pool.vx)
    vy = jnp.where(collision_mask, new_speed * jnp.sin(phi), pool.vy)
    return pool._replace(vx=vx, vy=vy)


def apply_ion_elastic_or_cx(pool: ParticlePool, collision_mask: jax.Array, key, gas_mass: float, gas_temperature: float, charge_exchange: bool = True):
    key_branch, key_vel = jax.random.split(key)
    sigma = jnp.sqrt(K_B * gas_temperature / gas_mass)
    gas_vel = jax.random.normal(key_vel, (pool.vx.shape[0], 3), dtype=jnp.float32) * sigma
    branch = jax.random.uniform(key_branch, pool.vx.shape, dtype=jnp.float32)
    elastic_mask = collision_mask & ((branch <= 0.5) if charge_exchange else jnp.ones_like(branch, dtype=bool))
    cx_mask = collision_mask & (~elastic_mask)
    new_vx_el = (pool.vx * gas_mass + gas_vel[:, 0] * gas_mass + gas_mass * (gas_vel[:, 0] - pool.vx)) / (2.0 * gas_mass)
    new_vy_el = (pool.vy * gas_mass + gas_vel[:, 1] * gas_mass + gas_mass * (gas_vel[:, 1] - pool.vy)) / (2.0 * gas_mass)
    new_vz_el = (pool.vz * gas_mass + gas_vel[:, 2] * gas_mass + gas_mass * (gas_vel[:, 2] - pool.vz)) / (2.0 * gas_mass)
    vx = jnp.where(elastic_mask, new_vx_el, pool.vx)
    vy = jnp.where(elastic_mask, new_vy_el, pool.vy)
    vz = jnp.where(elastic_mask, new_vz_el, pool.vz)
    vx = jnp.where(cx_mask, gas_vel[:, 0], vx)
    vy = jnp.where(cx_mask, gas_vel[:, 1], vy)
    vz = jnp.where(cx_mask, gas_vel[:, 2], vz)
    return pool._replace(vx=vx, vy=vy, vz=vz)


def apply_ionization(electrons: ParticlePool, ions: ParticlePool, ionization_mask: jax.Array, particle_mass_e: float):
    speed = velocity_norm(electrons.vx, electrons.vy, electrons.vz)
    energy = 0.5 * particle_mass_e * speed * speed
    ion_threshold = 16.0 * EV
    new_energy = energy - ion_threshold
    valid = ionization_mask & (new_energy > 0.0)
    new_speed = jnp.sqrt(jnp.maximum(2.0 * new_energy / particle_mass_e, 0.0))
    scale = new_speed / speed
    electrons = electrons._replace(
        vx=jnp.where(valid, electrons.vx * scale, electrons.vx),
        vy=jnp.where(valid, electrons.vy * scale, electrons.vy),
        vz=jnp.where(valid, electrons.vz * scale, electrons.vz),
    )
    spawn_count = int(valid.astype(jnp.int32).sum())
    if spawn_count == 0:
        return electrons, ions, 0
    source_idx = jnp.where(valid, size=electrons.alive.shape[0], fill_value=-1)[0][:spawn_count]
    source_valid = source_idx >= 0
    ex_idx, ex_valid = first_free_slots(electrons.alive, spawn_count)
    ion_idx, ion_valid = first_free_slots(ions.alive, spawn_count)
    src_safe = jnp.where(source_valid, source_idx, 0)
    x_new = electrons.x[src_safe]
    y_new = electrons.y[src_safe]
    zeros = jnp.zeros((spawn_count,), dtype=jnp.float32)
    electrons = write_particles(electrons, ex_idx, ex_valid, x_new, y_new, zeros, zeros, zeros)
    ions = write_particles(ions, ion_idx, ion_valid, x_new, y_new, zeros, zeros, zeros)
    created = int(jnp.minimum(ex_valid.astype(jnp.int32).sum(), ion_valid.astype(jnp.int32).sum()))
    return electrons, ions, created
