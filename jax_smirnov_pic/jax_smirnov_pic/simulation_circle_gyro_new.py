from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp

from .collisions import (
    apply_electron_elastic,
    apply_ion_elastic_or_cx,
    apply_ionization,
    electron_elastic_probability,
    ion_elastic_probability,
    ionization_probability,
    load_cross_section,
)
from .config import E_M, EV, smirnov_default_config, SimulationConfig
from .diagnostics import counters_row
from .geometry import inside_anode
from .interpolation import linear_charge_deposition, linear_field_gather, rho_filter_new
from .io import append_csv_row, ensure_output_dir, save_matrix_txt, write_metadata
from .gyro import gyro_push
from .particles import boris_push
from .poisson import compute_electric_field, solve_poisson_weighted_jacobi
from .sources_sinks import (
    cold_secondary_emission,
    hot_cathode_emission,
    inject_initial_disk,
    remove_on_anode,
    remove_some_ions_on_cold_cathode,
    update_counter,
)
from .state import create_initial_state


def _electron_charge(state):
    return -EV * state.runtime.ptcls_per_macro


def _ion_charge(state):
    return EV * state.runtime.ptcls_per_macro


def initialize_particles(state, config: SimulationConfig):
    speed_std_e = jnp.sqrt(2.0 * config.init_energy_ev * EV / (3.0 * E_M))
    speed_std_i = jnp.sqrt(2.0 * config.init_energy_ev * EV / (3.0 * config.m_ion))
    ntot_seed = max(1, int(jnp.pi * (state.geometry.radius_injection / state.geometry.dx) ** 2 * config.ptcls_per_cell))
    key_e, key_i, new_key = jax.random.split(state.rng_key, 3)
    electrons = inject_initial_disk(state.electrons, key_e, state.geometry, ntot_seed, speed_std_e)
    ions = inject_initial_disk(state.ions, key_i, state.geometry, ntot_seed, speed_std_i)
    return state._replace(electrons=electrons, ions=ions, rng_key=new_key)


def step_once(state, tables, config: SimulationConfig):
    rho_e = linear_charge_deposition(state.electrons, state.geometry, _electron_charge(state))
    rho_i = linear_charge_deposition(state.ions, state.geometry, _ion_charge(state))
    rho = rho_filter_new(rho_e + rho_i, state.geometry.nr_anode)
    phi = solve_poisson_weighted_jacobi(
        state.fields.phi, rho, state.geometry, config.poisson_omega, config.poisson_iterations
    )
    ex_grid, ey_grid = compute_electric_field(phi, state.geometry)
    ex_e, ey_e = linear_field_gather(state.electrons, ex_grid, ey_grid, state.geometry)
    ex_i, ey_i = linear_field_gather(state.ions, ex_grid, ey_grid, state.geometry)
    electrons = state.electrons._replace(ex=ex_e, ey=ey_e)
    ions = state.ions._replace(ex=ex_i, ey=ey_i)

    electrons = gyro_push(electrons, state.runtime.dt, -EV, E_M)
    if int(state.step) % state.runtime.ion_step == 0:
        ions = boris_push(ions, state.runtime.dt * state.runtime.ion_step, EV, config.m_ion)

    counters = state.counters
    if int(state.step) % config.collision_step_electron == 0:
        key1, key2, key3, new_key = jax.random.split(state.rng_key, 4)
        prob_el = electron_elastic_probability(
            electrons, tables["electron_elastic"], state.runtime.n_gas_scaled, config.collision_step_electron * state.runtime.dt, E_M
        )
        prob_ion = ionization_probability(
            electrons, tables["electron_ionization"], state.runtime.n_gas_scaled, config.collision_step_electron * state.runtime.dt, E_M
        )
        rand = jax.random.uniform(key1, prob_el.shape, dtype=jnp.float32)
        elastic_mask = rand < prob_el
        ionization_mask = (rand >= prob_el) & (rand < prob_el + prob_ion)
        electrons = apply_electron_elastic(electrons, elastic_mask, key2, config.m_ion, E_M)
        electrons, ions, created = apply_ionization(electrons, ions, ionization_mask, E_M)
        counters = counters._replace(ntot_ionized=counters.ntot_ionized + created)
        rng_key = new_key
    else:
        rng_key = state.rng_key

    if int(state.step) % config.collision_step_ion == 0:
        key4, rng_key = jax.random.split(rng_key)
        prob_i = ion_elastic_probability(
            ions, tables["ion_elastic"], state.runtime.n_gas_scaled, config.collision_step_ion * state.runtime.dt, config.m_ion
        )
        rand_i = jax.random.uniform(key4, prob_i.shape, dtype=jnp.float32)
        ions = apply_ion_elastic_or_cx(ions, rand_i < prob_i, key4, config.m_ion, config.t_gas)

    electrons, removed_e = remove_on_anode(electrons, state.geometry)
    counters = counters._replace(ntot_anode_leave=counters.ntot_anode_leave + removed_e)

    if int(state.step) % config.ion_leave_step == 0:
        key5, rng_key = jax.random.split(rng_key)
        ions, removed_i = remove_some_ions_on_cold_cathode(ions, key5, state.geometry, state.runtime)
        counters = counters._replace(ntot_cold_cathode_leave=counters.ntot_cold_cathode_leave + removed_i)

    if int(state.step) % config.electron_emission_hot_step == 0:
        key6, rng_key = jax.random.split(rng_key)
        electrons, emitted_hot = hot_cathode_emission(electrons, key6, state.geometry, state.runtime, E_M)
        counters = counters._replace(ntot_hot_cathode_emission=counters.ntot_hot_cathode_emission + emitted_hot)

    if int(state.step) % config.electron_emission_cold_step == 0:
        key7, rng_key = jax.random.split(rng_key)
        electrons, _ = cold_secondary_emission(electrons, key7, state.geometry, state.runtime, E_M)

    return state._replace(
        electrons=electrons,
        ions=ions,
        fields=state.fields._replace(rho_e=rho_e, rho_i=rho_i, rho=rho, phi=phi, ex=ex_grid, ey=ey_grid),
        counters=counters,
        step=state.step + 1,
        rng_key=rng_key,
    )


def run_simulation(config: SimulationConfig | None = None):
    config = config or smirnov_default_config()
    out_dir = ensure_output_dir(config.output_dir)
    write_metadata(out_dir / "metadata.json", {
        "config": asdict(config),
    })
    state = create_initial_state(config)
    state = initialize_particles(state, config)
    tables = {
        "electron_elastic": load_cross_section(config.cross_sections.electron_elastic),
        "electron_ionization": load_cross_section(config.cross_sections.electron_ionization),
        "ion_elastic": load_cross_section(config.cross_sections.ion_elastic),
    }

    for it in range(config.it_num):
        state = step_once(state, tables, config)
        if it % config.log_interval == 0:
            append_csv_row(out_dir / "counters.csv", counters_row(it, state.electrons, state.ions, state.geometry, state.counters))
        if it % config.field_dump_interval == 0:
            save_matrix_txt(out_dir / f"rho_e_{it}.txt", state.fields.rho_e)
            save_matrix_txt(out_dir / f"rho_i_{it}.txt", state.fields.rho_i)
            save_matrix_txt(out_dir / f"phi_{it}.txt", state.fields.phi)

    return state
