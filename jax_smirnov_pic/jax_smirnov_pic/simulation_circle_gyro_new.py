from __future__ import annotations

from dataclasses import asdict
import time

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
from .interpolation import linear_charge_deposition, linear_field_gather, rho_filter_new
from .io import append_csv_row, ensure_output_dir, save_matrix_txt, write_json, write_metadata
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


def _block_particles(pool):
    jax.block_until_ready(pool.x)
    jax.block_until_ready(pool.y)
    jax.block_until_ready(pool.vx)
    jax.block_until_ready(pool.vy)
    jax.block_until_ready(pool.vz)
    jax.block_until_ready(pool.alive)


def _block_fields(fields):
    jax.block_until_ready(fields.rho_e)
    jax.block_until_ready(fields.rho_i)
    jax.block_until_ready(fields.rho)
    jax.block_until_ready(fields.phi)
    jax.block_until_ready(fields.ex)
    jax.block_until_ready(fields.ey)


def _timed_stage(profile_row: dict, name: str, fn):
    t0 = time.perf_counter()
    result = fn()
    if isinstance(result, tuple):
        for item in result:
            if hasattr(item, "block_until_ready"):
                jax.block_until_ready(item)
    elif hasattr(result, "block_until_ready"):
        jax.block_until_ready(result)
    profile_row[name] = time.perf_counter() - t0
    return result


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


def step_once_profiled(state, tables, config: SimulationConfig):
    profile_row = {"step": int(state.step)}

    rho_e = _timed_stage(
        profile_row,
        "deposit_electrons_s",
        lambda: linear_charge_deposition(state.electrons, state.geometry, _electron_charge(state)),
    )
    rho_i = _timed_stage(
        profile_row,
        "deposit_ions_s",
        lambda: linear_charge_deposition(state.ions, state.geometry, _ion_charge(state)),
    )
    rho = _timed_stage(
        profile_row,
        "rho_filter_s",
        lambda: rho_filter_new(rho_e + rho_i, state.geometry.nr_anode),
    )
    phi = _timed_stage(
        profile_row,
        "poisson_s",
        lambda: solve_poisson_weighted_jacobi(
            state.fields.phi, rho, state.geometry, config.poisson_omega, config.poisson_iterations
        ),
    )
    ex_grid, ey_grid = _timed_stage(
        profile_row,
        "field_from_phi_s",
        lambda: compute_electric_field(phi, state.geometry),
    )
    ex_e, ey_e = _timed_stage(
        profile_row,
        "gather_electrons_s",
        lambda: linear_field_gather(state.electrons, ex_grid, ey_grid, state.geometry),
    )
    ex_i, ey_i = _timed_stage(
        profile_row,
        "gather_ions_s",
        lambda: linear_field_gather(state.ions, ex_grid, ey_grid, state.geometry),
    )
    electrons = state.electrons._replace(ex=ex_e, ey=ey_e)
    ions = state.ions._replace(ex=ex_i, ey=ey_i)

    electrons = _timed_stage(
        profile_row,
        "push_electrons_s",
        lambda: gyro_push(electrons, state.runtime.dt, -EV, E_M),
    )
    if int(state.step) % state.runtime.ion_step == 0:
        ions = _timed_stage(
            profile_row,
            "push_ions_s",
            lambda: boris_push(ions, state.runtime.dt * state.runtime.ion_step, EV, config.m_ion),
        )
    else:
        profile_row["push_ions_s"] = 0.0

    counters = state.counters
    rng_key = state.rng_key
    if int(state.step) % config.collision_step_electron == 0:
        key1, key2, key3, new_key = jax.random.split(rng_key, 4)

        prob_el = _timed_stage(
            profile_row,
            "electron_collision_probability_s",
            lambda: electron_elastic_probability(
                electrons,
                tables["electron_elastic"],
                state.runtime.n_gas_scaled,
                config.collision_step_electron * state.runtime.dt,
                E_M,
            ),
        )
        prob_ion = _timed_stage(
            profile_row,
            "ionization_probability_s",
            lambda: ionization_probability(
                electrons,
                tables["electron_ionization"],
                state.runtime.n_gas_scaled,
                config.collision_step_electron * state.runtime.dt,
                E_M,
            ),
        )
        rand = _timed_stage(
            profile_row,
            "electron_collision_rng_s",
            lambda: jax.random.uniform(key1, prob_el.shape, dtype=jnp.float32),
        )
        elastic_mask = rand < prob_el
        ionization_mask = (rand >= prob_el) & (rand < prob_el + prob_ion)
        electrons = _timed_stage(
            profile_row,
            "electron_elastic_collision_s",
            lambda: apply_electron_elastic(electrons, elastic_mask, key2, config.m_ion, E_M),
        )
        electrons, ions, created = _timed_stage(
            profile_row,
            "electron_ionization_s",
            lambda: apply_ionization(electrons, ions, ionization_mask, E_M),
        )
        counters = counters._replace(ntot_ionized=counters.ntot_ionized + created)
        rng_key = new_key
    else:
        profile_row["electron_collision_probability_s"] = 0.0
        profile_row["ionization_probability_s"] = 0.0
        profile_row["electron_collision_rng_s"] = 0.0
        profile_row["electron_elastic_collision_s"] = 0.0
        profile_row["electron_ionization_s"] = 0.0

    if int(state.step) % config.collision_step_ion == 0:
        key4, rng_key = jax.random.split(rng_key)
        prob_i = _timed_stage(
            profile_row,
            "ion_collision_probability_s",
            lambda: ion_elastic_probability(
                ions,
                tables["ion_elastic"],
                state.runtime.n_gas_scaled,
                config.collision_step_ion * state.runtime.dt,
                config.m_ion,
            ),
        )
        rand_i = _timed_stage(
            profile_row,
            "ion_collision_rng_s",
            lambda: jax.random.uniform(key4, prob_i.shape, dtype=jnp.float32),
        )
        ions = _timed_stage(
            profile_row,
            "ion_collision_apply_s",
            lambda: apply_ion_elastic_or_cx(ions, rand_i < prob_i, key4, config.m_ion, config.t_gas),
        )
    else:
        profile_row["ion_collision_probability_s"] = 0.0
        profile_row["ion_collision_rng_s"] = 0.0
        profile_row["ion_collision_apply_s"] = 0.0

    electrons, removed_e = _timed_stage(
        profile_row,
        "anode_loss_s",
        lambda: remove_on_anode(electrons, state.geometry),
    )
    counters = counters._replace(ntot_anode_leave=counters.ntot_anode_leave + removed_e)

    if int(state.step) % config.ion_leave_step == 0:
        key5, rng_key = jax.random.split(rng_key)
        ions, removed_i = _timed_stage(
            profile_row,
            "cold_cathode_ion_sink_s",
            lambda: remove_some_ions_on_cold_cathode(ions, key5, state.geometry, state.runtime),
        )
        counters = counters._replace(ntot_cold_cathode_leave=counters.ntot_cold_cathode_leave + removed_i)
    else:
        profile_row["cold_cathode_ion_sink_s"] = 0.0

    if int(state.step) % config.electron_emission_hot_step == 0:
        key6, rng_key = jax.random.split(rng_key)
        electrons, emitted_hot = _timed_stage(
            profile_row,
            "hot_cathode_emission_s",
            lambda: hot_cathode_emission(electrons, key6, state.geometry, state.runtime, E_M),
        )
        counters = counters._replace(ntot_hot_cathode_emission=counters.ntot_hot_cathode_emission + emitted_hot)
    else:
        profile_row["hot_cathode_emission_s"] = 0.0

    if int(state.step) % config.electron_emission_cold_step == 0:
        key7, rng_key = jax.random.split(rng_key)
        electrons, _ = _timed_stage(
            profile_row,
            "cold_secondary_emission_s",
            lambda: cold_secondary_emission(electrons, key7, state.geometry, state.runtime, E_M),
        )
    else:
        profile_row["cold_secondary_emission_s"] = 0.0

    state = state._replace(
        electrons=electrons,
        ions=ions,
        fields=state.fields._replace(rho_e=rho_e, rho_i=rho_i, rho=rho, phi=phi, ex=ex_grid, ey=ey_grid),
        counters=counters,
        step=state.step + 1,
        rng_key=rng_key,
    )
    _block_particles(state.electrons)
    _block_particles(state.ions)
    _block_fields(state.fields)
    profile_row["step_total_s"] = sum(
        value for key, value in profile_row.items() if key.endswith("_s") and key != "step_total_s"
    )
    return state, profile_row


def run_simulation(config: SimulationConfig | None = None, profile: bool = False, profile_summary_only: bool = False):
    config = config or smirnov_default_config()
    out_dir = ensure_output_dir(config.output_dir)
    write_metadata(out_dir / "metadata.json", {
        "config": asdict(config),
        "profiling_enabled": profile,
    })
    state = create_initial_state(config)
    state = initialize_particles(state, config)
    tables = {
        "electron_elastic": load_cross_section(config.cross_sections.electron_elastic),
        "electron_ionization": load_cross_section(config.cross_sections.electron_ionization),
        "ion_elastic": load_cross_section(config.cross_sections.ion_elastic),
    }
    profile_rows = []

    for it in range(config.it_num):
        io_t0 = time.perf_counter()
        if profile:
            state, profile_row = step_once_profiled(state, tables, config)
        else:
            state = step_once(state, tables, config)
            profile_row = None
        io_before = time.perf_counter()
        if it % config.log_interval == 0:
            append_csv_row(out_dir / "counters.csv", counters_row(it, state.electrons, state.ions, state.geometry, state.counters))
        if it % config.field_dump_interval == 0:
            save_matrix_txt(out_dir / f"rho_e_{it}.txt", state.fields.rho_e)
            save_matrix_txt(out_dir / f"rho_i_{it}.txt", state.fields.rho_i)
            save_matrix_txt(out_dir / f"phi_{it}.txt", state.fields.phi)
        io_after = time.perf_counter()
        if profile and profile_row is not None:
            profile_row["io_s"] = io_after - io_before
            profile_row["wall_total_with_io_s"] = io_after - io_t0
            if not profile_summary_only:
                profile_rows.append(profile_row)

    if profile:
        if profile_summary_only:
            profile_rows = []
        profile_path = out_dir / "profiling.csv"
        summary_path = out_dir / "profiling_summary.json"
        if profile_rows:
            for row in profile_rows:
                append_csv_row(profile_path, row)
            keys = [key for key in profile_rows[0].keys() if key.endswith("_s")]
            summary = {
                key: {
                    "mean_s": sum(row[key] for row in profile_rows) / len(profile_rows),
                    "max_s": max(row[key] for row in profile_rows),
                    "min_s": min(row[key] for row in profile_rows),
                    "sum_s": sum(row[key] for row in profile_rows),
                }
                for key in keys
            }
            write_json(summary_path, summary)
        else:
            write_json(summary_path, {"note": "Profiling enabled, but per-step rows were not stored."})

    return state
