#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time

import numpy as np
from scipy.constants import e, m_e

from pywarpx import callbacks, particle_containers, picmi


#################################
# USER CONSTANTS
#################################
R = 0.005  # [m] outer anode radius
Rinj = 0.0002  # [m] hot cathode radius
Rcold = 0.003  # [m] cold cathode outer radius
Nx = 200
Nz = 200
By0 = 5.0  # [T], must stay perpendicular to the X-Z plane

nAr = 3.865e21  # [m^-3] ~200 mTorr @ 500 K
TAr = 500.0  # [K]
m_i = 500.0 * m_e  # Smirnov-like optimal mass ratio

sigma_v = 6.08e5  # [m/s]
v_drift = 5.94e6  # [m/s]
I_hot_cathode = 0.12  # [A]

# Time step as in the C++ Smirnov reference: dt_e = gyro_coeff / (10 * omega_ce)
gyro_coeff = 100.0
omega_ce = e * By0 / m_e
dt = gyro_coeff / (10.0 * omega_ce)

max_steps = 1_000_000
diagnostic_period = 5_000

# Emit one macro-electron every 2.5 time steps on average.
emission_steps_per_macro = 2.5
emission_rate_per_step = 1.0 / emission_steps_per_macro
macro_weight = I_hot_cathode * dt * emission_steps_per_macro / e

# Hot cathode injector state.
emission_accumulator = 0.0
emitted_macro_electrons = 0

# Cold electrode monitor state. This is a safe Python-side scaffold for a later
# absorber/emitter implementation; it only reports ion occupancy in the cathode zone.
cold_cathode_ion_count_snapshot = 0
cold_cathode_ion_weight_snapshot = 0.0


#################################
# GEOMETRY (2D XZ)
#################################
grid = picmi.Cartesian2DGrid(
    number_of_cells=[Nx, Nz],
    lower_bound=[-R, -R],
    upper_bound=[R, R],
    lower_boundary_conditions=["dirichlet", "dirichlet"],
    upper_boundary_conditions=["dirichlet", "dirichlet"],
    lower_boundary_conditions_particles=["absorbing", "absorbing"],
    upper_boundary_conditions_particles=["absorbing", "absorbing"],
)


#################################
# ELECTROSTATIC SOLVER (Poisson)
#################################
solver = picmi.ElectrostaticSolver(grid=grid)


#################################
# EMBEDDED BOUNDARY: circular anode
#################################
embedded_boundary = picmi.EmbeddedBoundary(
    implicit_function=f"x*x + z*z - {R}*{R}",
    potential="0.0",
)


#################################
# SPECIES: electrons and ions
#################################
electrons = picmi.Species(
    particle_type="electron",
    name="electrons",
)

ions = picmi.Species(
    name="ions",
    charge=e,
    mass=m_i,
)


#################################
# COLLISIONS: e & ion with background Ar
#################################
coll_e = picmi.MCCCollisions(
    name="e_on_Ar",
    species=electrons,
    background_density=nAr,
    background_temperature=TAr,
    background_mass=6.6335e-26,
    scattering_processes={
        "elastic": {"cross_section": "e_ar_effective.txt"},
        "ionization": {
            "cross_section": "e_ar_ionization.txt",
            "energy": 15.8,
            "species": ions,
        },
    },
)

coll_i = picmi.MCCCollisions(
    name="ion_on_Ar",
    species=ions,
    background_density=nAr,
    background_temperature=TAr,
    background_mass=6.6335e-26,
    scattering_processes={
        "elastic": {"cross_section": "arplus_ar_backscat.txt"},
        "charge_exchange": {"cross_section": "arplus_ar_isotropic.txt"},
    },
)


#################################
# SIMULATION SETUP
#################################
sim = picmi.Simulation(
    solver=solver,
    time_step_size=dt,
    max_steps=max_steps,
    particle_shape="linear",
    warpx_grid_type="staggered",
    warpx_particle_pusher_algo="boris",
    warpx_embedded_boundary=embedded_boundary,
    warpx_collisions=[coll_e, coll_i],
)


#################################
# EXTERNAL MAGNETIC FIELD
#################################
B_ext = picmi.ConstantAppliedField(By=By0)
sim.add_applied_field(B_ext)

zero_layout = picmi.GriddedLayout(n_macroparticle_per_cell=[0, 0], grid=grid)
sim.add_species(electrons, layout=zero_layout, initialize_self_field=False)
sim.add_species(ions, layout=zero_layout, initialize_self_field=False)


#################################
# DIAGNOSTICS
#################################
diag_full = picmi.FieldDiagnostic(
    name="diag_fields_penning_gyro",
    grid=grid,
    period=diagnostic_period,
    data_list=["Ex", "Ez", "rho", "phi"],
    write_dir="diags",
    warpx_format="openpmd",
)
sim.add_diagnostic(diag_full)

diag_particles = picmi.ParticleDiagnostic(
    name="diag_particles_penning_gyro",
    period=diagnostic_period,
    species=[electrons, ions],
    data_list=["x", "z", "ux", "uy", "uz", "w"],
    write_dir="diags",
    warpx_format="openpmd",
)
sim.add_diagnostic(diag_particles)


#################################
# INITIALIZE WARPX AND SET PER-SPECIES PUSHERS
#################################
sim.initialize_inputs()

# PICMI currently exposes only the global pusher directly. The per-species
# override is still available through the WarpX species buckets before init.
electrons.species.particle_pusher = "gyrokinetic"
ions.species.particle_pusher = "boris"

sim.initialize_warpx()

electron_pc = particle_containers.ParticleContainerWrapper("electrons")
ion_pc = particle_containers.ParticleContainerWrapper("ions")


def concat_host(list_of_arrays):
    if len(list_of_arrays) == 0:
        return np.empty(0)
    return np.concatenate([np.asarray(arr) for arr in list_of_arrays])


def current_step():
    return sim.extension.warpx.getistep(lev=0)


def sample_hot_cathode_particles(n_particles):
    r = Rinj * np.sqrt(np.random.random(n_particles))
    theta = 2.0 * np.pi * np.random.random(n_particles)

    x = r * np.cos(theta)
    z = r * np.sin(theta)
    y = np.zeros_like(x)

    ux = np.random.normal(loc=0.0, scale=sigma_v, size=n_particles)
    uz = np.random.normal(loc=0.0, scale=sigma_v, size=n_particles)
    uy = np.full(n_particles, v_drift)

    w = np.full(n_particles, macro_weight)
    return x, y, z, ux, uy, uz, w


def inject_hot_cathode_electrons():
    global emission_accumulator
    global emitted_macro_electrons

    emission_accumulator += emission_rate_per_step
    n_new = int(emission_accumulator)
    if n_new == 0:
        return

    emission_accumulator -= n_new
    x, y, z, ux, uy, uz, w = sample_hot_cathode_particles(n_new)
    electron_pc.add_particles(x=x, y=y, z=z, ux=ux, uy=uy, uz=uz, w=w)
    emitted_macro_electrons += n_new


def monitor_cold_cathode_ions():
    global cold_cathode_ion_count_snapshot
    global cold_cathode_ion_weight_snapshot

    step = current_step()
    if step == 0 or step % diagnostic_period != 0:
        return

    x = concat_host(ion_pc.get_particle_x(level=0, copy_to_host=True))
    z = concat_host(ion_pc.get_particle_z(level=0, copy_to_host=True))
    w = concat_host(ion_pc.get_particle_weight(level=0, copy_to_host=True))

    if x.size == 0:
        cold_cathode_ion_count_snapshot = 0
        cold_cathode_ion_weight_snapshot = 0.0
        return

    radius = np.sqrt(x * x + z * z)
    mask = (radius >= Rinj) & (radius <= Rcold)
    cold_cathode_ion_count_snapshot = int(np.count_nonzero(mask))
    cold_cathode_ion_weight_snapshot = float(np.sum(w[mask]))


def runtime_status():
    step = current_step()
    if step == 0 or step % diagnostic_period != 0:
        return

    print(
        "[penning] "
        f"step={step} "
        f"Ne_macro_emitted={emitted_macro_electrons} "
        f"Nions_cold_zone={cold_cathode_ion_count_snapshot} "
        f"Wions_cold_zone={cold_cathode_ion_weight_snapshot:.6e}"
    )


callbacks.installbeforestep(inject_hot_cathode_electrons)
callbacks.installafterstep(monitor_cold_cathode_ions)
callbacks.installafterstep(runtime_status)


#################################
# RUN SIMULATION
#################################
print(
    "[penning] "
    f"dt={dt:.6e} s, "
    f"omega_ce={omega_ce:.6e} 1/s, "
    f"macro_weight={macro_weight:.6e}, "
    f"mass_ratio={m_i / m_e:.1f}"
)

start = time.time()
sim.step(max_steps)
end = time.time()

print(f"[penning] elapsed_time={end - start:.6f} s")
print(
    "[penning] "
    f"total_emitted_macro_electrons={emitted_macro_electrons} "
    f"cold_cathode_monitor_count={cold_cathode_ion_count_snapshot} "
    f"cold_cathode_monitor_weight={cold_cathode_ion_weight_snapshot:.6e}"
)
