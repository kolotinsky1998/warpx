#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time

import numpy as np
from scipy.constants import e, m_e

import pywarpx
from pywarpx import callbacks, particle_containers, picmi


#################################
# USER CONSTANTS
#################################
R = 0.005  # [m] outer anode radius
Rinj = 0.0002  # [m] hot cathode radius / cold cathode inner radius
Nx = 72
Nz = 72
By0 = 5.0  # [T], must stay perpendicular to the X-Z plane

nAr = 3.865e21  # [m^-3] ~200 mTorr @ 500 K
TAr = 500.0  # [K]

I_hot_cathode = 0.12  # [A]

hot_emission_energy_eV = 100.0

# Time step as in the C++ Smirnov reference: dt_e = gyro_coeff / (10 * omega_ce)
gyro_coeff = 100.0
omega_ce = e * By0 / m_e
dt = gyro_coeff / (10.0 * omega_ce)

max_steps = 100000
diagnostic_period = 500

# Built-in WarpX real-space binomial/bilinear filter.
# This is not identical to Smirnov's custom rho_filter_new, but it is the
# closest built-in option in WarpX and should help suppress hot-cathode
# charge-density spikes before the electrostatic solve.
use_warpx_filter = 1
filter_npass_each_dir = [2, 2]

# Smirnov-like scaling used to set the emitted macroparticle weight.
scale = 0.02
n_e_real = 1.0e15
n_e_scaled = n_e_real / scale
emission_density_scale = n_e_scaled / 10.0
dx = 2.0 * R / Nx
emission_radius = 0.3 * Nx * dx
emission_ptcls_per_cell = 1.0
emission_macro_count = max(
    1, int(np.pi * (emission_radius / dx) ** 2 * emission_ptcls_per_cell)
)
macro_weight = emission_density_scale * np.pi * emission_radius ** 2 / emission_macro_count

# External Ex drives an E x B drift along +z:
# v_d = |E x B| / B^2 = Ex / By.
# With Ex = 5e5 V/m and By = 5 T, v_d ~ 1e5 m/s, so electrons launched near
# the center cross ~5 mm in ~5e-8 s, i.e. in roughly 4.4e3 steps.
Ex0 = 5.0e5  # [V/m]

# Smirnov-style hot-emission cadence.
hot_cathode_interval = 500

N_hot_emit = max(1, int(I_hot_cathode * hot_cathode_interval * dt / e / macro_weight))

log_file = "log_electrons_only.txt"

total_hot_emitted = 0


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
# SPECIES: electrons only
#################################
electrons = picmi.Species(
    particle_type="electron",
    name="electrons",
)


#################################
# COLLISIONS: electrons with background Ar
#################################
coll_e = picmi.MCCCollisions(
    name="e_on_Ar",
    species=electrons,
    background_density=nAr,
    background_temperature=TAr,
    background_mass=6.6335e-26,
    scattering_processes={
        "elastic": {"cross_section": "e_ar_effective.txt"},
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
    warpx_use_filter=use_warpx_filter,
    warpx_grid_type="staggered",
    warpx_particle_pusher_algo="boris",
    warpx_embedded_boundary=embedded_boundary,
    warpx_collisions=[coll_e],
)


#################################
# EXTERNAL MAGNETIC FIELD
#################################
B_ext = picmi.ConstantAppliedField(Ex=Ex0, By=By0)
sim.add_applied_field(B_ext)

zero_layout = picmi.GriddedLayout(n_macroparticle_per_cell=[0, 0], grid=grid)
sim.add_species(electrons, layout=zero_layout, initialize_self_field=False)


#################################
# DIAGNOSTICS
#################################
diag_full = picmi.FieldDiagnostic(
    name="diag_fields_penning_gyro_electrons_only",
    grid=grid,
    period=diagnostic_period,
    data_list=["Ex", "Ez", "rho", "phi"],
    write_dir="diags_electrons_only",
    warpx_format="openpmd",
)
sim.add_diagnostic(diag_full)

diag_particles = picmi.ParticleDiagnostic(
    name="diag_particles_penning_gyro_electrons_only",
    period=diagnostic_period,
    species=[electrons],
    data_list=["x", "z", "ux", "uy", "uz", "w"],
    write_dir="diags_electrons_only",
    warpx_format="openpmd",
)
sim.add_diagnostic(diag_particles)


#################################
# INITIALIZE WARPX AND SET PER-SPECIES MODELS
#################################
sim.initialize_inputs()

# PICMI exposes warpx_use_filter directly on Simulation, while the number of
# filter passes is set on the WarpX runtime object before initialization.
pywarpx.warpx.filter_npass_each_dir = filter_npass_each_dir

electrons.species.particle_pusher = "gyrokinetic"

# Hot cathode source: batch source in the central disk.
electrons.species.cathode_hot_source = 1
electrons.species.cathode_hot_rmin = 0.0
electrons.species.cathode_hot_rmax = Rinj
electrons.species.cathode_hot_center = [0.0, 0.0]
electrons.species.cathode_hot_interval = hot_cathode_interval
electrons.species.cathode_hot_npart_per_injection = N_hot_emit
electrons.species.cathode_hot_macro_weight = macro_weight
electrons.species.cathode_hot_energy_eV = hot_emission_energy_eV

sim.initialize_warpx()

electron_pc = particle_containers.ParticleContainerWrapper("electrons")


def current_step():
    return sim.extension.warpx.getistep(lev=0)


def uniform_disk_positions(n_particles, radius):
    rr = radius * np.sqrt(np.random.random(n_particles))
    theta = 2.0 * np.pi * np.random.random(n_particles)
    x = rr * np.cos(theta)
    z = rr * np.sin(theta)
    y = np.zeros_like(x)
    return x, y, z


def update_hot_emission_counter():
    global total_hot_emitted

    step = current_step()
    if step == 0 or step % hot_cathode_interval != 0:
        return

    total_hot_emitted += N_hot_emit


def runtime_status():
    step = current_step()
    if step == 0 or step % diagnostic_period != 0:
        return

    ne = electron_pc.get_particle_count(local=False)
    print(
        "[penning] "
        f"step={step} "
        f"Ne={ne} "
        f"Nhot_total={total_hot_emitted}"
    )


with open(log_file, "w", encoding="utf-8") as f:
    f.write("step ne nhot_total\n")


def log_particle_counts():
    step = current_step()
    ne = electron_pc.get_particle_count(local=False)

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"{step} {ne} {total_hot_emitted}\n")

callbacks.installafterstep(update_hot_emission_counter)
callbacks.installafterstep(runtime_status)
callbacks.installafterstep(log_particle_counts)


#################################
# RUN SIMULATION
#################################
print(
    "[penning] "
    f"dt={dt:.6e} s, "
    f"omega_ce={omega_ce:.6e} 1/s, "
    f"use_filter={use_warpx_filter}, "
    f"filter_npass_each_dir={filter_npass_each_dir}, "
    f"macro_weight={macro_weight:.6e}, "
    f"initial_electrons=0, "
    f"emission_macro_count={emission_macro_count}, "
    f"emission_radius={emission_radius:.6e}, "
    f"Ex={Ex0:.6e} V/m, "
    f"v_ExB={Ex0 / By0:.6e} m/s, "
    f"hot_interval={hot_cathode_interval}, "
    f"N_hot_emit={N_hot_emit}"
)

start = time.time()
sim.step(max_steps)
end = time.time()

print(f"[penning] elapsed_time={end - start:.6f} s")
