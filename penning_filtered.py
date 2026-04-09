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
Rcold = 0.003  # [m] cold cathode outer radius
Nx = 72
Nz = 72
By0 = 5.0  # [T], must stay perpendicular to the X-Z plane

nAr = 3.865e21  # [m^-3] ~200 mTorr @ 500 K
TAr = 500.0  # [K]
m_i = 500.0 * m_e  # Smirnov-like optimal mass ratio

I_hot_cathode = 0.12  # [A]
gamma_secondary = 0.1
I_cold_cathode = I_hot_cathode * (4.0 / 6.0)  # same hot/cold current ratio as in Smirnov

hot_emission_energy_eV = 100.0
cold_secondary_energy_eV = 100.0
seed_energy_eV = 0.1

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

# Smirnov-like scaled plasma density and central seed region.
scale = 0.02
n_e_real = 1.0e15
n_e_scaled = n_e_real / scale
seed_density = n_e_scaled / 10.0
dx = 2.0 * R / Nx
seed_radius = 0.3 * Nx * dx
seed_ptcls_per_cell = 1.0
seed_macro_count = max(1, int(np.pi * (seed_radius / dx) ** 2 * seed_ptcls_per_cell))
macro_weight = seed_density * np.pi * seed_radius ** 2 / seed_macro_count

# Smirnov-style hot/cold cadences.
hot_cathode_interval = 500
cold_cathode_interval = 500

N_hot_emit = max(1, int(I_hot_cathode * hot_cathode_interval * dt / e / macro_weight))
I_ion_leave = I_cold_cathode / (1.0 + gamma_secondary)
N_ion_remove = max(1, int(I_ion_leave * cold_cathode_interval * dt / e / macro_weight))
I_cold_secondary = gamma_secondary * I_ion_leave
N_cold_secondary_emit = max(1, int(I_cold_secondary * cold_cathode_interval * dt / e / macro_weight))

log_file = "log.txt"

total_hot_emitted = 0
total_cold_secondary_emitted = 0
total_electron_outer_absorbed = 0
total_ion_outer_absorbed = 0


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
    warpx_save_particles_at_eb=True,
)

ions = picmi.Species(
    name="ions",
    charge=e,
    mass=m_i,
    warpx_save_particles_at_eb=True,
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
    warpx_use_filter=use_warpx_filter,
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
# INITIALIZE WARPX AND SET PER-SPECIES MODELS
#################################
sim.initialize_inputs()

# PICMI exposes warpx_use_filter directly on Simulation, while the number of
# filter passes is set on the WarpX runtime object before initialization.
pywarpx.warpx.filter_npass_each_dir = filter_npass_each_dir

electrons.species.particle_pusher = "gyrokinetic"
ions.species.particle_pusher = "boris"

# Hot cathode source: Smirnov-style batch source in the central disk.
electrons.species.cathode_hot_source = 1
electrons.species.cathode_hot_rmin = 0.0
electrons.species.cathode_hot_rmax = Rinj
electrons.species.cathode_hot_center = [0.0, 0.0]
electrons.species.cathode_hot_interval = hot_cathode_interval
electrons.species.cathode_hot_npart_per_injection = N_hot_emit
electrons.species.cathode_hot_macro_weight = macro_weight
electrons.species.cathode_hot_energy_eV = hot_emission_energy_eV

# Cold cathode ion sink in the annulus Rinj..Rcold.
ions.species.cathode_cold_sink = 1
ions.species.cathode_cold_rmin = Rinj
ions.species.cathode_cold_rmax = Rcold
ions.species.cathode_cold_center = [0.0, 0.0]
ions.species.cathode_cold_interval = cold_cathode_interval
ions.species.cathode_cold_nremove = N_ion_remove

sim.initialize_warpx()

electron_pc = particle_containers.ParticleContainerWrapper("electrons")
ion_pc = particle_containers.ParticleContainerWrapper("ions")
boundary_buffer = particle_containers.ParticleBoundaryBufferWrapper()


def current_step():
    return sim.extension.warpx.getistep(lev=0)


def count_particles_in_hot_cathode(pc):
    x_tiles = pc.get_particle_x(level=0, copy_to_host=True)
    z_tiles = pc.get_particle_z(level=0, copy_to_host=True)

    count = 0
    r2_max = Rinj * Rinj
    for x_tile, z_tile in zip(x_tiles, z_tiles):
        if len(x_tile) == 0:
            continue
        r2 = x_tile * x_tile + z_tile * z_tile
        count += int(np.count_nonzero(r2 < r2_max))

    return count


def count_scraped_this_step(species_name, boundary):
    arrays = boundary_buffer.get_particle_scraped_this_step(
        species_name, boundary, "x", level=0
    )
    return sum(len(arr) for arr in arrays)


def count_outer_absorbed_this_step(species_name):
    return count_scraped_this_step(species_name, "eb")


def uniform_disk_positions(n_particles, radius):
    rr = radius * np.sqrt(np.random.random(n_particles))
    theta = 2.0 * np.pi * np.random.random(n_particles)
    x = rr * np.cos(theta)
    z = rr * np.sin(theta)
    y = np.zeros_like(x)
    return x, y, z


def uniform_ring_positions(n_particles, rmin, rmax):
    rr = np.sqrt(rmin * rmin + (rmax * rmax - rmin * rmin) * np.random.random(n_particles))
    theta = 2.0 * np.pi * np.random.random(n_particles)
    x = rr * np.cos(theta)
    z = rr * np.sin(theta)
    y = np.zeros_like(x)
    return x, y, z


def isotropic_seed_velocities(n_particles, energy_eV, mass):
    std = np.sqrt(2.0 * energy_eV * e / (3.0 * mass))
    ux = np.random.normal(0.0, std, n_particles)
    uy = np.random.normal(0.0, std, n_particles)
    uz = np.random.normal(0.0, std, n_particles)
    return ux, uy, uz


def directed_emission_velocities(n_particles, energy_eV, mass):
    speed = np.sqrt(2.0 * energy_eV * e / mass)
    ux = np.zeros(n_particles)
    uy = np.full(n_particles, speed)
    uz = np.zeros(n_particles)
    return ux, uy, uz


def inject_seed_plasma():
    x, y, z = uniform_disk_positions(seed_macro_count, seed_radius)
    ux_e, uy_e, uz_e = isotropic_seed_velocities(seed_macro_count, seed_energy_eV, m_e)
    ux_i, uy_i, uz_i = isotropic_seed_velocities(seed_macro_count, seed_energy_eV, m_i)
    w = np.full(seed_macro_count, macro_weight)

    electron_pc.add_particles(x=x, y=y, z=z, ux=ux_e, uy=uy_e, uz=uz_e, w=w)
    ion_pc.add_particles(x=x, y=y, z=z, ux=ux_i, uy=uy_i, uz=uz_i, w=w)


def inject_cold_secondary_electrons():
    global total_cold_secondary_emitted

    step = current_step()
    if step == 0 or step % cold_cathode_interval != 0:
        return

    x, y, z = uniform_ring_positions(N_cold_secondary_emit, Rinj, Rcold)
    ux, uy, uz = directed_emission_velocities(N_cold_secondary_emit, cold_secondary_energy_eV, m_e)
    w = np.full(N_cold_secondary_emit, macro_weight)
    electron_pc.add_particles(x=x, y=y, z=z, ux=ux, uy=uy, uz=uz, w=w)
    total_cold_secondary_emitted += N_cold_secondary_emit


def update_hot_emission_counter():
    global total_hot_emitted

    step = current_step()
    if step == 0 or step % hot_cathode_interval != 0:
        return

    total_hot_emitted += N_hot_emit


def update_loss_counters():
    global total_electron_outer_absorbed, total_ion_outer_absorbed

    electron_outer_absorbed_step = count_outer_absorbed_this_step("electrons")
    ion_outer_absorbed_step = count_outer_absorbed_this_step("ions")
    total_electron_outer_absorbed += electron_outer_absorbed_step
    total_ion_outer_absorbed += ion_outer_absorbed_step


def runtime_status():
    step = current_step()
    if step == 0 or step % diagnostic_period != 0:
        return

    ne = electron_pc.get_particle_count(local=False)
    ni = ion_pc.get_particle_count(local=False)
    ne_hot = count_particles_in_hot_cathode(electron_pc)
    ni_hot = count_particles_in_hot_cathode(ion_pc)
    print(
        "[penning] "
        f"step={step} "
        f"Ne={ne} "
        f"Ni={ni} "
        f"Ne_hot={ne_hot} "
        f"Ni_hot={ni_hot} "
        f"Ne_outer_abs={total_electron_outer_absorbed} "
        f"Ni_outer_abs={total_ion_outer_absorbed} "
        f"Nhot_total={total_hot_emitted} "
        f"Ncoldsec_total={total_cold_secondary_emitted}"
    )


with open(log_file, "w", encoding="utf-8") as f:
    f.write(
        "step ne ni ne_hot ni_hot "
        "ne_outer_abs ni_outer_abs "
        "nhot_total ncoldsec_total\n"
    )


def log_particle_counts():
    step = current_step()
    ne = electron_pc.get_particle_count(local=False)
    ni = ion_pc.get_particle_count(local=False)
    ne_hot = count_particles_in_hot_cathode(electron_pc)
    ni_hot = count_particles_in_hot_cathode(ion_pc)

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(
            f"{step} {ne} {ni} {ne_hot} {ni_hot} "
            f"{total_electron_outer_absorbed} {total_ion_outer_absorbed} "
            f"{total_hot_emitted} {total_cold_secondary_emitted}\n"
        )


inject_seed_plasma()
callbacks.installafterstep(inject_cold_secondary_electrons)
callbacks.installafterstep(update_hot_emission_counter)
callbacks.installafterstep(update_loss_counters)
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
    f"seed_macro_count={seed_macro_count}, "
    f"seed_radius={seed_radius:.6e}, "
    f"mass_ratio={m_i / m_e:.1f}, "
    f"hot_interval={hot_cathode_interval}, "
    f"N_hot_emit={N_hot_emit}, "
    f"cold_interval={cold_cathode_interval}, "
    f"N_ion_remove={N_ion_remove}, "
    f"N_cold_secondary_emit={N_cold_secondary_emit}, "
    f"gamma={gamma_secondary:.3f}"
)

start = time.time()
sim.step(max_steps)
end = time.time()

print(f"[penning] elapsed_time={end - start:.6f} s")
