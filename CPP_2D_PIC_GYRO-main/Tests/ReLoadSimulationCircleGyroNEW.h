#ifndef CPP_2D_PIC_RELOADSIMULATIONCIRCLEGYRONEW_H
#define CPP_2D_PIC_RELOADSIMULATIONCIRCLEGYRONEW_H


#include <iostream>
#include <cmath>
#include <array>
#include <cassert>
#include "../Tools/Grid.h"
#include "../Particles/Particles.h"
#include "../Particles/GyroKineticParticles.h"
#include "../ParticleLeaveEmission/ParticleEmission.h"
#include "../ParticleLeaveEmission/ParticleLeave.h"
#include "SimulationCircleGyroNEW.h"
#include "../Tools/ParticlesLoad.h"

#define E_M 9.10938356e-31
#define EV 1.6021766208e-19
#define EPSILON_0 8.854187817620389e-12
#define K_B 1.380649e-23

using namespace std;


void test_reload_simulation_circle_gyro_new() {
    cout << "test_reload_simulation_circle_gyro_new() start" << endl;

    scalar scale = 0.02;
    int gyro_coeff = 10;
    int it_num = 1e1;
    scalar ptcls_per_cell = 0.1;

    // real system
    scalar R = 0.26;
    scalar B = 0.1;
    scalar n_e = 1e15; // m^(-3)
    scalar n_i = 1e15; // m^(-3)
    scalar T_e = 10 * 11604.52500617; //10eV
    scalar T_i = 10 * 11604.52500617; //10eV
    scalar I_hot = 6; // Ampere
    scalar I_cold = 4; // Ampere
    scalar p = 4 * 1e-3 * 133; // 4mTorr to Pascal
    scalar T_gas = 500; // Kelvin
    scalar n_gas = p / (K_B * T_gas);
    scalar m_ion = 100 * E_M; //6.6464731E-27; // He+

    //scaled system
    scalar R_scaled = R * scale;
    scalar B_scaled = B / scale;
    scalar I_hot_scaled = I_hot * scale;
    scalar I_cold_scaled = I_cold * scale;
    scalar n_gas_scaled = n_gas / scale;
    scalar n_e_scaled = n_e / scale;
    scalar n_i_scaled = n_i / scale;

    //Parameters of scaled system
    scalar dt_e = 1 / (EV * B_scaled / E_M) / 10 * gyro_coeff;
    scalar dt_i = 1 / (EV * B_scaled / m_ion) / 10;
    scalar r_d = debye_radius(n_e_scaled, n_i_scaled, T_e, T_i) * 2;
    cout << "dt_e = " << dt_e << endl;
    cout << "dt_i = " << dt_i << endl;
    cout << "debye_radius = " << r_d << " at energy = " << T_e / 11604.52500617 << "EV" << endl;

    /*****************************************************/
    // Grid init
    int Nx = R_scaled * 2 / r_d, Ny = R_scaled * 2 / r_d;
    scalar dx = r_d, dy = r_d;
    Grid grid(Nx, Ny, dx, dy);
    cout << "Grid: (" << Nx << "; " << Ny << ")" << endl;
    /*****************************************************/
    // Overall particle information
    scalar init_dens = n_e_scaled / 100;
    scalar radius_injection = 0.3 * Nx * dx;
    array<scalar, 2> center_injection = {0.5 * Nx * dx, 0.5 * Ny * dy};
    int Ntot = M_PI * pow(radius_injection / dx, 2) * ptcls_per_cell;
    scalar ptcls_per_macro = init_dens * M_PI * pow(radius_injection, 2) / Ntot;
    cout << "ptcls_per_macro: " << ptcls_per_macro << endl;
    /*****************************************************/
    // Particle Init
    scalar m_e = E_M;
    int iteration_start = 100000, iteration_end = 110000;
    // Electron Init
    string electron_pos_path = "init_electrons_positions.txt";
    string electron_vel_path = "init_electrons_velocities.txt";
    GyroKineticParticles electrons(m_e, -1*EV, 0, ptcls_per_macro);
    ParticlesLoad load_electrons(electrons, electron_pos_path, electron_vel_path, iteration_start, iteration_end);
    // Ion init
    string ion_pos_path = "init_ions_positions.txt";
    string ion_vel_path = "init_ions_velocities.txt";
    Particles ions(m_ion, 1*EV, 0, ptcls_per_macro);
    ParticlesLoad load_ions(ions, ion_pos_path, ion_vel_path, iteration_start, iteration_end);
    /*****************************************************/
    // Set const magnetic field to particles
    array<scalar, 3> mf = {0, 0, B_scaled};
    electrons.set_const_magnetic_field(mf);
    ions.set_const_magnetic_field(mf);
    /*****************************************************/
    // Set particle leave on Anode
    scalar radius_anode = (Nx - 1) * dx / 2;
    int Nr_anode = (Nx - 1) / 2;
    array<scalar, 2> domain_center = {0.5 * Nx * dx, 0.5 * Ny * dy};
    /*****************************************************/
    // Ion cathode leave
    scalar gamma = 0.1; //secondary emission coefficient
    scalar ion_radius_leave_min = 0.01 * scale; // 1cm in real system
    scalar ion_radius_leave_max = 0.15 * scale; // 15cm in real system
    array<scalar, 2> ion_center_leave = {0.5 * Nx * dx, 0.5 * Ny * dy};
    scalar I_ion_leave = I_cold_scaled / (1. + gamma);
    scalar dt_ion_leave = 5e-9; // !!!!!
    int Ntot_ion_leave = I_ion_leave * dt_ion_leave / EV / ptcls_per_macro;
    cout << "Ntot_ion_cathode: " << Ntot_ion_leave << endl;
    /*****************************************************/
    array<scalar, 2> electron_center_emission = {0.5 * Nx * dx, 0.5 * Ny * dy};

    // Electron emission cold cathode
    scalar energy_emission_cold = 100 * EV;
    scalar I_electron_emission_cold = gamma * I_ion_leave;
    scalar dt_electron_emission_cold = 5e-9;
    int Ntot_electron_emission_cold = I_electron_emission_cold * dt_electron_emission_cold / EV / ptcls_per_macro;
    scalar electron_radius_emission_cold_min = ion_radius_leave_min;
    scalar electron_radius_emission_cold_max = ion_radius_leave_max;
    cout << "Ntot_electron_emission_cold: " << Ntot_electron_emission_cold << endl;
    /*****************************************************/
    // Electron emission hot cathode
    scalar energy_emission_hot = 100 * EV;
    scalar I_electron_emission_hot = I_hot_scaled;
    scalar dt_electron_emission_hot = 5e-9;
    int Ntot_electron_emission_hot = I_electron_emission_hot * dt_electron_emission_hot / EV / ptcls_per_macro;
    scalar electron_radius_emission_hot_min = 0;
    scalar electron_radius_emission_hot_max = ion_radius_leave_min;
    cout << "Ntot_electron_emission_hot: " << Ntot_electron_emission_hot << endl;
    /*****************************************************/
    // Neutral gas init
    NeutralGas gas(n_gas_scaled, m_ion, T_gas);
    /*****************************************************/
    // PIC cycle parameters
    scalar dt = dt_e;
    scalar dt_collision_electron = 5e-11, dt_collision_ion = 1e-10;
    int collision_step_electron = dt_collision_electron / dt;
    int collision_step_ion = dt_collision_ion / dt;
    int ion_leave_step = dt_ion_leave / dt;
    int electron_emission_cold_step = dt_electron_emission_cold / dt;
    int electron_emission_hot_step = dt_electron_emission_hot / dt;
    int ion_step = dt_i / dt_e;
    cout << "it_num: " << it_num << endl;
    cout << "ion integration step: " << ion_step << endl;
    cout << "electron_emission_cold_step: " << electron_emission_cold_step << endl;
    cout << "electron_emission_hot_step: " << electron_emission_hot_step << endl;
    cout << "ion_leave_step: " << ion_leave_step << endl;
    cout << "collision_step_ion: " << collision_step_ion << endl;
    cout << "collision_step_electron: " << collision_step_electron << endl;
    assert(collision_step_electron > 0);
    assert(collision_step_ion > 0);
    assert(ion_leave_step > 0);
    assert(electron_emission_cold_step > 0);
    assert(electron_emission_hot_step > 0);
    assert(ion_step > 0);
    /*****************************************************/
    // Collisions init
    EnergyCrossSection elastic_electron_sigma("../Collisions/CrossSectionData/e-He_elastic_KARAT.txt");
    EnergyCrossSection ionization_sigma("../Collisions/CrossSectionData/e-He_ionization_KARAT.txt");
    EnergyCrossSection elastic_helium_sigma("../Collisions/CrossSectionData/He+-He_elastic.txt");

    ElectronNeutralElasticCollision electron_elastic(elastic_electron_sigma, collision_step_electron * dt, gas, electrons);
    Ionization ionization(ionization_sigma, collision_step_electron * dt, gas, electrons, ions);
    IonNeutralElasticCollision ion_elastic(elastic_helium_sigma, collision_step_ion * dt, gas, ions);
    /*****************************************************/
    // Init for sum of electron and ion rho
    Matrix rho(Nx, Ny);
    Matrix rho_e(Nx, Ny);
    Matrix rho_i(Nx, Ny);
    /*****************************************************/
    // Electric Field Init
    Matrix phi(Nx, Ny);
    Matrix Ex(Nx, Ny);
    Matrix Ey(Nx, Ny);
    scalar tol = 1e-4, betta = 1.93, max_iter = 1e6, it_conv_check = 100;
    InitDirichletConditionsCircle(phi, grid, Nr_anode);
    /*****************************************************/
    // Logger
    string phi_file = "phi.txt";
    Matrix phi_log(Nx, Ny);
    clear_file(phi_file);

    string electron_anode_current_file = "electron_anode_current.txt";
    clear_file(electron_anode_current_file);

    string ion_anode_current_file = "ion_anode_current.txt";
    clear_file(ion_anode_current_file);

    string Ntot_file = "Ntot_(iter).txt";
    int Ntot_step = 1000, energy_step = Ntot_step, pos_step = it_num/100, vel_step = it_num/100;
    ParticlesLogger electrons_logger(electrons, "electrons");
    ParticlesLogger ions_logger(ions, "ions");
    /*****************************************************/
    // PIC cycle
    clock_t start = clock();
    //scalar start = omp_get_wtime();
    electrons.vel_pusher(-0.5*dt);
    ions.vel_pusher(-0.5*dt*ion_step);
    for (int it = 0; it < it_num; it++) {
        /*****************************************************/
        // Charge Interpolation
        LinearChargeInterpolation(rho_e, electrons, grid);
        if (it % ion_step == 0) {
            LinearChargeInterpolation(rho_i, ions, grid);
        }
        rho = rho_e + rho_i;
        rho_filter(rho, 2);
        /*****************************************************/
        // Electric Field calc
        PoissonSolverCircle(phi, rho, grid, Nr_anode, tol, betta, max_iter, it_conv_check);
        compute_E(Ex, Ey, phi, grid);
        LinearFieldInterpolation(electrons, Ex, Ey, grid);
        if (it % ion_step == 0) {
            LinearFieldInterpolation(ions, Ex, Ey, grid);
        }
        /*****************************************************/
        // Pusher
        electrons.pusher(dt);
        if (it % ion_step == 0) {
            ions.pusher(dt*ion_step);
        }
        /*****************************************************/
        // Collisions
        if (it % collision_step_electron == 0) {
            electron_null_collisions(electron_elastic, ionization);
        }
        if (it % collision_step_ion == 0) {
            ion_null_collisions(ion_elastic);
        }
        /*****************************************************/
        // Anode leave
        particle_leave(electrons, grid, radius_anode, domain_center);
        if (it % ion_step == 0) {
            particle_leave(ions, grid, radius_anode, domain_center);
        }
        /*****************************************************/
        // ElectronEmission/IonLeave (Cold Cathode)
        if (it % electron_emission_cold_step == 0) {
            particle_emission(electrons, electron_radius_emission_cold_min, electron_radius_emission_cold_max,
                              electron_center_emission, Ntot_electron_emission_cold, energy_emission_cold);
        }
        if (it % ion_leave_step == 0) {
            some_particle_leave(ions, ion_radius_leave_min, ion_radius_leave_max, ion_center_leave, Ntot_ion_leave);
        }
        /*****************************************************/
        // ElectronEmission (Hot Cathode)
        if (it % electron_emission_hot_step == 0) {
            particle_emission(electrons, electron_radius_emission_hot_min, electron_radius_emission_hot_max,
                              electron_center_emission, Ntot_electron_emission_hot, energy_emission_hot);
        }
        /*****************************************************/
        // Log Info
        //cout << "Log Info" << endl;
        if (it % 10000 == 0) {
            string phi_filename = "phi"+std::to_string(it)+".txt";
            clear_file(phi_filename);
            element_logging(phi, phi_filename);

            string rho_e_filename = "rho_e_"+std::to_string(it)+".txt";
            clear_file(rho_e_filename);
            element_logging(rho_e, rho_e_filename);

            string rho_i_filename = "rho_i_"+std::to_string(it)+".txt";
            clear_file(rho_i_filename);
            element_logging(rho_i, rho_i_filename);

        }

        electrons_logger.n_total_log(it, Ntot_step);
        ions_logger.n_total_log(it, Ntot_step);
        electrons_logger.mean_energy_log(it, energy_step);
        ions_logger.mean_energy_log(it, energy_step);
        /*****************************************************/
    }

    //scalar end = omp_get_wtime();
    clock_t end = clock();
    cout << "elapsed time: " << (scalar)(end - start) / CLOCKS_PER_SEC << endl;
    //cout << "elapsed time: " << (scalar)(end - start) << endl;

    cout << "test_reload_simulation_circle_gyro_new() end" << endl;
}

#endif //CPP_2D_PIC_RELOADSIMULATIONCIRCLEGYRONEW_H
