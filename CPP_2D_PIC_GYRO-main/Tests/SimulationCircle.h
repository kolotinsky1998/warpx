//
// Created by Vladimir Smirnov on 06.11.2021.
//

#ifndef CPP_2D_PIC_SIMULATIONCIRCLE_H
#define CPP_2D_PIC_SIMULATIONCIRCLE_H


#include <iostream>
#include "../Tools/Grid.h"
#include "../Particles/Particles.h"
#include "../ParticleLeaveEmission/ParticleEmission.h"
#include "../Collisions/NeutralGas.h"
#include "../Collisions/Collision.h"
#include "../Field/PoissonSolver.h"
#include "../Interpolation/Interpolation.h"
#include "../Tools/Logger.h"
#include "../Tools/ParticlesLogger.h"
#include "../Collisions/NullCollisions.h"
#include "../ParticleLeaveEmission/ParticleLeave.h"
#include <array>
#include <cmath>
#define E_M 9.10938356e-31
#define EV 1.6021766208e-19
#define EPSILON_0 8.854187817620389e-12
#define K_B 1.380649e-23

using namespace std;

int anode_current(Particles& particles, scalar Radius, const array<scalar, 2> circle_center) {
    int Ntot = 0;
    for(int ptcl_idx = 0; ptcl_idx < particles.get_Ntot(); ptcl_idx++) {
        if (pow(particles.x[ptcl_idx] - circle_center[0], 2) +
            pow(particles.y[ptcl_idx] - circle_center[1], 2) >= pow(Radius, 2)) {
            Ntot++;
        }
    }
    return Ntot;
}

array<scalar, 3> mean_energy_xyz(Particles& particles) {
    array<scalar, 3> energy{};
    array<scalar, 3> vel{};
    for(int ptcl_idx = 0; ptcl_idx < particles.get_Ntot(); ptcl_idx++) {
        vel = particles.get_velocity(ptcl_idx);
        energy[0] += particles.get_mass() * pow(vel[0], 2) / (2 * EV) / particles.get_Ntot();
        energy[1] += particles.get_mass() * pow(vel[1], 2) / (2 * EV) / particles.get_Ntot();
        energy[2] += particles.get_mass() * pow(vel[2], 2) / (2 * EV) / particles.get_Ntot();
    }

    return energy;
}


void test_simulation_circle() {
    cout << "simulation start" << endl;

    /*****************************************************/
    // Grid init
    int Nx = 100, Ny = 100;
    scalar dx = 1e-5, dy = 1e-5;
    Grid grid(Nx, Ny, dx, dy);
    /*****************************************************/
    // Overall particle information
    int ptcls_per_cell = 10;
    scalar init_dens = 1e16;
    scalar radius_injection = 0.3 * Nx * dx;
    array<scalar, 2> center_injection = {0.5 * Nx * dx, 0.5 * Ny * dy};
    int Ntot = M_PI * pow(radius_injection / dx, 2) * ptcls_per_cell;
    scalar ptcls_per_macro = init_dens * M_PI * pow(radius_injection, 2) / Ntot;
    cout << "ptcls_per_macro: " << ptcls_per_macro << endl;
    cout << "num_of_macro_ptcls: " << Ntot << endl;
    cout << "debye radius: " << pow(EPSILON_0 * EV * 10 / (EV*EV*100*init_dens), 0.5) << endl;
    /*****************************************************/
    // Particle Init
    scalar m_e = E_M;
    Particles electrons(m_e, -1*EV, 0, ptcls_per_macro);
    scalar m_ion = 100 * m_e;
    Particles ions(m_ion, 1*EV, 0, ptcls_per_macro);
    /*****************************************************/
    // Particle initial injection
    int seed = 1;
    scalar init_energy = 0.1*EV;
    init_particle_emission(electrons, radius_injection, center_injection, Ntot, init_energy, seed+10);
    init_particle_emission(ions, radius_injection, center_injection, Ntot, init_energy, seed+2); //0.000872269
    /*****************************************************/
    // Set const magnetic field to particles
    array<scalar, 3> B = {0, 0, 10};
    electrons.set_const_magnetic_field(B);
    ions.set_const_magnetic_field(B);
    /*****************************************************/
    // Set particle leave on Anode
    scalar radius_anode = Nx * dx / 2;
    int Nr_anode = Nx / 2;
    array<scalar, 2> domain_center = {0.5 * Nx * dx, 0.5 * Ny * dy};
    /*****************************************************/
    // Ion cathode leave
    scalar I_total = 0.1; // Ampere
    scalar gamma = 0.1; //secondary emission coefficient
    scalar ion_radius_leave = 0.6 * Nx * dx;
    array<scalar, 2> ion_center_leave = {0.5 * Nx * dx, 0.5 * Ny * dy};
    scalar I_ion_leave = I_total / (1. + gamma);
    scalar dt_ion_leave = 5e-12;
    int Ntot_ion_leave = I_ion_leave * dt_ion_leave / EV / ptcls_per_macro;
    cout << "Ntot_ion_cathode: " << Ntot_ion_leave << endl;

    /*****************************************************/
    // Emission electron
    scalar energy_emission = 100 * EV;
    scalar I_electron_emission = gamma * I_ion_leave;
    scalar dt_electron_emission = 1e-12;
    int Ntot_electron_emission = I_electron_emission * dt_electron_emission / EV / ptcls_per_macro;
    cout << "Ntot_electron_emission: " << Ntot_electron_emission << endl;
    scalar electron_radius_emission = ion_radius_leave;
    array<scalar, 2> electron_center_emission = {0.5 * Nx * dx, 0.5 * Ny * dy};
    /*****************************************************/
    // Neutral gas init
    scalar n = 1.5e21, m_gas = m_ion, T_gas = 500;
    NeutralGas gas(n, m_gas, T_gas);
    /*****************************************************/
    // Collisions init
    scalar dt_collision_electron = 5e-11, dt_collision_ion = 5e-10;
    EnergyCrossSection elastic_electron_sigma("../Collisions/CrossSectionData/e-He_elastic_KARAT.txt");
    EnergyCrossSection ionization_sigma("../Collisions/CrossSectionData/e-He_ionization_KARAT.txt");
    EnergyCrossSection elastic_helium_sigma("../Collisions/CrossSectionData/He+-He_elastic.txt");

    ElectronNeutralElasticCollision electron_elastic(elastic_electron_sigma, dt_collision_electron, gas, electrons);
    Ionization ionization(ionization_sigma, dt_collision_electron, gas, electrons, ions);
    IonNeutralElasticCollision ion_elastic(elastic_helium_sigma, dt_collision_ion, gas, ions);

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
    // PIC cycle parameters
    int it_num = 0;
    scalar dt = 5e-14;
    int collision_step_electron = dt_collision_electron / dt;
    int collision_step_ion = dt_collision_ion / dt;
    int ion_leave_step = dt_ion_leave / dt;
    int electron_emission_step = dt_electron_emission / dt;
    int ion_step = sqrt(m_ion/m_e);
    cout << "it_num: " << it_num << endl;
    cout << "ion integration step: " << ion_step << endl;
    cout << "electron_emission_step: " << electron_emission_step << endl;
    cout << "ion_leave_step: " << ion_leave_step << endl;
    cout << "collision_step_ion: " << collision_step_ion << endl;
    cout << "collision_step_electron: " << collision_step_electron << endl;
    /*****************************************************/
    // Logger
    string phi_file = "phi.txt";
    Matrix phi_log(Nx, Ny);
    clear_file(phi_file);

    clear_file("mean_energy_xyz_electron.txt");
    clear_file("mean_energy_xyz_ion.txt");

    string electron_anode_current_file = "electron_anode_current.txt";
    clear_file(electron_anode_current_file);

    string ion_anode_current_file = "ion_anode_current.txt";
    clear_file(ion_anode_current_file);

    string Ntot_file = "Ntot_(iter).txt";
    int Ntot_step = 1000, energy_step = Ntot_step, pos_step = it_num/10, vel_step = it_num/10;
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
        //cout << "Charge Interpolation" << endl;
        LinearChargeInterpolation(rho_e, electrons, grid);
        if (it % ion_step == 0) {
            LinearChargeInterpolation(rho_i, ions, grid);
        }
        rho = rho_e + rho_i;
        /*****************************************************/
        // Electric Field calc
        //cout << "Electric Field calc" << endl;
        PoissonSolverCircle(phi, rho, grid, Nr_anode, tol, betta, max_iter, it_conv_check);
        compute_E(Ex, Ey, phi, grid);
        LinearFieldInterpolation(electrons, Ex, Ey, grid);
        if (it % ion_step == 0) {
            LinearFieldInterpolation(ions, Ex, Ey, grid);
        }
        /*****************************************************/
        // Pusher
        //cout << "Pusher" << endl;
        electrons.pusher(dt);
        if (it % ion_step == 0) {
            ions.pusher(dt*ion_step);
        }
        /*****************************************************/
        // Collisions
        //cout << "Collisions" << endl;
        if (it % collision_step_electron == 0) {
            electron_null_collisions(electron_elastic, ionization);
        }
        if (it % collision_step_ion == 0) {
            ion_null_collisions(ion_elastic);
        }
        /*****************************************************/
        // Anode leave
        //cout << "Anode leave" << endl;
        if (it % 100 == 0) {

            element_logging(it, "mean_energy_xyz_electron.txt", " ");
            array<scalar, 3> vel_electron = mean_energy_xyz(electrons);
            element_logging(vel_electron[0], "mean_energy_xyz_electron.txt", " ");
            element_logging(vel_electron[1], "mean_energy_xyz_electron.txt", " ");
            element_logging(vel_electron[2], "mean_energy_xyz_electron.txt", "\n");

            element_logging(it, "mean_energy_xyz_ion.txt", " ");
            array<scalar, 3> vel_ion = mean_energy_xyz(ions);
            element_logging(vel_ion[0], "mean_energy_xyz_ion.txt", " ");
            element_logging(vel_ion[1], "mean_energy_xyz_ion.txt", " ");
            element_logging(vel_ion[2], "mean_energy_xyz_ion.txt", "\n");

            element_logging(it, electron_anode_current_file, " ");
            element_logging(anode_current(electrons, radius_anode, domain_center), electron_anode_current_file);

            element_logging(it, ion_anode_current_file, " ");
            element_logging(anode_current(ions, radius_anode, domain_center), ion_anode_current_file);
        }

        particle_leave(electrons, grid, radius_anode, domain_center);
        if (it % ion_step == 0) {
            particle_leave(ions, grid, radius_anode, domain_center);
        }
        /*****************************************************/
        // ElectronEmission/IonLeave (domain center)
        //cout << "ElectronEmission/IonLeave (domain center)" << endl;
        if (it % electron_emission_step == 0) {
            particle_emission(electrons, electron_radius_emission, electron_center_emission, Ntot_electron_emission,
                    energy_emission);
        }
        if (it % ion_leave_step == 0) {
            some_particle_leave(ions, ion_radius_leave, ion_center_leave, Ntot_ion_leave);
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
    cout << "simulation finish" << endl;
}


#endif //CPP_2D_PIC_SIMULATIONCIRCLE_H
