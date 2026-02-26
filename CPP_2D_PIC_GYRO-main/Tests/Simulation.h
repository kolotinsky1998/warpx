#ifndef CPP_2D_PIC_SIMULATION_H
#define CPP_2D_PIC_SIMULATION_H


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

using namespace std;

int boundary_leave_stat(Particles& particles, Matrix& leave_cells, Grid& grid) {
    int cell_x, cell_y;
    int Ntot_leave = 0;
    for(int ptcl_idx = 0; ptcl_idx < particles.get_Ntot(); ptcl_idx++) {
        cell_x = particles.x[ptcl_idx] / grid.dx;
        cell_y = particles.y[ptcl_idx] / grid.dy;
        if (leave_cells(cell_x, cell_y) == 1) {
            Ntot_leave += 1;
        }
    }

    return Ntot_leave;
}

void simulation() {
    cout << "simulation start" << endl;

    /*****************************************************/
    // Grid init
    int Nx = 50, Ny = 50;
    scalar dx = 1e-4, dy = 1e-4;
    Grid grid(Nx, Ny, dx, dy);
    /*****************************************************/
    // Overall particles information
    int ptcls_per_cell = 10;
    scalar init_dens = 1e15;
    scalar radius_injection = 0.2 * Nx * dx;
    array<scalar, 2> center_injection = {0.5 * Nx * dx, 0.5 * Ny * dy};
    int Ntot = M_PI * pow(radius_injection / dx, 2) * ptcls_per_cell;
    scalar ptcls_per_macro = init_dens * M_PI * pow(radius_injection, 2) / Ntot;
    cout << "ptcls_per_macro: " << ptcls_per_macro << endl;
    cout << "num_of_macro_ptcls: " << Ntot << endl;
    /*****************************************************/
    // Particle Init
    scalar m_e = E_M;
    Particles electrons(m_e, -1*EV, 0, ptcls_per_macro);
    scalar m_ion = 100 * m_e;
    Particles ions(m_ion, 1*EV, 0, ptcls_per_macro);
    /*****************************************************/
    // Particle initial injection
    int seed = 1;
    scalar init_energy = 1*EV;
    init_particle_emission(electrons, radius_injection, center_injection, Ntot, init_energy, seed+10);
    init_particle_emission(ions, radius_injection, center_injection, Ntot, init_energy, seed+2); //0.000872269
    /*****************************************************/
    // Set const magnetic field to particles
    array<scalar, 3> B = {0, 0, 0.1};
    electrons.set_const_magnetic_field(B);
    ions.set_const_magnetic_field(B);
    /*****************************************************/
    // Set particle leave matrix (on boundaries)
    Matrix leave_cells(Nx, Ny);
    for(int i = 0; i < leave_cells.rows(); i++) {
        for(int j = 0; j < leave_cells.columns(); j++) {
            if ((i <= 1) or (i >= Nx - 2) or (j <= 1) or (j >= Ny - 2)) {
                leave_cells(i, j) = 1;
            }
        }
    }
    //leave_cells.print();
    /*****************************************************/
    // Emission electrons/Leave ions on Cathode
    scalar I_total = 0.1; // 100mA
    scalar gamma = 0.1; //secondary emission coefficient

    scalar radius_leave = 0.1 * Nx * dx;
    array<scalar, 2> center_leave = {0.5 * Nx * dx, 0.5 * Ny * dy};
    scalar I_ion_leave = I_total / (1 + gamma);
    scalar dt_ion_leave = 1e-10;
    int Ntot_ion_leave = I_ion_leave * dt_ion_leave / EV / ptcls_per_macro;
    cout << "Ntot_ion_cathode: " << Ntot_ion_leave << endl;

    scalar energy_emission = 10 * EV;
    scalar I_electron_emission = gamma * I_ion_leave; // 1000mA
    scalar dt_electron_emission = 1e-10;
    int Ntot_electron_emission = I_electron_emission * dt_electron_emission / EV / ptcls_per_macro;
    cout << "Ntot_electron_emission: " << Ntot_electron_emission << endl;

    scalar radius_emission = 0.1 * Nx * dx;
    array<scalar, 2> center_emission = {0.5 * Nx * dx, 0.5 * Ny * dy};
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
    scalar tol = 1e-4, betta = 1.93, max_iter = 1e6, it_conv_check = 10;
    InitDirichletConditions(phi);
    //PoissonSolver(phi, rho, grid, tol, betta, max_iter, it_conv_check);
    /*****************************************************/
    // PIC cycle parameters
    int it_num = 160000/10;
    scalar dt = 1e-12*10;
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

    string electron_leave_boundary_file = "electron_leave_boundary.txt";
    clear_file(electron_leave_boundary_file);

    string ion_leave_boundary_file = "ion_leave_boundary.txt";
    clear_file(ion_leave_boundary_file);

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
        //cout << "iter: " << it << endl;
        /*****************************************************/
        // Emission/leave
        /*****************************************************/
        // Charge Interpolation
        LinearChargeInterpolation(rho_e, electrons, grid);
        if (it % ion_step == 0) {
            LinearChargeInterpolation(rho_i, ions, grid);
        }
        rho = rho_e + rho_i;
        /*****************************************************/
        // Electric Field calc
        PoissonSolver(phi, rho, grid, tol, betta, max_iter, it_conv_check);
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
        // Leave on boundaries
        if (it % 10 == 0) {
            int Ntot_leave_electron = boundary_leave_stat(electrons, leave_cells, grid);
            element_logging(Ntot_leave_electron, electron_leave_boundary_file);

            int Ntot_leave_ion = boundary_leave_stat(ions, leave_cells, grid);
            element_logging(Ntot_leave_ion, ion_leave_boundary_file);
        }

        particle_leave(electrons, leave_cells, grid);
        if (it % ion_step == 0) {
            particle_leave(ions, leave_cells, grid);
        }
        /*****************************************************/
        // ElectronEmission/IonLeave (domain center)
        if (it % electron_emission_step == 0) {
            particle_emission(electrons, radius_emission, center_emission, Ntot_electron_emission, energy_emission);
        }
        if (it % ion_leave_step == 0) {
            some_particle_leave(ions, radius_leave, center_leave, Ntot_ion_leave);
        }
        /*****************************************************/
        // Log Info

        if (it % 1 == 0) {
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

        if (it == 1) {
            electrons_logger.position_log(it, 1);
            ions_logger.position_log(it, 1);
            electrons_logger.velocity_log(it, 1);
            ions_logger.velocity_log(it, 1);
        }
        electrons_logger.position_log(it, pos_step);
        ions_logger.position_log(it, pos_step);
        electrons_logger.velocity_log(it, vel_step);
        ions_logger.velocity_log(it, vel_step);
        /*****************************************************/

    }
    /*****************************************************/
    //scalar end = omp_get_wtime();
    clock_t end = clock();
    cout << "elapsed time: " << (scalar)(end - start) / CLOCKS_PER_SEC << endl;
    //cout << "elapsed time: " << (scalar)(end - start) << endl;
    cout << "simulation finish" << endl;
}


#endif //CPP_2D_PIC_SIMULATION_H
