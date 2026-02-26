#ifndef CPP_2D_PIC_SIMULATIONCIRCLEGYRONEW_H
#define CPP_2D_PIC_SIMULATIONCIRCLEGYRONEW_H


#include <iostream>
#include <fstream>
#include <cmath>
#include <array>
#include <cassert>
#include "../Tools/Grid.h"
#include "../Particles/Particles.h"
#include "../Particles/GyroKineticParticles.h"
#include "../ParticleLeaveEmission/ParticleEmission.h"
#include "../ParticleLeaveEmission/ParticleLeave.h"

#define E_M 9.10938356e-31
#define EV 1.6021766208e-19
#define EPSILON_0 8.854187817620389e-12
#define K_B 1.380649e-23

using namespace std;


scalar debye_radius(scalar n_e, scalar n_ion, scalar T_e, scalar T_ion) {
    scalar k_cgs = 1.38e-16;
    scalar q_cgs = 4.8e-10;
    scalar n_e_cgs = n_e / 1e6;
    scalar n_ion_cgs = n_ion / 1e6;
    scalar d_cgs = pow(4*M_PI*q_cgs*q_cgs*n_ion_cgs/(k_cgs*T_ion) + 4*M_PI*q_cgs*q_cgs*n_e_cgs/(k_cgs*T_e), -0.5);
    scalar d_si = d_cgs / 100.;
    return d_si;
}


void rho_filter(Matrix& rho, int iterations=2) {
    Matrix rho_f(rho.rows(), rho.columns());
    for(int it = 0; it < iterations; it++) {
        for(int i = 1; i < rho.rows() - 1 ; i++) {
            for(int j = 1; j < rho.columns() - 1; j++) {
                rho_f(i, j) = 0.5*(0.25 * (rho(i - 1, j) + rho(i + 1, j) + rho(i, j - 1) + rho(i, j + 1))) + 0.5*rho(i, j);
            }
        }
    }
    for(int i = 1; i < rho.rows() - 1; i++) {
        for(int j = 1; j < rho.columns() - 1; j++) {
            rho(i, j) = rho_f(i, j);
        }
    }
}

void rho_filter_new(Matrix& rho, scalar R) {
    Matrix rho_f_1(rho.rows(), rho.columns());
    for(int i = 1; i < rho.rows() - 1 ; i++) {
        for(int j = 1; j < rho.columns() - 1; j++) {
            if (pow(i - R, 2) + pow(j - R, 2) < pow(R, 2)) {
                rho_f_1(i, j) = 4 * rho(i, j) + 2 * (rho(i - 1, j) + rho(i + 1, j) + rho(i, j - 1) + rho(i, j + 1)) + (
                        rho(i - 1, j - 1) + rho(i + 1, j - 1) + rho(i - 1, j + 1) + rho(i + 1, j + 1));
                rho_f_1(i, j) = rho_f_1(i, j) / 16;
            }
        }
    }
    Matrix rho_f_2(rho.rows(), rho.columns());
    for(int i = 1; i < rho.rows() - 1 ; i++) {
        for(int j = 1; j < rho.columns() - 1; j++) {
            if (pow(i - R, 2) + pow(j - R, 2) < pow(R, 2)) {
                rho_f_2(i, j) = 20 * rho_f_1(i, j) +
                                (-1) * (rho_f_1(i - 1, j) + rho_f_1(i + 1, j) + rho(i, j - 1) + rho_f_1(i, j + 1)) +
                                (-1) * (
                                        rho_f_1(i - 1, j - 1) + rho_f_1(i + 1, j - 1) + rho_f_1(i - 1, j + 1) +
                                        rho_f_1(i + 1, j + 1));
                rho_f_2(i, j) = rho_f_2(i, j) / 12;
            }
        }
    }
    for(int i = 1; i < rho.rows() - 1; i++) {
        for(int j = 1; j < rho.columns() - 1; j++) {
            if (pow(i - R, 2) + pow(j - R, 2) < pow(R, 2)) {
                rho(i, j) = rho_f_2(i, j);
            }
        }
    }
}


void test_simulation_circle_gyro_new() {
    cout << "simulation gyro start" << endl;

    /*
     0 1386
    1000 1777
     */

    scalar scale = 0.02;
    int gyro_coeff = 100;
    int it_num = 1e4;
    scalar ptcls_per_cell = 1; //svs

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
    scalar m_ion = 500 * E_M; //6.6464731E-27; // He+

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
    scalar dt = dt_e;

    /*****************************************************/
    // Grid init
    int Nx = R_scaled * 2 / r_d, Ny = R_scaled * 2 / r_d;
    scalar dx = r_d, dy = r_d;
    Grid grid(Nx, Ny, dx, dy);
    cout << "Grid: (" << Nx << "; " << Ny << ")" << endl;
    /*****************************************************/
    // Overall particle information
    scalar init_dens = n_e_scaled / 10;
    scalar radius_injection = 0.3 * Nx * dx;
    array<scalar, 2> center_injection = {0.5 * Nx * dx, 0.5 * Ny * dy};
    int Ntot = M_PI * pow(radius_injection / dx, 2) * ptcls_per_cell;
    scalar ptcls_per_macro = init_dens * M_PI * pow(radius_injection, 2) / Ntot;
    cout << "ptcls_per_macro: " << ptcls_per_macro << endl;
    cout << "num_of_macro_ptcls: " << Ntot << endl;
    /*****************************************************/
    // Particle Init
    scalar m_e = E_M;
    GyroKineticParticles electrons(m_e, -1*EV, 0, ptcls_per_macro);
    Particles ions(m_ion, 1*EV, 0, ptcls_per_macro);
    /*****************************************************/
    // Particle initial injection
    int seed = 1;
    scalar init_energy = 0.1*EV;
    init_particle_emission(electrons, radius_injection, center_injection, Ntot, init_energy, seed);
    init_particle_emission(ions, radius_injection, center_injection, Ntot, init_energy, seed);
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
    int ion_leave_step = 500;
    scalar dt_ion_leave = ion_leave_step * dt; // !!!!!
    int Ntot_ion_leave = I_ion_leave * dt_ion_leave / EV / ptcls_per_macro;
    cout << "Ntot_ion_cathode: " << Ntot_ion_leave << endl;
    /*****************************************************/
    array<scalar, 2> electron_center_emission = {0.5 * Nx * dx, 0.5 * Ny * dy};

    // Electron emission cold cathode
    scalar energy_emission_cold = 100 * EV;
    scalar I_electron_emission_cold = gamma * I_ion_leave;
    int electron_emission_cold_step = 500;
    scalar dt_electron_emission_cold = electron_emission_cold_step * dt;
    int Ntot_electron_emission_cold = I_electron_emission_cold * dt_electron_emission_cold / EV / ptcls_per_macro;
    scalar electron_radius_emission_cold_min = ion_radius_leave_min;
    scalar electron_radius_emission_cold_max = ion_radius_leave_max;
    cout << "Ntot_electron_emission_cold: " << Ntot_electron_emission_cold << endl;
    /*****************************************************/
    // Electron emission hot cathode
    scalar energy_emission_hot = 100 * EV;
    scalar I_electron_emission_hot = I_hot_scaled;
    int electron_emission_hot_step = 500;
    scalar dt_electron_emission_hot = electron_emission_hot_step * dt;
    int Ntot_electron_emission_hot = I_electron_emission_hot * dt_electron_emission_hot / EV / ptcls_per_macro;
    scalar electron_radius_emission_hot_min = 0;
    scalar electron_radius_emission_hot_max = ion_radius_leave_min;
    cout << "Ntot_electron_emission_hot: " << Ntot_electron_emission_hot << endl;
    /*****************************************************/
    // Ag+ Pb+ injection
    scalar m_Ag = 1.7911901E-25; //kg
    scalar m_Pb = 3.4406366E-25; //kg
    scalar dt_Ag_Pb = 1. / (EV * B_scaled / m_Ag) / 100;
    Particles Ag(m_Ag, 1*EV, 0, 1);
    Particles Pb(m_Pb, 1*EV, 0, 1);
    Ag.set_const_magnetic_field(mf);
    Pb.set_const_magnetic_field(mf);
    scalar injection_radius_Ag_Pb = 0.005 * scale; // diameter - 1cm in real system
    scalar Ntot_injected_Ag_Pb = 10;
    int Ag_Pb_step = dt_Ag_Pb / dt;
    int inject_step_Ag_Pb = dt_Ag_Pb / dt * 100;
    int start_inject_Ag_Pb = 1;
    array<scalar, 2>  injection_center_Ag_Pb = {0.8 * Nx * dx, 0.5 * Ny * dy};
    vector<int> track_Ag_Pb;
    for(int i = 0; i < Ntot_injected_Ag_Pb; i++) {
        track_Ag_Pb.push_back(i);
    }
    //emission_energy_Ag_Pb = 20*EV;
    array<scalar, 2> energy_Ag_Pb = {0, 20*EV};
    array<scalar, 2> phi_Ag_Pb = {0, 2 * M_PI};
    array<scalar, 2> alpha_Ag_Pb = {0, M_PI/6};

    cout << "dt_Ag_Pb: " << dt_Ag_Pb << endl;
    cout << "Ag_Pb_step: " << Ag_Pb_step << endl;
    cout << "inject_step_Ag_Pb: " << inject_step_Ag_Pb << endl;
    cout << "start_inject_Ag_Pb: " << start_inject_Ag_Pb << endl;
    cout << "Ntot_injected_Ag_Pb: " << Ntot_injected_Ag_Pb << endl;

    /*****************************************************/
    // Neutral gas init
    NeutralGas gas(n_gas_scaled, m_ion, T_gas);
    /*****************************************************/
    // PIC cycle parameters
    int ion_step = dt_i / dt_e;
    int collision_step_electron = 5;
    int collision_step_ion = ion_step * 5;
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
    EnergyCrossSection elastic_electron_sigma("../Collisions/CrossSectionData/e-Ar_elastic.txt");
    EnergyCrossSection ionization_sigma("../Collisions/CrossSectionData/e-Ar_ionization.txt");
    EnergyCrossSection elastic_helium_sigma("../Collisions/CrossSectionData/Ar+-Ar_elastic.txt");

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
    string phi_file = "phi_hist/phi.txt";
    Matrix phi_log(Nx, Ny);
    clear_file(phi_file);

    string phi_oscillations_file = "phi_oscillations.txt";
    clear_file(phi_oscillations_file);

    string electron_anode_current_file = "electron_anode_current.txt";
    clear_file(electron_anode_current_file);

    string ion_anode_current_file = "ion_anode_current.txt";
    clear_file(ion_anode_current_file);

    string Ntot_file = "Ntot_(iter).txt";
    int Ntot_step = 1000, energy_step = Ntot_step, pos_step = 100000, vel_step = 100000;
    ParticlesLogger electrons_logger(electrons, "electrons");
    ParticlesLogger ions_logger(ions, "ions");

    ParticlesLogger electrons_traj(electrons, "electrons_traj");
    ParticlesLogger ions_traj(ions, "ions_traj");
    ParticlesLogger Ag_traj(Ag, "Ag_traj");
    ParticlesLogger Pb_traj(Pb, "Pb_traj");
    int pos_traj_step_electron = 1, pos_traj_step_ion = ion_step;
    vector<int> track_ptcls = {0, 1, 2};
    /*****************************************************/
    scalar time_charge_interp = 0, time_field_interp = 0, time_pusher = 0, time_col = 0, time_anode = 0;
    scalar time_cold_cathode = 0, time_hot_cathode = 0, time_log = 0;
    scalar tmp_start_time;
    /*****************************************************/
    // PIC cycle
    clock_t start = clock();
    //scalar start = omp_get_wtime();
    electrons.vel_pusher(-0.5*dt);
    ions.vel_pusher(-0.5*dt*ion_step);

    int Ntot_ionized = 0;
    int Ntot_cold_cathode_leave = 0;
    int Ntot_anode_leave = 0;
    int Ntot_hot_cathode_emission = 0;
    int Ntot_tmp;
    for (int it = 0; it < it_num; it++) {
        /*****************************************************/
        // Charge Interpolation
        //tmp_start_time = omp_get_wtime();
        LinearChargeInterpolation(rho_e, electrons, grid);
        if (it % ion_step == 0) {
            LinearChargeInterpolation(rho_i, ions, grid);
        }
        rho = rho_e + rho_i;
        //rho_filter(rho, 2);
        rho_filter_new(rho, Nr_anode);
        //time_charge_interp += (omp_get_wtime() - tmp_start_time);
        /*****************************************************/
        // Electric Field calc
        //tmp_start_time = omp_get_wtime();
        PoissonSolverCircle(phi, rho, grid, Nr_anode, tol, betta, max_iter, it_conv_check);
        compute_E(Ex, Ey, phi, grid);
        LinearFieldInterpolation(electrons, Ex, Ey, grid);
        if (it % ion_step == 0) {
            LinearFieldInterpolation(ions, Ex, Ey, grid);
        }
        //time_field_interp += (omp_get_wtime() - tmp_start_time);
        /*****************************************************/
        // Pusher
        //tmp_start_time = omp_get_wtime();
        electrons.pusher(dt);
        if (it % ion_step == 0) {
            ions.pusher(dt*ion_step);
        }
        //time_pusher += (omp_get_wtime() - tmp_start_time);
        /*****************************************************/
        // Collisions
        //tmp_start_time = omp_get_wtime();
        if (it % collision_step_electron == 0) {
            Ntot_tmp = electrons.get_Ntot();
            electron_null_collisions(electron_elastic, ionization);
            Ntot_ionized += electrons.get_Ntot() - Ntot_tmp;
        }
        if (it % collision_step_ion == 0) {
            ion_null_collisions(ion_elastic);
        }
        //time_col += (omp_get_wtime() - tmp_start_time);
        /*****************************************************/
        // Anode leave
        //tmp_start_time = omp_get_wtime();
        Ntot_tmp = electrons.get_Ntot();
        particle_leave(electrons, grid, radius_anode, domain_center);
        Ntot_anode_leave += electrons.get_Ntot() - Ntot_tmp;
        if (it % ion_step == 0) {
            particle_leave(ions, grid, radius_anode, domain_center);
        }
        //time_anode += (omp_get_wtime() - tmp_start_time);
        /*****************************************************/
        // ElectronEmission/IonLeave (Cold Cathode)
        //tmp_start_time = omp_get_wtime();
        if (it % electron_emission_cold_step == 0) {
            particle_emission(electrons, electron_radius_emission_cold_min, electron_radius_emission_cold_max,
                              electron_center_emission, Ntot_electron_emission_cold, energy_emission_cold);
        }
        if (it % ion_leave_step == 0) {
            Ntot_tmp = ions.get_Ntot();
            some_particle_leave(ions, ion_radius_leave_min, ion_radius_leave_max, ion_center_leave, Ntot_ion_leave);
            Ntot_cold_cathode_leave += ions.get_Ntot() - Ntot_tmp;
        }
        //time_cold_cathode += omp_get_wtime() - tmp_start_time;
        /*****************************************************/
        // ElectronEmission (Hot Cathode)
        //tmp_start_time = omp_get_wtime();
        if (it % electron_emission_hot_step == 0) {
            Ntot_tmp = electrons.get_Ntot();
            particle_emission(electrons, electron_radius_emission_hot_min, electron_radius_emission_hot_max,
                              electron_center_emission, Ntot_electron_emission_hot, energy_emission_hot);
            Ntot_hot_cathode_emission += electrons.get_Ntot() - Ntot_tmp;
        }
        //time_hot_cathode += (omp_get_wtime() - tmp_start_time);
        /*****************************************************/
        // Log Info
        //cout << "Log Info" << endl;
        //tmp_start_time = omp_get_wtime();
        if (it % 10000 == 0 /*and it > 2000000*/) {
            string phi_filename = "phi_hist/phi"+std::to_string(it)+".txt";
            clear_file(phi_filename);
            element_logging(phi, phi_filename);

            string rho_e_filename = "rho_e_hist/rho_e_"+std::to_string(it)+".txt";
            clear_file(rho_e_filename);
            element_logging(rho_e, rho_e_filename);

            string rho_i_filename = "rho_i_hist/rho_i_"+std::to_string(it)+".txt";
            clear_file(rho_i_filename);
            element_logging(rho_i, rho_i_filename);

        }

        if (it == start_inject_Ag_Pb) {
            particle_emission(Ag, 0, injection_radius_Ag_Pb, injection_center_Ag_Pb, Ntot_injected_Ag_Pb, energy_Ag_Pb,
                              phi_Ag_Pb, alpha_Ag_Pb);
            particle_emission(Pb, 0, injection_radius_Ag_Pb, injection_center_Ag_Pb, Ntot_injected_Ag_Pb, energy_Ag_Pb,
                              phi_Ag_Pb, alpha_Ag_Pb);
        }

        if (it % Ag_Pb_step == 0 and it >= start_inject_Ag_Pb) {
            LinearFieldInterpolation(Ag, Ex, Ey, grid);
            LinearFieldInterpolation(Pb, Ex, Ey, grid);
            Ag.pusher(Ag_Pb_step*dt);
            Pb.pusher(Ag_Pb_step*dt);
            Ag_traj.position_log(it, 1, track_Ag_Pb);
            Pb_traj.position_log(it, 1, track_Ag_Pb);
        }

        if (it % 10 == 0) {
            for(int k = 0; k < 40; k = k + 5) {
                if (k == 35) {
                    element_logging(phi(35, k), phi_oscillations_file);
                } else {
                    element_logging(phi(35, k), phi_oscillations_file, " ");
                }
            }
            //element_logging("", phi_oscillations_file);
        }

        electrons_logger.n_total_log(it, Ntot_step);
        ions_logger.n_total_log(it, Ntot_step);
        electrons_logger.mean_energy_log(it, energy_step);
        ions_logger.mean_energy_log(it, energy_step);

        electrons_logger.position_log(it, pos_step);
        ions_logger.position_log(it, pos_step);
        electrons_logger.velocity_log(it, vel_step);
        ions_logger.velocity_log(it, vel_step);

        electrons_traj.position_log(it, pos_traj_step_electron, track_ptcls);
        ions_traj.position_log(it, pos_traj_step_ion, track_ptcls);
        //time_log += (omp_get_wtime() - tmp_start_time);
        /*****************************************************/
        if (it % 10000 == 0) {
            cout << it << endl;
            cout << "time_charge_interp: " << time_charge_interp << endl;
            cout << "time_field_interp: " << time_field_interp << endl;
            cout << "time_pusher: " << time_pusher << endl;
            cout << "time_col: " << time_col << endl;
            cout << "time_anode: " << time_anode << endl;
            cout << "time_hot_cathode: " << time_hot_cathode << endl;
            cout << "time_cold_cathode: " << time_cold_cathode << endl;
            cout << "time_log: " << time_log << endl;
            cout << endl;
        }

        if (it % inject_step_Ag_Pb == 0) {
            cout << "iter: " << it << " Ntot_ionized: " << Ntot_ionized << endl;
            cout << "iter: " << it << " Ntot_cold_cathode_leave: " << Ntot_cold_cathode_leave << endl;
            cout << "iter: " << it << " Ntot_anode_leave: " << Ntot_anode_leave << endl;
            cout << "iter: " << it << " Ntot_hot_cathode_emission: " << Ntot_hot_cathode_emission << endl;
        }
        /*****************************************************/
    }

    cout << "time_charge_interp: " << time_charge_interp << endl;
    cout << "time_field_interp: " << time_field_interp << endl;
    cout << "time_pusher: " << time_pusher << endl;
    cout << "time_col: " << time_col << endl;
    cout << "time_anode: " << time_anode << endl;
    cout << "time_hot_cathode: " << time_hot_cathode << endl;
    cout << "time_cold_cathode: " << time_cold_cathode << endl;

    //scalar end = omp_get_wtime();
    clock_t end = clock();
    cout << "elapsed time: " << (scalar)(end - start) / CLOCKS_PER_SEC << endl;
    //cout << "elapsed time: " << (scalar)(end - start) << endl;

    cout << "simulation gyro end" << endl;
}

#endif //CPP_2D_PIC_SIMULATIONCIRCLEGYRONEW_H
