#ifndef CPP_2D_PIC_TEST_COLLISION_H
#define CPP_2D_PIC_TEST_COLLISION_H

#include "../Collisions/EnergyCrossSection.h"
#include "../Collisions/Collision.h"
#include "SimulationCircleGyroNEW.h"
#include "../Collisions/NullCollisions.h"
#include <random>

#define E_M 9.10938356e-31
#define EV 1.6021766208e-19
#define EPSILON_0 8.854187817620389e-12
#define K_B 1.380649e-23

void test_collision() {
    scalar scale = 0.02;
    int gyro_coeff = 100;
    int it_num = 1e7;
    scalar ptcls_per_cell = 1;

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
    scalar m_ion = 1e3 * E_M; //6.6464731E-27; // He+

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
    // PIC cycle parameters
    int ion_step = dt_i / dt_e;
    int collision_step_electron = 1;
    int collision_step_ion = ion_step * 1;
    cout << "it_num: " << it_num << endl;
    cout << "ion integration step: " << ion_step << endl;
    cout << "collision_step_ion: " << collision_step_ion << endl;
    cout << "collision_step_electron: " << collision_step_electron << endl;
    assert(collision_step_electron > 0);
    assert(collision_step_ion > 0);
    assert(ion_step > 0);
    /*****************************************************/
    // Grid init
    int Nx = R_scaled * 2 / r_d, Ny = R_scaled * 2 / r_d;
    scalar dx = r_d, dy = r_d;
    Grid grid(Nx, Ny, dx, dy);
    cout << "Grid: (" << Nx << "; " << Ny << ")" << endl;
    /*****************************************************/
    // Neutral gas init
    NeutralGas gas(n_gas_scaled, m_ion, T_gas);
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
    GyroKineticParticles electrons(m_e, -1*EV, 1, ptcls_per_macro);
    Particles ions(m_ion, 1*EV, 1, ptcls_per_macro);
    /*****************************************************/
    // NEW!!!!
    // Ar: probs: 0.0265007 0.00784819 energy before: 100; energy_after_ionization: 84; collision_step_electron = 1dt
    // He: probs: 0.00179212 0.000930349 energy before: 100; energy_after_ionization: 75.5 collision_step_electron = 1dt
    // !!!!

    // NEW!!!!
    // Ar: probs: 0.132503 0.039241 energy before: 100; energy_after_ionization: 84; collision_step_electron = 1dt
    // He: probs: 0.00179212 0.000930349 energy before: 100; energy_after_ionization: 75.5 collision_step_electron = 1dt
    // !!!!

    // 45 - Ar
    // 2 - He


    EnergyCrossSection elastic_electron_sigma("../Collisions/CrossSectionData/e-Ar_elastic.txt");
    EnergyCrossSection ionization_sigma("../Collisions/CrossSectionData/e-Ar_ionization.txt");
    EnergyCrossSection elastic_helium_sigma("../Collisions/CrossSectionData/Ar+-Ar_elastic.txt");
    /*
    EnergyCrossSection elastic_electron_sigma("../Collisions/CrossSectionData/e-He_elastic_KARAT.txt");
    EnergyCrossSection ionization_sigma("../Collisions/CrossSectionData/e-He_ionization_KARAT.txt");
    EnergyCrossSection elastic_helium_sigma("../Collisions/CrossSectionData/He+-He_elastic.txt");
    */
    ElectronNeutralElasticCollision electron_elastic(elastic_electron_sigma, collision_step_electron * dt, gas, electrons);
    Ionization ionization(ionization_sigma, collision_step_electron * dt, gas, electrons, ions);
    IonNeutralElasticCollision ion_elastic(elastic_helium_sigma, collision_step_ion * dt, gas, ions);

    scalar prob_ion = ion_elastic.probability(0);
    cout << "prob_ion!!!!!!!!!!!!!!!!!! " << prob_ion << endl;
    /*****************************************************/
    scalar prob_1, prob_2, prob_3, random_number;
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_real_distribution<double> distribution(0.0,1.0);
    Ntot = electron_elastic.particles->get_Ntot();
    electrons.set_velocity(0, {0, sqrt(2*1e4*EV/m_e), 0});
    ions.set_velocity(0, {0, sqrt(2*1e3*EV/m_ion), 0});
    array<scalar, 3> vel = electrons.get_velocity(0);
    cout << "energy before: " << electrons.get_mass() * (vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]) / EV / 2 << endl;
    for (int ptcl_idx = 0; ptcl_idx < Ntot; ptcl_idx++) {
        //electrons.set_velocity(ptcl_idx, {0, sqrt(100*EV/m_e), 0});
        prob_1 = electron_elastic.probability(ptcl_idx);
        prob_2 = ionization.probability(ptcl_idx);
        prob_3 = ion_elastic.probability(ptcl_idx);
        //random_number = distribution(generator);
        random_number = prob_1;
        cout << "probs: " << prob_1 << " " << prob_2 << " " << endl;
        cout << "prob ion: " << prob_3 << endl;
        if (random_number < prob_1) {
            //cout << "elastic " << random_number << endl;
            electron_elastic.collision(ptcl_idx);
        } else if (random_number >= prob_1 and random_number < prob_1 + prob_2) {
            //cout << "ionization " << random_number << endl;
            ionization.collision(ptcl_idx);
        }

    }

    cout << electrons.get_Ntot() - Ntot << endl;
    vel = electrons.get_velocity(0);
    cout << "energy after: " << electrons.get_mass() * (vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]) / EV / 2 << endl;
    vel = electrons.get_velocity(1);
    cout << "energy after: " << electrons.get_mass() * (vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]) / EV / 2 << endl;

}

#endif //CPP_2D_PIC_TEST_COLLISION_H
