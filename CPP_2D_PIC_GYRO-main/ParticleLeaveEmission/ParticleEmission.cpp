//
// Created by Vladimir Smirnov on 06.10.2021.
//

#include "ParticleEmission.h"
#include <cmath>
#include <random>


void particle_emission(Particles &particles, const scalar emission_radius, const array<scalar, 2>& circle_center,
                       const int Ntot_emission, const scalar energy) {
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_real_distribution<scalar> distribution(0.0, 1.0);
    scalar radius, theta;
    array<scalar, 2> pos{};
    array<scalar, 3> vel{};
    for(int ptcl_idx = 0; ptcl_idx < Ntot_emission; ptcl_idx++) {
        //radius = emission_radius * distribution(generator);
        //theta = 2*M_PI*distribution(generator);
        //pos[0] = radius * cos(theta) + circle_center[0];
        //pos[1] = radius * sin(theta) + circle_center[1];
        pos[0] = circle_center[0] - emission_radius + distribution(generator) * 2 * emission_radius;
        pos[1] = circle_center[1] - emission_radius + distribution(generator) * 2 * emission_radius;
        vel[0] = 0;
        vel[1] = 0;
        vel[2] = sqrt(2 * energy / particles.get_mass());
        particles.append(pos, vel);
    }
}

void particle_emission(Particles &particles, const scalar emission_radius_min, const scalar emission_radius_max,
                           const array<scalar, 2>& circle_center,
                           const int Ntot_emission, const scalar energy) {
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_real_distribution<scalar> distribution(0.0, 1.0);
    scalar radius, theta;
    array<scalar, 2> pos{};
    array<scalar, 3> vel{};
    for(int ptcl_idx = 0; ptcl_idx < Ntot_emission; ptcl_idx++) {
        radius = emission_radius_min + (emission_radius_max - emission_radius_min) * sqrt(distribution(generator));
        theta = 2 * M_PI * distribution(generator);
        pos[0] = radius * cos(theta) + circle_center[0];
        pos[1] = radius * sin(theta) + circle_center[1];
        vel[0] = 0;
        vel[1] = 0;
        vel[2] = sqrt(2 * energy / particles.get_mass());
        particles.append(pos, vel);
    }
}

void particle_emission(Particles &particles, const scalar emission_radius_min, const scalar emission_radius_max,
                       const array<scalar, 2>& circle_center,
                       const int Ntot_emission,
                       const array<scalar, 2>& energy,
                       const array<scalar, 2>& phi,
                       const array<scalar, 2>& alpha) {
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_real_distribution<scalar> distribution(0.0, 1.0);
    scalar radius, theta, alpha_rand, phi_rand, energy_rand;
    array<scalar, 2> pos{};
    array<scalar, 3> vel{};
    for(int ptcl_idx = 0; ptcl_idx < Ntot_emission; ptcl_idx++) {
        radius = emission_radius_min + (emission_radius_max - emission_radius_min) * sqrt(distribution(generator));
        theta = 2 * M_PI * distribution(generator);
        phi_rand = (phi[1] - phi[0]) * distribution(generator) + phi[0];
        alpha_rand = (alpha[1] - alpha[0]) * distribution(generator) + alpha[0];
        pos[0] = radius * cos(theta) + circle_center[0];
        pos[1] = radius * sin(theta) + circle_center[1];
        energy_rand = (energy[1] - energy[0]) * distribution(generator) + energy[0];
        vel[0] = sqrt(2 * energy_rand / particles.get_mass()) * sin(alpha_rand) * cos(phi_rand);
        vel[1] = sqrt(2 * energy_rand / particles.get_mass()) * sin(alpha_rand) * sin(phi_rand);
        vel[2] = sqrt(2 * energy_rand / particles.get_mass()) * cos(alpha_rand);
        particles.append(pos, vel);
    }
}

void init_particle_emission(Particles &particles, const scalar emission_radius, const array<scalar, 2> &circle_center,
                            const int Ntot_emission, const scalar energy, int seed) {
    std::random_device rd;
    std::default_random_engine generator(seed);
    std::uniform_real_distribution<scalar> distribution(0.0, 1.0);
    scalar _std = sqrt(2*energy/(3*particles.get_mass()));
    std::normal_distribution<scalar> distribution_normal(0.0, _std);
    scalar radius, theta;
    array<scalar, 2> pos{};
    array<scalar, 3> vel{};
    for(int ptcl_idx = 0; ptcl_idx < Ntot_emission; ptcl_idx++) {
        radius = emission_radius * sqrt(distribution(generator));
        theta = 2*M_PI*distribution(generator);
        pos[0] = radius * cos(theta) + circle_center[0];
        pos[1] = radius * sin(theta) + circle_center[1];
        //pos[0] = circle_center[0] - emission_radius + distribution(generator) * 2 * emission_radius;
        //pos[1] = circle_center[1] - emission_radius + distribution(generator) * 2 * emission_radius;

        while (true) {
            vel[0] = distribution_normal(generator);
            vel[1] = distribution_normal(generator);
            vel[2] = distribution_normal(generator);
            if (sqrt(vel[0]*vel[0] + vel[1]*vel[1] + vel[2]*vel[2]) < _std * 3) {
                break;
            }
        }

        particles.append(pos, vel);
    }
}
