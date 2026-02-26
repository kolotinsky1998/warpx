//
// Created by Vladimir Smirnov on 06.10.2021.
//

#ifndef CPP_2D_PIC_PARTICLEEMISSION_H
#define CPP_2D_PIC_PARTICLEEMISSION_H


#include "../Particles/Particles.h"

void particle_emission(Particles& particles, const scalar emission_radius, const array<scalar, 2>& circle_center,
                       const int Ntot_emission, const scalar energy);

void particle_emission(Particles &particles, const scalar emission_radius_min, const scalar emission_radius_max,
                       const array<scalar, 2>& circle_center, const int Ntot_emission, const scalar energy);

void particle_emission(Particles &particles, const scalar emission_radius_min, const scalar emission_radius_max,
                       const array<scalar, 2>& circle_center, const int Ntot_emission, const array<scalar, 2>& energy,
                       const array<scalar, 2>& phi, const array<scalar, 2>& alpha);

void init_particle_emission(Particles& particles, const scalar emission_radius, const array<scalar, 2>& circle_center,
                            const int Ntot_emission, const scalar energy, int seed);


#endif //CPP_2D_PIC_PARTICLEEMISSION_H
