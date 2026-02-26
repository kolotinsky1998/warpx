//
// Created by Vladimir Smirnov on 06.10.2021.
//

#ifndef CPP_2D_PIC_PARTICLELEAVE_H
#define CPP_2D_PIC_PARTICLELEAVE_H


#include "../Particles/Particles.h"

void particle_leave(Particles& particles, const Matrix &leave_cells, const Grid& grid);

void particle_leave(Particles& particles, const Grid& grid, const scalar Radius, const array<scalar, 2> circle_center);

void some_particle_leave(Particles& particles, const Matrix &leave_cells, const Grid& grid, const int Ntot_leave);

void some_particle_leave(Particles& particles, const scalar leave_radius, const array<scalar, 2> circle_center,
                         const int Ntot_leave);

void some_particle_leave(Particles& particles, const scalar leave_radius_min, const scalar leave_radius_max,
                         const array<scalar, 2> circle_center, const int Ntot_leave);

#endif //CPP_2D_PIC_PARTICLELEAVE_H
