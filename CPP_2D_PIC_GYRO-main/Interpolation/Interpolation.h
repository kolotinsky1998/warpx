//
// Created by Vladimir Smirnov on 03.10.2021.
//

#ifndef CPP_2D_PIC_INTERPOLATION_H
#define CPP_2D_PIC_INTERPOLATION_H

#include "../Tools/Matrix.h"
#include "../Tools/Grid.h"
#include "../Particles/Particles.h"

void __LinearFieldInterpolation(scalar Ex[], scalar Ey[], const scalar x[], const scalar y[],
                              const scalar Ex_grid[], const scalar Ey_grid[], const Grid& grid, int Ntot);

void __LinearChargeInterpolation(scalar rho[], const scalar x[], const scalar y[], const Grid& grid,
                                 scalar charge, int Ntot);

void LinearFieldInterpolation(Particles& ptcl, const Matrix& Ex_grid, const Matrix& Ey_grid, const Grid& grid);

void LinearChargeInterpolation(Matrix& rho, const Particles& ptcl, const Grid& grid);


#endif //CPP_2D_PIC_INTERPOLATION_H
