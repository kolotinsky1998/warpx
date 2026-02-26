
//
// Created by Vladimir Smirnov on 11.09.2021.
//

#ifndef CPP_2D_PIC_POISSONSOLVER_H
#define CPP_2D_PIC_POISSONSOLVER_H


#include "../Tools/Matrix.h"
#include "../Tools/Grid.h"
#include <cmath>

void InitDirichletConditions(Matrix& phi);

bool convergence_check(const Matrix& phi, const Matrix& rho, const Grid& grid, const scalar tol);

void PoissonSolver(Matrix& phi, const Matrix& rho, const Grid& grid, const scalar tol, scalar betta = 1.93, int max_iter = 1e6,
        int it_conv_check=1);

void compute_E(Matrix& Ex, Matrix& Ey, const Matrix& phi, const Grid& grid);


#endif //CPP_2D_PIC_POISSONSOLVER_H
