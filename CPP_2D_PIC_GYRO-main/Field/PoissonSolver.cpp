//
// Created by Vladimir Smirnov on 11.09.2021.
//

#include "PoissonSolver.h"
#define EPSILON_0 8.854187817620389e-12


void InitDirichletConditions(Matrix& phi) {
    scalar phi_val = 0;
    int i, j;

    i = 0;
    for (j = 0; j < phi.columns(); j++) {
        phi(i, j) = phi_val;
    }

    i = phi.rows() - 1;
    for (j = 0; j < phi.columns(); j++) {
        phi(i, j) = phi_val;
    }

    j = 0;
    for (i = 0; i < phi.rows(); i++) {
        phi(i, j) = phi_val;
    }

    j = phi.columns();
    for (i = 0; i < phi.rows(); i++) {
        phi(i, j) = phi_val;
    }
}

bool convergence_check(const Matrix& phi, const Matrix& rho, const Grid& grid, const scalar tol) {
    scalar res;
    scalar dx2 = grid.dx*grid.dx;
    scalar dy2  = grid.dy*grid.dy;
    for (int i = 1; i < grid.Nx-1; i++) {
        for (int j = 1; j < grid.Ny-1; j++) {
            res = phi(i, j) - (rho(i, j) / EPSILON_0 +
                               (phi(i-1, j) + phi(i+1, j)) / dx2 +
                               (phi(i, j-1) + phi(i, j+1)) / dy2) / (2 / dx2 + 2 / dy2)
                    ;
            if (abs(res) > tol) {
                return false;
            }
        }
    }

    return true;
}

void PoissonSolver(Matrix& phi, const Matrix& rho, const Grid& grid, const scalar tol, const scalar betta,
                   const int max_iter, const int it_conv_check) {
    int Nx = grid.Nx;
    int Ny = grid.Ny;
    scalar dx2 = grid.dx*grid.dx;
    scalar dy2  = grid.dy*grid.dy;

    for (int it = 0; it < max_iter; it++) {
        for (int i = 1; i < Nx - 1; i++) {
            for (int j = 1; j < Ny - 1; j++) {
                /*
                phi(i, j) = betta * (rho(i, j) / EPSILON_0 +
                                     (phi(i-1, j) + phi(i+1, j)) / dx2 +
                                     (phi(i, j-1) + phi(i, j+1)) / dy2) / (2 / dx2 + 2 / dy2)
                            + phi(i, j) * (1 - betta);
                */
                phi(i, j) = (rho(i, j) / EPSILON_0 +
                             (phi(i-1, j) + phi(i+1, j)) / dx2 +
                             (phi(i, j-1) + phi(i, j+1)) / dy2) / (2 / dx2 + 2 / dy2);

            }
        }

        if (it % it_conv_check == 0) {
            if (convergence_check(phi, rho, grid, tol)) {
                return;
            }
        }
    }
}

void compute_E(Matrix &Ex, Matrix &Ey, const Matrix &phi, const Grid &grid) {
    // central difference, not right on walls
    for (int i = 0; i < grid.Nx; i++) {
        for (int j = 0; j < grid.Ny; j++) {
            if (i != 0 and i != grid.Nx-1)
                Ex(i, j) = (phi(i-1, j) - phi(i+1, j)) / (grid.dx*2);
            else if (i == 0)
                Ex(i, j) = (phi(i, j) - phi(i+1, j)) / grid.dx;
            else if (i == grid.Nx-1)
                Ex(i, j) = (phi(i-1, j) - phi(i, j)) / grid.dx;
            if (j != 0 and j != grid.Ny-1)
                Ey(i, j) = (phi(i, j-1) - phi(i, j+1)) / (grid.dy*2);
            else if (j == 0)
                Ey(i, j) = (phi(i, j) - phi(i, j+1)) / grid.dy;
            else if (j == grid.Ny-1)
                Ey(i, j) = (phi(i, j-1) - phi(i, j)) / grid.dy;
        }
    }
}
