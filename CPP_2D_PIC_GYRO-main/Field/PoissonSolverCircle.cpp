//
// Created by Vladimir Smirnov on 06.11.2021.
//

#include "PoissonSolverCircle.h"
#include <cmath>
#include <cassert>
#define EPSILON_0 8.854187817620389e-12


void InitDirichletConditionsCircle(Matrix &phi, const Grid &grid, const int R) {
    assert(grid.dx == grid.dy);

    scalar phi_val = 0;

    for(int i = 0; i < phi.rows(); i++) {
        for(int j = 0; j < phi.columns(); j++) {
            if (pow(i - R, 2) + pow(j - R, 2) >= pow(R, 2)) {
                phi(i, j) = phi_val;
            }
        }
    }
}

bool convergence_check_circle(const Matrix &phi, const Matrix &rho, const Grid &grid, const int R, const scalar tol) {
    scalar res;
    scalar dx2 = grid.dx*grid.dx;
    scalar dy2  = grid.dy*grid.dy;

    for(int i = 0; i < phi.rows(); i++) {
        for(int j = 0; j < phi.columns(); j++) {
            if (pow(i - R, 2) + pow(j - R, 2) < pow(R, 2)) {
                res = phi(i, j) - (rho(i, j) / EPSILON_0 +
                                   (phi(i-1, j) + phi(i+1, j)) / dx2 +
                                   (phi(i, j-1) + phi(i, j+1)) / dy2) / (2 / dx2 + 2 / dy2);
                if (abs(res) > tol) {
                    return false;
                }
            }
        }
    }

    return true;
}

void PoissonSolverCircle(Matrix &phi, const Matrix &rho, const Grid &grid, const int R, const scalar tol, scalar betta,
                         int max_iter, int it_conv_check) {
    int Nx = grid.Nx;
    int Ny = grid.Ny;
    scalar dx2 = grid.dx*grid.dx;
    scalar dy2  = grid.dy*grid.dy;

    for (int it = 0; it < max_iter; it++) {
        for(int i = 0; i < Nx; i++) {
            for(int j = 0; j < Ny; j++) {
                if (pow(i - R, 2) + pow(j - R, 2) < pow(R, 2)) {

                    phi(i, j) = betta * (rho(i, j) / EPSILON_0 +
                                        (phi(i-1, j) + phi(i+1, j)) / dx2 +
                                        (phi(i, j-1) + phi(i, j+1)) / dy2) / (2 / dx2 + 2 / dy2)
                                + phi(i, j) * (1 - betta);
                    /*
                    phi(i, j) = (rho(i, j) / EPSILON_0 +
                                 (phi(i-1, j) + phi(i+1, j)) / dx2 +
                                 (phi(i, j-1) + phi(i, j+1)) / dy2) / (2 / dx2 + 2 / dy2);
                    */
                }
            }
        }

        if (it % it_conv_check == 0) {
            if (convergence_check_circle(phi, rho, grid, R, tol)) {
                return;
            }
        }
    }
}
