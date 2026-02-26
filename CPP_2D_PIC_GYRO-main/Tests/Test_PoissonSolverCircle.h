//
// Created by Vladimir Smirnov on 06.11.2021.
//

#ifndef CPP_2D_PIC_TEST_POISSONSOLVERCIRCLE_H
#define CPP_2D_PIC_TEST_POISSONSOLVERCIRCLE_H


#include "../Tools/Matrix.h"
#include "../Tools/Grid.h"
#include "../Field/PoissonSolverCircle.h"

void test_PoissonSolverCircle() {
    int Nx = 11, Ny = 11, R = Nx / 2;
    scalar dx = 5e-2, dy = 5e-2;
    Matrix phi(Nx, Ny);
    Matrix rho(Nx, Ny);
    Grid grid(Nx, Ny, dx, dy);

    phi.print();
    InitDirichletConditionsCircle(phi, grid, R);

    phi.print();
    cout  << endl;

    PoissonSolverCircle(phi, rho, grid, R, 1e-4);

    phi.print();
    cout  << endl;

    PoissonSolverCircle(phi, rho, grid, R, 1e-5);

    phi.print();
    cout  << endl;

    PoissonSolverCircle(phi, rho, grid, R, 1e-6);

    phi.print();
}


#endif //CPP_2D_PIC_TEST_POISSONSOLVERCIRCLE_H
