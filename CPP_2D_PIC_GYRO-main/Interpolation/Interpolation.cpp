//
// Created by Vladimir Smirnov on 03.10.2021.
//

#include "Interpolation.h"
#include <cmath>

void __LinearFieldInterpolation(scalar *Ex, scalar *Ey, const scalar *x, const scalar *y, const scalar *Ex_grid,
                                const scalar *Ey_grid, const Grid &grid, int Ntot) {
    int cell_x, cell_y, Ny=grid.Ny;
    scalar hx, hy;
    //#pragma omp parallel for private(hz, hr, cell_z, cell_r) num_threads(NUM_THREADS)
    for (int i = 0; i < Ntot; i++) {
        cell_x = floor(x[i]/grid.dx);
        cell_y = floor(y[i]/grid.dy);
        hx = (x[i] - cell_x*grid.dx) / grid.dx;
        hy = (y[i] - cell_y*grid.dy) / grid.dy;

        Ex[i] = Ex_grid[cell_x*Ny + cell_y] * (1 - hx) * (1 - hy);
        Ex[i] += Ex_grid[(cell_x+1)*Ny + cell_y] * hx * (1 - hy);
        Ex[i] += Ex_grid[(cell_x+1)*Ny + cell_y+1] * hx * hy;
        Ex[i] += Ex_grid[cell_x*Ny + cell_y+1] * (1 - hx) * hy;

        Ey[i] = Ey_grid[cell_x*Ny + cell_y] * (1 - hx) * (1 - hy);
        Ey[i] += Ey_grid[(cell_x+1)*Ny + cell_y] * hx * (1 - hy);
        Ey[i] += Ey_grid[(cell_x+1)*Ny + cell_y+1] * hx * hy;
        Ey[i] += Ey_grid[cell_x*Ny + cell_y+1] * (1 - hx) * hy;
    }
}

void __LinearChargeInterpolation(scalar *rho, const scalar *x, const scalar *y, const Grid &grid, scalar charge, int Ntot) {
    int cell_x, cell_y, Ny = grid.Ny;
    scalar hx, hy;
    for (int i = 0; i < grid.Nx*grid.Ny; i++) {
        rho[i] = 0;
    }
    //#pragma omp parallel for private(hz, hr, cell_z, cell_r) num_threads(NUM_THREADS)
    for (int i = 0; i < Ntot; i++) {
        cell_x = floor(x[i] / grid.dx);
        cell_y = floor(y[i] / grid.dy);
        hx = (x[i] - cell_x * grid.dx) / grid.dx;
        hy = (y[i] - cell_y * grid.dy) / grid.dy;
        /*
        if (cell_x > grid.Nx - 1 or cell_x < 0 or cell_y > grid.Ny - 1 or cell_y < 0) {
            cout << "ptcl_idx = " << i << "; cell_x = " << cell_x << "; cell_y = " << cell_y << "; Ntot = " << Ntot << endl;
            break;
        }
        */
        //#pragma omp atomic
        rho[cell_x * Ny + cell_y] += (charge * (1 - hx) * (1 - hy) / (grid.dx*grid.dy));
        //#pragma omp atomic
        rho[(cell_x + 1) * Ny + cell_y] += (charge * hx * (1 - hy) / (grid.dx*grid.dy));
        //#pragma omp atomic
        rho[(cell_x + 1) * Ny + cell_y + 1] += (charge * hx * hy / (grid.dx*grid.dy));
        //#pragma omp atomic
        rho[cell_x * Ny + cell_y + 1] += (charge * (1 - hx) * hy / (grid.dx*grid.dy));
    }
}

void LinearFieldInterpolation(Particles &ptcl, const Matrix &Ex_grid, const Matrix &Ey_grid, const Grid &grid) {
    __LinearFieldInterpolation(ptcl.Ex.data(), ptcl.Ey.data(), ptcl.x.data(), ptcl.y.data(), Ex_grid.data_const_ptr(),
                               Ey_grid.data_const_ptr(), grid, ptcl.get_Ntot());
}

void LinearChargeInterpolation(Matrix &rho, const Particles &ptcl, const Grid &grid) {
    __LinearChargeInterpolation(rho.data_ptr(), ptcl.x.data(), ptcl.y.data(), grid,
                                ptcl.get_charge()*ptcl.get_ptcls_per_macro(),
                                ptcl.get_Ntot());
}
