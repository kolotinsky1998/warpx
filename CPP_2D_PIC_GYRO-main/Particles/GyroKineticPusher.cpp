//
// Created by Vladimir Smirnov on 16.11.2021.
//

#include "GyroKineticPusher.h"
#include <cmath>

scalar DotProduct(const scalar v1[], const scalar v2[]) {
    return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
}

void GyroUpdateSingleVelocityBoris(scalar &vel_x_c, scalar &vel_y_c, scalar &vel_z_c, scalar &vel_x, scalar &vel_y, scalar &vel_z,
                                   const scalar Ex, const scalar Ey, const scalar Bx, const scalar By, const scalar Bz,
                                   const scalar dt, const scalar q, const scalar m) {
    int vel_dim = 3;
    scalar v_B_dot, E_B_dot, phi, v_perpend_norm;
    scalar E_B_cross[vel_dim], v_E[vel_dim], v[vel_dim], v_parallel[vel_dim], E_parallel[vel_dim], v_perpend[vel_dim];
    scalar E[] = {Ex, Ey, 0};
    scalar B[] = {Bx, By, Bz};

    scalar B_norm_2 = Bx*Bx + By*By + Bz*Bz;
    CrossProduct(E, B, E_B_cross);

    v_E[0] = E_B_cross[0] / B_norm_2;
    v_E[1] = E_B_cross[1] / B_norm_2;
    v_E[2] = E_B_cross[2] / B_norm_2;

    v[0] = vel_x - v_E[0];
    v[1] = vel_y - v_E[1];
    v[2] = vel_z - v_E[2];

    v_B_dot = DotProduct(v, B);
    E_B_dot = DotProduct(E, B);

    v_parallel[0] = B[0] * v_B_dot / B_norm_2;
    v_parallel[1] = B[1] * v_B_dot / B_norm_2;
    v_parallel[2] = B[2] * v_B_dot / B_norm_2;

    v_perpend[0] = v[0] - v_parallel[0];
    v_perpend[1] = v[1] - v_parallel[1];
    v_perpend[2] = v[2] - v_parallel[2];


    E_parallel[0] = B[0] * E_B_dot / B_norm_2;
    E_parallel[1] = B[1] * E_B_dot / B_norm_2;
    E_parallel[2] = B[2] * E_B_dot / B_norm_2;

    v_parallel[0] = v_parallel[0] + (q/m) * E_parallel[0] * dt;
    v_parallel[1] = v_parallel[1] + (q/m) * E_parallel[1] * dt;
    v_parallel[2] = v_parallel[2] + (q/m) * E_parallel[2] * dt;

    // right only for B = [0, 0, Bz] !!!!!
    v_perpend_norm = sqrt(pow(v_perpend[0], 2) + pow(v_perpend[1], 2) + pow(v_perpend[2], 2));
    phi = acos(v_perpend[0] / v_perpend_norm) + abs(q/m) * sqrt(B_norm_2) * dt;
    v_perpend[0] = v_perpend_norm * cos(phi);
    v_perpend[1] = v_perpend_norm * sin(phi);
    v_perpend[2] = 0;

    //update gyrocenter velocity
    vel_x_c = v_parallel[0] + v_E[0];
    vel_y_c = v_parallel[1] + v_E[1];
    vel_z_c = v_parallel[2] + v_E[2];

    //update velocity
    vel_x = vel_x_c + v_perpend[0];
    vel_y = vel_y_c + v_perpend[1];
    vel_z = vel_z_c + v_perpend[2];
}

void GyroUpdateVelocity(scalar vel_x_c[], scalar vel_y_c[], scalar vel_z_c[], scalar vel_x[], scalar vel_y[], scalar vel_z[],
                        const scalar Ex[], const scalar Ey[], const scalar Bx[], const scalar By[], const scalar Bz[],
                        const scalar dt, const scalar q, const scalar m, const int Ntot) {
    //#pragma omp for
    for (int ip = 0; ip < Ntot; ip++) {
        GyroUpdateSingleVelocityBoris(vel_x_c[ip], vel_y_c[ip], vel_z_c[ip], vel_x[ip], vel_y[ip], vel_z[ip], Ex[ip], Ey[ip], Bx[ip], By[ip], Bz[ip], dt, q, m);
    }
}

void GyroParticlePush(scalar pos_x[], scalar pos_y[], scalar vel_x_c[], scalar vel_y_c[], scalar vel_z_c[],
                      scalar vel_x[], scalar vel_y[], scalar vel_z[],
                      const scalar Ex[], const scalar Ey[], const scalar Bx[], const scalar By[], const scalar Bz[],
                      const scalar dt, const scalar q, const scalar m, const int Ntot) {
    //#pragma omp parallel num_threads(NUM_THREADS)
    //{
    GyroUpdateVelocity(vel_x_c, vel_y_c, vel_z_c, vel_x, vel_y, vel_z, Ex, Ey, Bx, By, Bz, dt, q, m, Ntot);
    UpdatePosition(pos_x, pos_y, vel_x_c, vel_y_c, dt, Ntot);
    //}
}
