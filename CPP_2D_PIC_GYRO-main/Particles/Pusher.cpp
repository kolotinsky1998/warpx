//
// Created by Vladimir Smirnov on 02.10.2021.
//

#include "Pusher.h"
#include <iostream>
//#include <omp.h>
//#define NUM_THREADS 100


void CrossProduct(const scalar v1[], const scalar v2[], scalar result[]) {
    result[0] = v1[1] * v2[2] - v1[2] * v2[1];
    result[1] = v1[2] * v2[0] - v1[0] * v2[2];
    result[2] = v1[0] * v2[1] - v1[1] * v2[0];
}

void UpdateSingleVelocityBoris(scalar &vel_x, scalar &vel_y, scalar &vel_z, const scalar Ex, const scalar Ey,
                               const scalar Bx, const scalar By, const scalar Bz, const scalar dt, const scalar q,
                               const scalar m) {
    int vel_dim = 3;
    scalar t[vel_dim], s[vel_dim], v_minus[vel_dim], v_minus_cross[vel_dim], v_prime[vel_dim], v_prime_cross[vel_dim],
    v_plus[vel_dim];
    scalar q_div_m = q/m;

    t[0] = q_div_m*Bx*0.5*dt;
    t[1] = q_div_m*By*0.5*dt;
    t[2] = q_div_m*Bz*0.5*dt;
    scalar t_mag2 = t[0]*t[0] + t[1]*t[1] + t[2]*t[2];
    s[0] = 2*t[0]/(1+t_mag2);
    s[1] = 2*t[1]/(1+t_mag2);
    s[2] = 2*t[2]/(1+t_mag2);

    v_minus[0] = vel_x + q_div_m*Ex*0.5*dt;
    v_minus[1] = vel_y + q_div_m*Ey*0.5*dt;
    v_minus[2] = vel_z;

    CrossProduct(v_minus, t, v_minus_cross);

    v_prime[0] = v_minus[0] + v_minus_cross[0];
    v_prime[1] = v_minus[1] + v_minus_cross[1];
    v_prime[2] = v_minus[2] + v_minus_cross[2];

    CrossProduct(v_prime, s, v_prime_cross);

    v_plus[0] = v_minus[0] + v_prime_cross[0];
    v_plus[1] = v_minus[1] + v_prime_cross[1];
    v_plus[2] = v_minus[2] + v_prime_cross[2];

    vel_x = v_plus[0] + q_div_m*Ex*0.5*dt;
    vel_y = v_plus[1] + q_div_m*Ey*0.5*dt;
    vel_z = v_plus[2];
}

void UpdatePosition(scalar *pos_x, scalar *pos_y, const scalar *vel_x, const scalar *vel_y, const scalar dt,
                    const int Ntot) {
    //#pragma omp for
    for (int ip = 0; ip < Ntot; ip++) {
        pos_x[ip] += vel_x[ip]*dt;
        pos_y[ip] += vel_y[ip]*dt;
    }
}

void UpdateVelocity(scalar *vel_x, scalar *vel_y, scalar *vel_z, const scalar *Ex, const scalar *Ey,
                    const scalar *Bx, const scalar *By, const scalar *Bz, const scalar dt, const scalar q,
                    const scalar m, const int Ntot) {
    //#pragma omp for
    for (int ip = 0; ip < Ntot; ip++) {
        UpdateSingleVelocityBoris(vel_x[ip], vel_y[ip], vel_z[ip], Ex[ip], Ey[ip], Bx[ip], By[ip], Bz[ip], dt, q, m);
    }
}

void ParticlePush(scalar *pos_x, scalar *pos_y, scalar *vel_x, scalar *vel_y, scalar *vel_z,
                  const scalar *Ex, const scalar *Ey,
                  const scalar *Bx, const scalar *By, const scalar *Bz,
                  const scalar dt, const scalar q, const scalar m, const int Ntot) {
    //#pragma omp parallel num_threads(NUM_THREADS)
    //{
        UpdateVelocity(vel_x, vel_y, vel_z, Ex, Ey, Bx, By, Bz, dt, q, m, Ntot);
        UpdatePosition(pos_x, pos_y, vel_x, vel_y, dt, Ntot);
    //}
}


