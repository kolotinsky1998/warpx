//
// Created by Vladimir Smirnov on 16.11.2021.
//

#ifndef CPP_2D_PIC_GYROKINETICPUSHER_H
#define CPP_2D_PIC_GYROKINETICPUSHER_H


#include "Pusher.h"


scalar DotProduct(const scalar v1[], const scalar v2[]);

void GyroUpdateSingleVelocityBoris(scalar &vel_x_c, scalar &vel_y_c, scalar &vel_z_c, scalar &vel_x, scalar &vel_y, scalar &vel_z,
                                   const scalar Ex, const scalar Ey, const scalar Bx, const scalar By, const scalar Bz,
                                   const scalar dt, const scalar q, const scalar m);

void GyroUpdateVelocity(scalar vel_x_c[], scalar vel_y_c[], scalar vel_z_c[], scalar vel_x[], scalar vel_y[], scalar vel_z[],
                        const scalar Ex[], const scalar Ey[], const scalar Bx[], const scalar By[], const scalar Bz[],
                        const scalar dt, const scalar q, const scalar m, const int Ntot);

void GyroParticlePush(scalar pos_x[], scalar pos_y[], scalar vel_x_c[], scalar vel_y_c[], scalar vel_z_c[],
                      scalar vel_x[], scalar vel_y[], scalar vel_z[],
                      const scalar Ex[], const scalar Ey[], const scalar Bx[], const scalar By[], const scalar Bz[],
                      const scalar dt, const scalar q, const scalar m, const int Ntot);


#endif //CPP_2D_PIC_GYROKINETICPUSHER_H
