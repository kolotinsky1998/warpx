//
// Created by Vladimir Smirnov on 02.10.2021.
//

#ifndef CPP_2D_PIC_PUSHER_H
#define CPP_2D_PIC_PUSHER_H

#include "../Tools/ProjectTypes.h"


void CrossProduct(const scalar v1[], const scalar v2[], scalar result[]);

void UpdateSingleVelocityBoris(scalar& vel_x, scalar& vel_y, scalar& vel_z, scalar Ex, scalar Ey,
                               scalar Bx, scalar By, scalar Bz, scalar dt, scalar q, scalar m);

void UpdateVelocity(scalar vel_x[], scalar vel_y[], scalar vel_z[], const scalar Ex[],
                    const scalar Ey[], const scalar Bx[], const scalar By[], const scalar Bz[],
                    const scalar dt, const scalar q, const scalar m, const int Ntot);

void UpdatePosition(scalar pos_x[], scalar pos_y[], const scalar vel_x[], const scalar vel_y[], const scalar dt,
                    const int Ntot);

void ParticlePush(scalar pos_x[], scalar pos_y[], scalar vel_x[], scalar vel_y[],
                  scalar vel_z[], const scalar Ex[], const scalar Ey[],
                  const scalar Bx[], const scalar By[], const scalar Bz[], const scalar dt, const scalar q,
                  const scalar m, const int Ntot);


#endif //CPP_2D_PIC_PUSHER_H
