//
// Created by Vladimir Smirnov on 11.09.2021.
//

#ifndef CPP_2D_PIC_GRID_H
#define CPP_2D_PIC_GRID_H


#include "ProjectTypes.h"

class Grid {
public:
    int Nx, Ny;
    scalar dx, dy;
    Grid(int Nx, int Ny, scalar dx, scalar dy);
};


#endif //CPP_2D_PIC_GRID_H
