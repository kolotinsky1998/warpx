//
// Created by Vladimir Smirnov on 06.10.2021.
//

#ifndef CPP_2D_PIC_NEUTRALGAS_H
#define CPP_2D_PIC_NEUTRALGAS_H


#include <array>
#include "../Tools/ProjectTypes.h"


class NeutralGas {
public:
    const scalar n;
    const scalar mass;
    const scalar T;
    NeutralGas(scalar n, scalar mass, scalar T);
    std::array<scalar, 3> generate_velocity() const;
};


#endif //CPP_2D_PIC_NEUTRALGAS_H
