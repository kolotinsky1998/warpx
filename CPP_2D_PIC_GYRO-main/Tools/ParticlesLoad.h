#ifndef CPP_2D_PIC_PARTICLESLOAD_H
#define CPP_2D_PIC_PARTICLESLOAD_H


#include "../Particles/Particles.h"
#include <string>


class ParticlesLoad {
private:
    Particles* particles;
    string vel_path, pos_path;
    void position_velocity_load(int it_start, int it_end);
public:
    ParticlesLoad(Particles& particles, const string& pos_path, const string& vel_path, int it_start, int it_end);
};


#endif //CPP_2D_PIC_PARTICLESLOAD_H
