//
// Created by Vladimir Smirnov on 11.09.2021.
//

#ifndef CPP_2D_PIC_PARTICLES_H
#define CPP_2D_PIC_PARTICLES_H


#include "../Tools/ProjectTypes.h"
#include "../Tools/Matrix.h"
#include "../Tools/Grid.h"
#include <vector>
#include <array>
using namespace std;

class Particles {
protected:
    int Ntot;
    scalar ptcls_per_macro, mass, charge, Bx_const, By_const, Bz_const;
public:
    vector<scalar> vx;
    vector<scalar> vy;
    vector<scalar> vz;
    vector<scalar> x;
    vector<scalar> y;
    vector<scalar> Bx;
    vector<scalar> By;
    vector<scalar> Bz;
    vector<scalar> Ex;
    vector<scalar> Ey;
    Particles(scalar m, scalar q, int N, scalar N_per_macro = 1);

    virtual void append(const array<scalar, 2>& position, const array<scalar, 3>& velocity);

    virtual void pop(int ptcl_idx);

    virtual void pop_list(const vector<int>& ptcl_idx_list);

    virtual void vel_pusher(scalar  dt);
    virtual void pusher(scalar  dt);

    //setters & getters
    void set_const_magnetic_field(const array<scalar, 3>& B);
    void set_position(const int ptcl_idx, const array<scalar, 2>& position);
    virtual void set_velocity(const int ptcl_idx, const array<scalar, 3>& velocity);
    array<scalar, 2> get_position(const int ptcl_idx) const;

    virtual array<scalar, 3> get_velocity(const int ptcl_idx) const;

    int get_Ntot() const;
    scalar get_mass() const;
    scalar get_charge() const;
    scalar get_ptcls_per_macro() const;
};


#endif //CPP_2D_PIC_PARTICLES_H
