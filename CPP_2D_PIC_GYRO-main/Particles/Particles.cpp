//
// Created by Vladimir Smirnov on 11.09.2021.
//

#include "Particles.h"
#include "Pusher.h"
#include <algorithm>
#include <cassert>


void Particles::append(const array<scalar, 2> &position, const array<scalar, 3> &velocity) {
    x.push_back(position[0]);
    y.push_back(position[1]);
    vx.push_back(velocity[0]);
    vy.push_back(velocity[1]);
    vz.push_back(velocity[2]);
    Bx.push_back(Bx_const);
    By.push_back(By_const);
    Bz.push_back(Bz_const);
    Ex.push_back(0);
    Ey.push_back(0);
    Ntot++;
}

void Particles::pop(const int ptcl_idx) {
    swap(x[ptcl_idx], x[Ntot-1]);
    swap(y[ptcl_idx], y[Ntot-1]);
    swap(vx[ptcl_idx], vx[Ntot-1]);
    swap(vy[ptcl_idx], vy[Ntot-1]);
    swap(vz[ptcl_idx], vz[Ntot-1]);
    swap(Bx[ptcl_idx], Bx[Ntot-1]);
    swap(By[ptcl_idx], By[Ntot-1]);
    swap(Bz[ptcl_idx], Bz[Ntot-1]);
    swap(Ex[ptcl_idx], Ex[Ntot-1]);
    swap(Ey[ptcl_idx], Ey[Ntot-1]);
    x.pop_back();
    y.pop_back();
    vx.pop_back();
    vy.pop_back();
    vz.pop_back();
    Bx.pop_back();
    By.pop_back();
    Bz.pop_back();
    Ex.pop_back();
    Ey.pop_back();
    Ntot--;
}

void Particles::vel_pusher(const scalar dt) {
    UpdateVelocity(vx.data(), vy.data(), vz.data(), Ex.data(), Ey.data(), Bx.data(), By.data(), Bz.data(), dt, charge, mass, Ntot);
}

void Particles::pusher(const scalar dt) {
    ParticlePush(x.data(), y.data(), vx.data(), vy.data(), vz.data(), Ex.data(), Ey.data(), Bx.data(), By.data(), Bz.data(), dt, charge, mass, Ntot);
}

Particles::Particles(scalar m, scalar q, int N, scalar N_per_macro) {
    Ntot = N;
    if (N_per_macro > 1) {
        ptcls_per_macro = N_per_macro;
    } else {
        ptcls_per_macro = 1;
    }
    mass = m * ptcls_per_macro;
    charge = q * ptcls_per_macro;
    Bx_const = 0;
    By_const = 0;
    Bz_const = 0;
    x.resize(Ntot, 0);
    y.resize(Ntot, 0);
    vx.resize(Ntot, 0);
    vy.resize(Ntot, 0);
    vz.resize(Ntot, 0);
    Ex.resize(Ntot, 0);
    Ey.resize(Ntot, 0);
    Bx.resize(Ntot, 0);
    By.resize(Ntot, 0);
    Bz.resize(Ntot, 0);
}

scalar Particles::get_ptcls_per_macro() const {
    return ptcls_per_macro;
}

scalar Particles::get_charge() const {
    return charge/ptcls_per_macro;;
}

scalar Particles::get_mass() const {
    return mass/ptcls_per_macro;
}

int Particles::get_Ntot() const {
    return Ntot;
}

array<scalar, 3> Particles::get_velocity(const int ptcl_idx) const {
    array<scalar, 3> vel = {vx[ptcl_idx], vy[ptcl_idx], vz[ptcl_idx]};
    return vel;
}

array<scalar, 2> Particles::get_position(const int ptcl_idx) const {
    array<scalar, 2> pos = {x[ptcl_idx], y[ptcl_idx]};
    return pos;
}

void Particles::set_velocity(const int ptcl_idx, const array<scalar, 3> &velocity) {
    vx[ptcl_idx] = velocity[0];
    vy[ptcl_idx] = velocity[1];
    vz[ptcl_idx] = velocity[2];
}

void Particles::set_position(const int ptcl_idx, const array<scalar, 2> &position) {
    x[ptcl_idx] = position[0];
    y[ptcl_idx] = position[1];
}

void Particles::set_const_magnetic_field(const array<scalar, 3> &B) {
    Bx.assign(Ntot, B[0]);
    By.assign(Ntot, B[1]);
    Bz.assign(Ntot, B[2]);
    Bx_const = B[0];
    By_const = B[1];
    Bz_const = B[2];
}

void Particles::pop_list(const vector<int> &ptcl_idx_list) {
    assert(x.size() == Ntot);
    assert(ptcl_idx_list.size() <= Ntot);

    int leave_ptcl_idx;
    int main_ptcl_idx = Ntot - 1;
    for(int i = 0; i < ptcl_idx_list.size(); i++) {
        leave_ptcl_idx = ptcl_idx_list[i];
        swap(x[leave_ptcl_idx], x[main_ptcl_idx]);
        swap(y[leave_ptcl_idx], y[main_ptcl_idx]);
        swap(vx[leave_ptcl_idx], vx[main_ptcl_idx]);
        swap(vy[leave_ptcl_idx], vy[main_ptcl_idx]);
        swap(vz[leave_ptcl_idx], vz[main_ptcl_idx]);
        swap(Bx[leave_ptcl_idx], Bx[main_ptcl_idx]);
        swap(By[leave_ptcl_idx], By[main_ptcl_idx]);
        swap(Bz[leave_ptcl_idx], Bz[main_ptcl_idx]);
        swap(Ex[leave_ptcl_idx], Ex[main_ptcl_idx]);
        swap(Ey[leave_ptcl_idx], Ey[main_ptcl_idx]);
        main_ptcl_idx--;
    }

    for(int i = 0; i < ptcl_idx_list.size(); i++) {
        x.pop_back();
        y.pop_back();
        vx.pop_back();
        vy.pop_back();
        vz.pop_back();
        Bx.pop_back();
        By.pop_back();
        Bz.pop_back();
        Ex.pop_back();
        Ey.pop_back();
    }
    Ntot = x.size();
}





