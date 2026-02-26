//
// Created by Vladimir Smirnov on 27.10.2021.
//

#include "GyroKineticParticles.h"
#include "GyroKineticPusher.h"

void GyroKineticParticles::vel_pusher(scalar dt) {
    GyroUpdateVelocity(vx_c.data(), vy_c.data(), vz_c.data(),
            vx.data(), vy.data(), vz.data(),Ex.data(), Ey.data(),
            Bx.data(), By.data(), Bz.data(), dt, charge, mass, Ntot);
}

void GyroKineticParticles::pusher(scalar dt) {
    GyroParticlePush(x.data(), y.data(), vx_c.data(), vy_c.data(), vz_c.data(),
            vx.data(), vy.data(), vz.data(), Ex.data(), Ey.data(),
            Bx.data(), By.data(), Bz.data(), dt, charge, mass, Ntot);
}

void GyroKineticParticles::set_velocity(const int ptcl_idx, const array<scalar, 3> &velocity) {
    Particles::set_velocity(ptcl_idx, velocity);
    vx_c[ptcl_idx] = velocity[0];
    vy_c[ptcl_idx] = velocity[1];
    vz_c[ptcl_idx] = velocity[2];
}

array<scalar, 3> GyroKineticParticles::get_velocity(const int ptcl_idx) const {
    return Particles::get_velocity(ptcl_idx);
}

GyroKineticParticles::GyroKineticParticles(scalar m, scalar q, int N, scalar N_per_macro) : Particles(m, q, N,
                                                                                                      N_per_macro) {
    vx_c.resize(Ntot);
    vy_c.resize(Ntot);
    vz_c.resize(Ntot);
}

void GyroKineticParticles::append(const array<scalar, 2> &position, const array<scalar, 3> &velocity) {
    Particles::append(position, velocity);
    vx_c.push_back(velocity[0]);
    vy_c.push_back(velocity[1]);
    vz_c.push_back(velocity[2]);
}

void GyroKineticParticles::pop(int ptcl_idx) {
    Particles::pop(ptcl_idx);
    vx_c.pop_back();
    vy_c.pop_back();
    vz_c.pop_back();
}

void GyroKineticParticles::pop_list(const vector<int> &ptcl_idx_list) {
    //Particles::pop_list(ptcl_idx_list);
    assert(x.size() == Ntot);
    assert(ptcl_idx_list.size() <= Ntot);

    int leave_ptcl_idx;
    int main_ptcl_idx = Ntot - 1;
    for(int i = 0; i < ptcl_idx_list.size(); i++) {
        leave_ptcl_idx = ptcl_idx_list[i];
        swap(x[leave_ptcl_idx], x[main_ptcl_idx]);
        swap(y[leave_ptcl_idx], y[main_ptcl_idx]);
        swap(vx_c[leave_ptcl_idx], vx_c[main_ptcl_idx]);
        swap(vy_c[leave_ptcl_idx], vy_c[main_ptcl_idx]);
        swap(vz_c[leave_ptcl_idx], vz_c[main_ptcl_idx]);
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
        vx_c.pop_back();
        vy_c.pop_back();
        vz_c.pop_back();
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
