/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_PUSHER_UPDATEMOMENTUM_GYROKINETIC_H_
#define WARPX_PARTICLES_PUSHER_UPDATEMOMENTUM_GYROKINETIC_H_

#include "Utils/WarpXConst.H"

#include <AMReX_GpuAtomic.H>
#include <AMReX_REAL.H>

#include <cmath>

/** \brief Update particle momentum with a gyrokinetic approximation.
 *
 *  This implementation follows the reference algorithm in CPP_2D_PIC_GYRO-main
 *  and is valid for 2D XZ with magnetic field perpendicular to the dynamics
 *  plane (i.e. B = [0, By, 0] in Cartesian coordinates).
 */
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void UpdateMomentumGyroKinetic(
    amrex::ParticleReal& ux,
    amrex::ParticleReal& uy,
    amrex::ParticleReal& uz,
    amrex::ParticleReal& ux_gc,
    amrex::ParticleReal& uy_gc,
    amrex::ParticleReal& uz_gc,
    const amrex::ParticleReal Ex,
    const amrex::ParticleReal Ey,
    const amrex::ParticleReal Ez,
    const amrex::ParticleReal Bx,
    const amrex::ParticleReal By,
    const amrex::ParticleReal Bz,
    const amrex::ParticleReal q,
    const amrex::ParticleReal m,
    const amrex::Real dt,
    int* const invalid_geometry_flag = nullptr)
{
    using namespace amrex::literals;

    constexpr amrex::ParticleReal tiny = 1.0e-30_prt;
    constexpr amrex::ParticleReal rel_tol = 1.0e-12_prt;
    constexpr amrex::ParticleReal abs_tol = 1.0e-40_prt;

    const amrex::ParticleReal b2 = Bx*Bx + By*By + Bz*Bz;
    const amrex::ParticleReal by2 = By*By;
    const amrex::ParticleReal perp2 = Bx*Bx + Bz*Bz;
    const amrex::ParticleReal perp_tol = abs_tol + rel_tol*(b2 > by2 ? b2 : by2);

    if (invalid_geometry_flag != nullptr) {
        if (b2 <= tiny || std::abs(By) <= tiny || perp2 > perp_tol) {
            amrex::Gpu::Atomic::Exch(invalid_geometry_flag, 1);
        }
    }

    if (b2 <= tiny) {
        ux_gc = ux;
        uy_gc = uy;
        uz_gc = uz;
        return;
    }

    // E x B drift velocity
    const amrex::ParticleReal vex = (Ey*Bz - Ez*By) / b2;
    const amrex::ParticleReal vey = (Ez*Bx - Ex*Bz) / b2;
    const amrex::ParticleReal vez = (Ex*By - Ey*Bx) / b2;

    amrex::ParticleReal vx = ux - vex;
    amrex::ParticleReal vy = uy - vey;
    amrex::ParticleReal vz = uz - vez;

    // Decompose into parallel and perpendicular components
    const amrex::ParticleReal v_dot_b = vx*Bx + vy*By + vz*Bz;
    amrex::ParticleReal vparx = Bx * v_dot_b / b2;
    amrex::ParticleReal vpary = By * v_dot_b / b2;
    amrex::ParticleReal vparz = Bz * v_dot_b / b2;

    amrex::ParticleReal vperpx = vx - vparx;
    amrex::ParticleReal vperpy = vy - vpary;
    amrex::ParticleReal vperpz = vz - vparz;

    // Parallel electric field acceleration
    const amrex::ParticleReal e_dot_b = Ex*Bx + Ey*By + Ez*Bz;
    const amrex::ParticleReal eparx = Bx * e_dot_b / b2;
    const amrex::ParticleReal epary = By * e_dot_b / b2;
    const amrex::ParticleReal eparz = Bz * e_dot_b / b2;

    const amrex::ParticleReal qmdt = (m > 0._prt) ? (q/m)*dt : 0._prt;
    vparx += qmdt * eparx;
    vpary += qmdt * epary;
    vparz += qmdt * eparz;

    // Rotate perpendicular velocity
    const amrex::ParticleReal bnorm = std::sqrt(b2);
    const amrex::ParticleReal phi = std::abs(qmdt) * bnorm;
    const amrex::ParticleReal cphi = std::cos(phi);
    const amrex::ParticleReal sphi = std::sin(phi);

    const amrex::ParticleReal bunitx = Bx / bnorm;
    const amrex::ParticleReal bunity = By / bnorm;
    const amrex::ParticleReal bunitz = Bz / bnorm;

    const amrex::ParticleReal bdotvperp = bunitx*vperpx + bunity*vperpy + bunitz*vperpz;
    const amrex::ParticleReal cx = bunity*vperpz - bunitz*vperpy;
    const amrex::ParticleReal cy = bunitz*vperpx - bunitx*vperpz;
    const amrex::ParticleReal cz = bunitx*vperpy - bunity*vperpx;

    const amrex::ParticleReal vperpx_rot = vperpx*cphi + cx*sphi + bunitx*bdotvperp*(1._prt-cphi);
    const amrex::ParticleReal vperpy_rot = vperpy*cphi + cy*sphi + bunity*bdotvperp*(1._prt-cphi);
    const amrex::ParticleReal vperpz_rot = vperpz*cphi + cz*sphi + bunitz*bdotvperp*(1._prt-cphi);

    // Gyrocenter velocity
    ux_gc = vparx + vex;
    uy_gc = vpary + vey;
    uz_gc = vparz + vez;

    // Total velocity
    ux = ux_gc + vperpx_rot;
    uy = uy_gc + vperpy_rot;
    uz = uz_gc + vperpz_rot;
}

#endif // WARPX_PARTICLES_PUSHER_UPDATEMOMENTUM_GYROKINETIC_H_
