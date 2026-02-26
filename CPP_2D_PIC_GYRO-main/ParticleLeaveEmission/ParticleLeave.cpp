//
// Created by Vladimir Smirnov on 06.10.2021.
//

#include "ParticleLeave.h"
#include <cmath>
#include <random>

void particle_leave(Particles &particles, const Matrix &leave_cells, const Grid &grid) {
    int cell_x, cell_y;
    int ptcl_idx = 0;
    while (ptcl_idx < particles.get_Ntot()) {
        cell_x = floor(particles.x[ptcl_idx] / grid.dx);
        cell_y = floor(particles.y[ptcl_idx] / grid.dy);
        if (leave_cells(cell_x, cell_y) == 1) {
            //cout << "leave on boundaries occured" << endl;
            particles.pop(ptcl_idx);
        }
        else {
            ptcl_idx++;
        }
    }
}

void some_particle_leave(Particles &particles, const Matrix &leave_cells, const Grid &grid, const int Ntot_leave) {
    //search for candidates to leave
    int cell_x, cell_y;
    vector<int> list_candidate_leave_idx;
    for(int ptcl_idx = 0; ptcl_idx < particles.get_Ntot(); ptcl_idx++) {
        cell_x = floor(particles.x[ptcl_idx] / grid.dx);
        cell_y = floor(particles.y[ptcl_idx] / grid.dy);
        if (leave_cells(cell_x, cell_y) == 1) {
            list_candidate_leave_idx.push_back(ptcl_idx);
        }
    }

    if (list_candidate_leave_idx.size() > Ntot_leave){
        std::random_device rd;
        std::default_random_engine generator(rd());
        vector<int> list_leave_idx;
        int leave_idx;
        for (int i = 0; i < Ntot_leave; i++) {
            std::uniform_int_distribution<int> distribution(0, list_candidate_leave_idx.size() - 1);
            leave_idx = distribution(generator);
            list_leave_idx.push_back(list_candidate_leave_idx[leave_idx]);
            swap(list_candidate_leave_idx[leave_idx], list_candidate_leave_idx[list_candidate_leave_idx.size() - 1]);
            list_candidate_leave_idx.pop_back();
        }
        particles.pop_list(list_leave_idx);
    } else {
        particles.pop_list(list_candidate_leave_idx);
    }
}

void some_particle_leave(Particles &particles, const scalar leave_radius, const array<scalar, 2> circle_center,
                         const int Ntot_leave) {
    //search for candidates to leave
    vector<int> list_candidate_leave_idx;
    for(int ptcl_idx = 0; ptcl_idx < particles.get_Ntot(); ptcl_idx++) {
        if (sqrt(pow(particles.x[ptcl_idx] - circle_center[0], 2) +
                 pow(particles.y[ptcl_idx] - circle_center[1], 2)) < leave_radius)
        {
            list_candidate_leave_idx.push_back(ptcl_idx);
        }
    }

    if (list_candidate_leave_idx.size() > Ntot_leave){
        std::random_device rd;
        std::default_random_engine generator(rd());
        vector<int> list_leave_idx;
        int leave_idx;
        for (int i = 0; i < Ntot_leave; i++) {
            std::uniform_int_distribution<int> distribution(0, list_candidate_leave_idx.size() - 1);
            leave_idx = distribution(generator);
            list_leave_idx.push_back(list_candidate_leave_idx[leave_idx]);
            swap(list_candidate_leave_idx[leave_idx], list_candidate_leave_idx[list_candidate_leave_idx.size() - 1]);
            list_candidate_leave_idx.pop_back();
        }
        particles.pop_list(list_leave_idx);
        //cout << "list_leave_idx.size(): " << list_leave_idx.size() << endl;
    } else {
        particles.pop_list(list_candidate_leave_idx);
        //cout << "list_candidate_leave_idx.size(): " << list_candidate_leave_idx.size() << endl;
    }
}

void particle_leave(Particles &particles, const Grid &grid, const scalar Radius, const array<scalar, 2> circle_center) {
    int ptcl_idx = 0;
    while (ptcl_idx < particles.get_Ntot()) {
        if (pow(particles.x[ptcl_idx] - circle_center[0], 2) +
            pow(particles.y[ptcl_idx] - circle_center[1], 2) >= pow(Radius, 2)) {
            particles.pop(ptcl_idx);
        }
        else {
            ptcl_idx++;
        }
    }
}

void some_particle_leave(Particles& particles, const scalar leave_radius_min, const scalar leave_radius_max,
                         const array<scalar, 2> circle_center, const int Ntot_leave) {
    //search for candidates to leave
    vector<int> list_candidate_leave_idx;
    for(int ptcl_idx = 0; ptcl_idx < particles.get_Ntot(); ptcl_idx++) {
        if (sqrt(pow(particles.x[ptcl_idx] - circle_center[0], 2) +
                 pow(particles.y[ptcl_idx] - circle_center[1], 2)) >= leave_radius_min and
            sqrt(pow(particles.x[ptcl_idx] - circle_center[0], 2) +
                 pow(particles.y[ptcl_idx] - circle_center[1], 2)) < leave_radius_max
            )
        {
            list_candidate_leave_idx.push_back(ptcl_idx);
        }
    }

    if (list_candidate_leave_idx.size() > Ntot_leave){
        std::random_device rd;
        std::default_random_engine generator(rd());
        vector<int> list_leave_idx;
        int leave_idx;
        for (int i = 0; i < Ntot_leave; i++) {
            std::uniform_int_distribution<int> distribution(0, list_candidate_leave_idx.size() - 1);
            leave_idx = distribution(generator);
            list_leave_idx.push_back(list_candidate_leave_idx[leave_idx]);
            swap(list_candidate_leave_idx[leave_idx], list_candidate_leave_idx[list_candidate_leave_idx.size() - 1]);
            list_candidate_leave_idx.pop_back();
        }
        particles.pop_list(list_leave_idx);
        //cout << "list_leave_idx.size(): " << list_leave_idx.size() << endl;
    } else {
        particles.pop_list(list_candidate_leave_idx);
        //cout << "list_candidate_leave_idx.size(): " << list_candidate_leave_idx.size() << endl;
    }
}
