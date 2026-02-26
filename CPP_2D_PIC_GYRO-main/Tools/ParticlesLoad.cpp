#include "ParticlesLoad.h"
#include <fstream>


ParticlesLoad::ParticlesLoad(Particles &particles, const string& pos_path, const string& vel_path, int it_start, int it_end) : particles(&particles), pos_path(pos_path), vel_path(vel_path) {
    position_velocity_load(it_start, it_end);
}

void ParticlesLoad::position_velocity_load(int it_start, int it_end) {
    ifstream input_vel(vel_path);
    ifstream input_pos(pos_path);
    array<scalar, 3> vel_tmp{};
    array<scalar, 2> pos_tmp{};
    vector<array<scalar, 3>> vel{};
    vector<array<scalar, 2>> pos{};
    string element;

    if (input_pos) {
        do {
            input_pos >> element;
        }
        while (element != to_string(it_start));
        while (input_pos) {
            input_pos >> element;
            if (element == to_string(it_end)) {
                break;
            }
            pos_tmp[0] = stod(element);
            input_pos >> element;
            pos_tmp[1] = stod(element);
            pos.push_back(pos_tmp);
        }
    } else {
        cout << "can't load positions from file";
        throw;
    }

    if (input_vel) {
        do {
            input_vel >> element;
        }
        while (element != to_string(it_start));
        while (input_vel) {
            input_vel >> element;
            if (element == to_string(it_end)) {
                break;
            }
            vel_tmp[0] = stod(element);
            input_vel >> element;
            vel_tmp[1] = stod(element);
            input_vel >> element;
            vel_tmp[2] = stod(element);
            vel.push_back(vel_tmp);
        }
    } else {
        cout << "can't load velocities from file";
        throw;
    }

    assert(vel.size() == pos.size() and "velocities and positions not the same size");

    for(int ptcl_idx = 0; ptcl_idx < vel.size(); ptcl_idx++) {
        particles->append(pos[ptcl_idx], vel[ptcl_idx]);
    }
}