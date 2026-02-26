//
// Created by Vladimir Smirnov on 09.10.2021.
//

#include "NullCollisions.h"
#include <random>

void electron_null_collisions(ElectronNeutralElasticCollision &electron_elastic, Ionization &ionization) {
    int Ntot;
    scalar prob_1, prob_2, random_number;
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_real_distribution<double> distribution(0.0,1.0);
    Ntot = electron_elastic.particles->get_Ntot();
    for (int ptcl_idx = 0; ptcl_idx < Ntot; ptcl_idx++) {
        prob_1 = electron_elastic.probability(ptcl_idx);
        prob_2 = ionization.probability(ptcl_idx);
        random_number = distribution(generator);
        if (random_number < prob_1) {
            //cout << "elastic " << random_number << endl;
            electron_elastic.collision(ptcl_idx);
        }
        else if (random_number >= prob_1 and random_number < prob_1 + prob_2) {
            //cout << "ionization " << random_number << endl;
            ionization.collision(ptcl_idx);
        }

    }
}

void ion_null_collisions(IonNeutralElasticCollision &ion_elastic) {
    int Ntot;
    scalar prob_1, random_number;
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_real_distribution<double> distribution(0.0,1.0);

    Ntot = ion_elastic.particles->get_Ntot();
    for (int ptcl_idx = 0; ptcl_idx < Ntot; ptcl_idx++) {
        prob_1 = ion_elastic.probability(ptcl_idx);
        random_number = distribution(generator);
        if (random_number < prob_1) {
            //cout << "elastic " << random_number << endl;
            ion_elastic.collision(ptcl_idx);
        }
    }
}
