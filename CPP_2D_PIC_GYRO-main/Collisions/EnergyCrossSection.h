//
// Created by Vladimir Smirnov on 06.10.2021.
//

#ifndef CPP_2D_PIC_ENERGYCROSSSECTION_H
#define CPP_2D_PIC_ENERGYCROSSSECTION_H


#include <string>
#include <vector>
#include <array>
#include "../Tools/ProjectTypes.h"
using namespace std;


class EnergyCrossSection {
private:
    vector<array<scalar, 2>> energy_cross_section;
    int binary_search(scalar energy);
    scalar linear_interp(int ptr_1, int ptr_2, scalar energy);
public:
    EnergyCrossSection();
    explicit EnergyCrossSection(const string& filename);
    explicit EnergyCrossSection(const vector<array<scalar, 2>>& energy_cross_section);
    scalar get_cross_section(scalar energy);
    array<scalar, 2> get_energy_cross_section(int idx);
    bool empty();
};


#endif //CPP_2D_PIC_ENERGYCROSSSECTION_H
