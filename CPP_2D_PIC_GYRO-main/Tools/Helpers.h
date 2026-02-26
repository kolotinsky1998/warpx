//
// Created by Vladimir Smirnov on 06.10.2021.
//

#ifndef CPP_2D_PIC_HELPERS_H
#define CPP_2D_PIC_HELPERS_H


#include "ProjectTypes.h"
#include <vector>
#include <string>
#include <fstream>
using  namespace std;

vector<scalar> string_to_numeric_vector(const string& s);

int rows_count(const string& filename);


#endif //CPP_2D_PIC_HELPERS_H
