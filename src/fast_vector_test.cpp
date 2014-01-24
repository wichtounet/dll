//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "fast_vector.hpp"

#include <iostream>

int main(){
    auto e = fast_vector<double, 3>(1) + fast_vector<double, 3>(2) + fast_vector<double, 3>(4) + fast_vector<double, 3>(3);

    std::cout << e[2] << std::endl;

    return 0;
}
