//=======================================================================
// Copyright Baptiste Wicht 2014.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//=======================================================================

#include <iostream>

#include "rbm.hpp"

int main(){
    dbn::rbm<char,char> rbm(6, 2);

    std::vector<std::vector<char>> training = {{1,1,1,0,0,0},{1,1,1,0,0,0},{0,0,1,1,0,0},{0,0,1,1,0,0},{0,0,1,1,1,0}};
    rbm.train(training, 5000);
    rbm.run_visible(std::vector<char>({0,0,0,0,0,0}));
    rbm.display();

    //TODO
    //

    return 0;
}
