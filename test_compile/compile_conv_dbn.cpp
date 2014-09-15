//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "dll/dbn.hpp"
#include "dll/conv_dbn_desc.hpp"
#include "dll/dbn_layers.hpp"

template<typename DBN>
void test_dbn(){
    DBN dbn;

    dbn.display();

    std::vector<etl::dyn_vector<double>> images;
    std::vector<uint8_t> labels;

    dbn.pretrain(images, 10);
    dbn.fine_tune(images, labels, 10, 10);
}

template <typename RBM>
using pcd2_trainer_t = dll::persistent_cd_trainer<2, RBM>;

int main(){
    //Basic example

    typedef dll::conv_dbn_desc<
        dll::dbn_layers<
        dll::conv_rbm_desc<28, 12, 40, dll::momentum, dll::batch_size<50>, dll::init_weights>::rbm_t,
        dll::conv_rbm_desc<12*12*40, 6, 40, dll::momentum, dll::batch_size<50>>::rbm_t>>::dbn_t dbn_1;

    //Test them all

    test_dbn<dbn_1>();

    return 0;
}