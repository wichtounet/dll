//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "dll/dbn.hpp"

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

    typedef dll::dbn<
        dll::layer<28 * 28, 100, dll::in_dbn, dll::momentum, dll::batch_size<50>, dll::init_weights, dll::weight_decay<dll::decay_type::L2>, dll::sparsity>,
        dll::layer<100, 100, dll::in_dbn, dll::momentum, dll::batch_size<50>>,
        dll::layer<110, 200, dll::in_dbn, dll::batch_size<50>, dll::momentum, dll::weight_decay<dll::decay_type::L2_FULL>>
    > dbn_1;

    //Test them all

    test_dbn<dbn_1>();

    return 0;
}