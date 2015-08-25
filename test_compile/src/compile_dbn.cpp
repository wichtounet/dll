//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "dll/rbm.hpp"
#include "dll/dbn.hpp"

template<typename DBN>
void test_dbn(){
    DBN dbn;

    dbn.display();

    std::vector<etl::dyn_vector<float>> images;
    std::vector<uint8_t> labels;

    dbn.pretrain(images, 10);
    dbn.fine_tune(images, labels, 10);
}

int main(){
    //Basic example

    typedef dll::dbn_desc<
        dll::dbn_layers<
        dll::rbm_desc<28 * 28, 100, dll::momentum, dll::batch_size<50>, dll::init_weights, dll::weight_decay<dll::decay_type::L2>, dll::sparsity<>>::rbm_t,
        dll::rbm_desc<100, 100, dll::momentum, dll::batch_size<50>>::rbm_t,
        dll::rbm_desc<100, 200, dll::batch_size<50>, dll::momentum, dll::weight_decay<dll::decay_type::L2_FULL>>::rbm_t
    >, dll::watcher<dll::silent_dbn_watcher>>::dbn_t dbn_1;

    //With labels

    typedef dll::dbn_desc<
        dll::dbn_label_layers<
        dll::rbm_desc<28 * 28, 100, dll::momentum, dll::batch_size<50>, dll::init_weights, dll::weight_decay<dll::decay_type::L2>, dll::sparsity<>>::rbm_t,
        dll::rbm_desc<100, 100, dll::momentum, dll::batch_size<50>>::rbm_t,
        dll::rbm_desc<110, 200, dll::batch_size<50>, dll::momentum, dll::weight_decay<dll::decay_type::L2_FULL>>::rbm_t
    >, dll::watcher<dll::silent_dbn_watcher>>::dbn_t dbn_2;

    //Test them all

    test_dbn<dbn_1>();
    test_dbn<dbn_2>();

    return 0;
}
