//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "dll/dyn_dbn.hpp"
#include "dll/ocv_visualizer.hpp"

template<typename DBN>
void test_dbn(DBN& dbn){
    dbn->display();

    std::vector<etl::dyn_vector<double>> images;

    dbn->pretrain(images, 10);
}

int main(){
    using dbn_t =
        dll::dyn_dbn_desc<
            dll::dbn_layers<
                dll::dyn_rbm_desc<dll::momentum, dll::init_weights>::rbm_t,
                dll::dyn_rbm_desc<dll::momentum>::rbm_t,
                dll::dyn_rbm_desc<dll::momentum, dll::hidden<dll::unit_type::SOFTMAX>>::rbm_t
        >, dll::watcher<dll::opencv_dbn_visualizer>
        >::dbn_t;

    auto dbn = std::make_unique<dbn_t>(
        std::make_tuple(28*28,100),
        std::make_tuple(100,200),
        std::make_tuple(200,10));

    test_dbn(dbn);

    return 0;
}
