//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#define DLL_SVM_SUPPORT

#include "dll/conv_rbm.hpp"
#include "dll/dbn.hpp"

template<typename DBN>
void test_dbn(){
    DBN dbn;

    dbn.display();

    std::vector<etl::dyn_matrix<double, 3>> images;
    std::vector<uint8_t> labels;

    dbn.pretrain(images, 10);
    dbn.svm_train(images, labels);
    dbn.svm_train(images.begin(), images.end(), labels.begin(), labels.end());
    dbn.svm_grid_search(images, labels);
    dbn.svm_grid_search(images.begin(), images.end(), labels.begin(), labels.end());
    dbn.svm_predict(images[1]);
}

int main(){
    //Basic example

    typedef dll::dbn_desc<
        dll::dbn_layers<
        dll::conv_rbm_desc<28, 1, 12, 40, dll::momentum, dll::batch_size<50>>::rbm_t,
        dll::conv_rbm_desc<12, 40, 6, 40, dll::momentum, dll::batch_size<50>>::rbm_t>>::dbn_t dbn_1;

    //Test them all

    test_dbn<dbn_1>();

    return 0;
}
