//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "dll/rbm.hpp"

template<typename RBM>
void test_rbm(){
    RBM rbm;

    std::vector<std::vector<double>> training;
    rbm.train(training, 10);
}

template <typename RBM>
using pcd2_trainer_t = dll::persistent_cd_trainer<2, RBM>;

int main(){
    //Very basic RBM that must compile
    typedef dll::rbm_desc<100, 100, dll::weight_decay<dll::decay_type::L2>>::rbm_t rbm_1;

    //Mix units
    typedef dll::rbm_desc<100, 100, dll::momentum, dll::batch_size<50>, dll::visible<dll::unit_type::GAUSSIAN>, dll::hidden<dll::unit_type::RELU>>::rbm_t rbm_2;

    //Sparsity
    typedef dll::rbm_desc<100, 100, dll::momentum, dll::sparsity>::rbm_t rbm_3;

    //PCD-2

    typedef dll::rbm_desc<100, 100, dll::trainer<pcd2_trainer_t>>::rbm_t rbm_4;

    //PCD-2 and sparsity

    typedef dll::rbm_desc<100, 100, dll::trainer<pcd2_trainer_t>, dll::sparsity>::rbm_t rbm_5;

    //Test them all

    test_rbm<rbm_1>();
    test_rbm<rbm_2>();
    test_rbm<rbm_3>();
    test_rbm<rbm_4>();
    test_rbm<rbm_5>();

    return 0;
}