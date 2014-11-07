//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "dll/conv_rbm.hpp"

template<typename RBM>
void test_rbm(){
    RBM rbm;

    etl::dyn_vector<double> test(28 * 28, 1.0);
    rbm.reconstruct(test);

    std::vector<etl::dyn_vector<double>> test_full;
    rbm.train(test_full, 40);

    std::vector<double> v(28 * 28 * 1);
    std::vector<double> h(12 * 12 * 40);
    rbm.energy(v, h);
}

int main(){
    //Very basic RBM that must compile
    //typedef dll::conv_rbm<dll::conv_rbm_desc<dll::conv_conf<true, 50, dll::unit_type::BINARY, dll::unit_type::SIGMOID>, 32, 12, 40>> crbm_1;
    typedef dll::conv_rbm_desc<28, 1, 12, 40>::rbm_t crbm_1;

    std::cout << "NV*NV=" << std::remove_reference<decltype(crbm_1::v1)>::type::etl_size << std::endl;
    std::cout << "NH*NH=" << std::remove_reference<decltype(crbm_1::h1_a)>::type::etl_size << std::endl;
    std::cout << "NW*NW=" << std::remove_reference<decltype(crbm_1::w)>::type::etl_size << std::endl;

    typedef dll::conv_rbm_desc<28, 1, 12, 40, dll::momentum, dll::batch_size<50>>::rbm_t crbm_2;

    //Test them all

    test_rbm<crbm_1>();
    test_rbm<crbm_2>();

    return 0;
}
