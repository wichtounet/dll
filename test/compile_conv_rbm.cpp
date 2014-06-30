#include "dll/conv_rbm.hpp"
#include "dll/conv_layer.hpp"

template<typename RBM>
void test_rbm(){
    RBM rbm;

    vector<double> test(32 * 32, 1.0);
    rbm.reconstruct(test);

    std::vector<vector<double>> test_full;
    rbm.train(test_full, 40);
}

int main(){
    //Very basic RBM that must compile
    //typedef dll::conv_rbm<dll::conv_layer<dll::conv_conf<true, 50, dll::Type::SIGMOID, dll::Type::SIGMOID>, 32, 12, 40>> crbm_1;
    typedef dll::conv_rbm<dll::conv_layer<32, 12, 40>> crbm_1;

    std::cout << crbm_1::Momentum << std::endl;
    std::cout << crbm_1::BatchSize << std::endl;

    typedef dll::conv_rbm<dll::conv_layer<32, 12, 40, dll::momentum, dll::batch_size<50>>> crbm_2;

    std::cout << crbm_2::Momentum << std::endl;
    std::cout << crbm_2::BatchSize << std::endl;

    //Test them all

    test_rbm<crbm_1>();
    test_rbm<crbm_2>();

    return 0;
}