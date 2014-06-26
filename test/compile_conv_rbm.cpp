#include "dbn/conv_rbm.hpp"
#include "dbn/conv_layer.hpp"
#include "dbn/conv_conf.hpp"

template<typename RBM>
void test_rbm(){
    RBM rbm;
}

int main(){
    //Very basic RBM that must compile
    typedef dbn::conv_rbm<dbn::conv_layer<dbn::conv_conf<true, 50, dbn::Type::SIGMOID, dbn::Type::SIGMOID>, 32, 12, 40>> crbm_1;

    //Test them all

    test_rbm<crbm_1>();

    return 0;
}