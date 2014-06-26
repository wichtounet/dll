#include "dbn/rbm.hpp"

template<typename RBM>
void test_rbm(){
    RBM rbm;
}

int main(){
    //Very basic RBM that must compile
    typedef dbn::rbm<dbn::layer<dbn::conf<true, 50, true, false, dbn::DecayType::L2, dbn::Type::SIGMOID, dbn::Type::SIGMOID>, 100, 100>> rbm_1;

    //Mix units
    typedef dbn::rbm<dbn::layer<dbn::conf<true, 50, true, false, dbn::DecayType::NONE, dbn::Type::GAUSSIAN, dbn::Type::NRLU>, 100, 100>> rbm_2;

    test_rbm<rbm_1>();
    test_rbm<rbm_2>();

    return 0;
}