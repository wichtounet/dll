#include "dbn/rbm.hpp"

template<typename RBM>
void test_rbm(){
    RBM rbm;

    std::vector<vector<double>> training;
    rbm.train(training, 10);
}

int main(){
    //Very basic RBM that must compile
    typedef dbn::rbm<dbn::layer<dbn::conf<true, 50, true, false, dbn::DecayType::L2, false, dbn::Type::SIGMOID, dbn::Type::SIGMOID>, 100, 100>> rbm_1;

    //Mix units
    typedef dbn::rbm<dbn::layer<dbn::conf<true, 50, true, false, dbn::DecayType::NONE, false, dbn::Type::GAUSSIAN, dbn::Type::NRLU>, 100, 100>> rbm_2;

    //Sparsity
    typedef dbn::rbm<dbn::layer<dbn::conf<true, 50, true, false, dbn::DecayType::L2, true, dbn::Type::SIGMOID, dbn::Type::SIGMOID>, 100, 100>> rbm_3;

    test_rbm<rbm_1>();
    test_rbm<rbm_2>();
    test_rbm<rbm_3>();

    return 0;
}