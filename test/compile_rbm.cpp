#include "dbn/rbm.hpp"

template<typename RBM>
void test_rbm(){
    RBM rbm;

    std::vector<vector<double>> training;
    rbm.train(training, 10);
}

template <typename RBM>
using pcd2_trainer_t = dbn::persistent_cd_trainer<2, RBM>;

int main(){
    //Very basic RBM that must compile
    typedef dbn::rbm<dbn::layer<dbn::conf<true, 50, true, false, dbn::DecayType::L2, false, dbn::Type::SIGMOID, dbn::Type::SIGMOID>, 100, 100>> rbm_1;

    //Mix units
    typedef dbn::rbm<dbn::layer<dbn::conf<true, 50, true, false, dbn::DecayType::NONE, false, dbn::Type::GAUSSIAN, dbn::Type::NRLU>, 100, 100>> rbm_2;

    //Sparsity
    typedef dbn::rbm<dbn::layer<dbn::conf<true, 50, true, false, dbn::DecayType::L2, true, dbn::Type::SIGMOID, dbn::Type::SIGMOID>, 100, 100>> rbm_3;

    //PCD-2

    typedef dbn::rbm<dbn::layer<dbn::conf<true, 50, true, false, dbn::DecayType::NONE, false, dbn::Type::GAUSSIAN, dbn::Type::NRLU, pcd2_trainer_t>, 100, 100>> rbm_4;

    //PCD-2 and sparsity

    typedef dbn::rbm<dbn::layer<dbn::conf<true, 50, true, false, dbn::DecayType::NONE, true, dbn::Type::GAUSSIAN, dbn::Type::SIGMOID, pcd2_trainer_t>, 100, 100>> rbm_5;

    //Test them all

    test_rbm<rbm_1>();
    test_rbm<rbm_2>();
    test_rbm<rbm_3>();
    test_rbm<rbm_4>();
    test_rbm<rbm_5>();

    return 0;
}