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
    typedef dbn::rbm<dbn::layer<100, 100, dbn::weight_decay<dbn::DecayType::L2>>> rbm_1;

    //Mix units
    typedef dbn::rbm<dbn::layer<100, 100, dbn::momentum, dbn::batch_size<50>, dbn::visible_unit<dbn::Type::GAUSSIAN>, dbn::hidden_unit<dbn::Type::NRLU>>> rbm_2;

    //Sparsity
    typedef dbn::rbm<dbn::layer<100, 100, dbn::momentum, dbn::sparsity>> rbm_3;

    //PCD-2

    typedef dbn::rbm<dbn::layer<100, 100, dbn::trainer<pcd2_trainer_t>>> rbm_4;

    //PCD-2 and sparsity

    typedef dbn::rbm<dbn::layer<100, 100, dbn::trainer<pcd2_trainer_t>, dbn::sparsity>> rbm_5;

    //Test them all

    test_rbm<rbm_1>();
    test_rbm<rbm_2>();
    test_rbm<rbm_3>();
    test_rbm<rbm_4>();
    test_rbm<rbm_5>();

    return 0;
}