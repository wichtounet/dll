#include "dll/rbm.hpp"
#include "dll/generic_trainer.hpp"

template<typename RBM>
void test_rbm(){
    RBM rbm;

    std::vector<vector<double>> training;
    rbm.train(training, 10);
}

template <typename RBM>
using pcd2_trainer_t = dll::persistent_cd_trainer<2, RBM>;

int main(){
    //Very basic RBM that must compile
    typedef dll::rbm<dll::layer<100, 100, dll::weight_decay<dll::DecayType::L2>>> rbm_1;

    //Mix units
    typedef dll::rbm<dll::layer<100, 100, dll::momentum, dll::batch_size<50>, dll::visible_unit<dll::Type::GAUSSIAN>, dll::hidden_unit<dll::Type::NRLU>>> rbm_2;

    //Sparsity
    typedef dll::rbm<dll::layer<100, 100, dll::momentum, dll::sparsity>> rbm_3;

    //PCD-2

    typedef dll::rbm<dll::layer<100, 100, dll::trainer<pcd2_trainer_t>>> rbm_4;

    //PCD-2 and sparsity

    typedef dll::rbm<dll::layer<100, 100, dll::trainer<pcd2_trainer_t>, dll::sparsity>> rbm_5;

    //Test them all

    test_rbm<rbm_1>();
    test_rbm<rbm_2>();
    test_rbm<rbm_3>();
    test_rbm<rbm_4>();
    test_rbm<rbm_5>();

    return 0;
}