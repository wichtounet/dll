#include "dbn/dbn.hpp"

template<typename DBN>
void test_dbn(){
    DBN dbn;

    //TODO Train
}

template <typename RBM>
using pcd2_trainer_t = dll::persistent_cd_trainer<2, RBM>;

int main(){
    //Basic example

    typedef dll::dbn<
        dll::layer<28 * 28, 100, dll::momentum, dll::batch_size<50>, dll::init_weights, dll::weight_decay<dll::DecayType::L2>, dll::sparsity>,
        dll::layer<100, 100, dll::momentum, dll::batch_size<50>>,
        dll::layer<110, 200, dll::batch_size<50>, dll::momentum, dll::weight_decay<dll::DecayType::L2_FULL>>
    > dbn_1;

    //Test them all

    test_dbn<dbn_1>();

    return 0;
}