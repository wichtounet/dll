#include "dbn/dbn.hpp"

template<typename DBN>
void test_dbn(){
    DBN dbn;

    //TODO Train
}

template <typename RBM>
using pcd2_trainer_t = dbn::persistent_cd_trainer<2, RBM>;

int main(){
    //Basic example

    typedef dbn::dbn<
        dbn::layer<28 * 28, 100, dbn::momentum, dbn::batch_size<50>, dbn::init_weights, dbn::weight_decay<dbn::DecayType::L2>, dbn::sparsity>,
        dbn::layer<100, 100, dbn::momentum, dbn::batch_size<50>>,
        dbn::layer<110, 200, dbn::batch_size<50>, dbn::momentum, dbn::weight_decay<dbn::DecayType::L2_FULL>>
    > dbn_1;

    //Test them all

    test_dbn<dbn_1>();

    return 0;
}