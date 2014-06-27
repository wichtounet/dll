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
        dbn::layer<dbn::conf<true, 50, true, false, dbn::DecayType::L2, true>, 28 * 28, 100>,
        dbn::layer<dbn::conf<true, 50, false, false, dbn::DecayType::L2, true>, 100, 100>,
        dbn::layer<dbn::conf<true, 50, false, false, dbn::DecayType::L2, true>, 110, 200>> dbn_1;

    //Test them all

    test_dbn<dbn_1>();

    return 0;
}