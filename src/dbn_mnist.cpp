//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <memory>

#include "dbn/dbn.hpp"
#include "dbn/layer.hpp"
#include "dbn/conf.hpp"
#include "dbn/image_utils.hpp"
#include "dbn/labels.hpp"
#include "dbn/test.hpp"

#include "mnist/mnist_reader.hpp"

namespace {

template<typename DBN, typename P>
void test_all(DBN& dbn, std::vector<vector<double>>& training_images, const std::vector<uint8_t>& training_labels, P&& predictor){
    auto test_images = mnist::read_test_images<std::vector, vector, double>();
    auto test_labels = mnist::read_test_labels<std::vector>();

    if(test_images.empty() || test_labels.empty()){
        std::cout << "Impossible to read test set" << std::endl;
        return;
    }

    std::cout << "Start testing" << std::endl;

    std::cout << "Training Set" << std::endl;
    auto error_rate = dbn::test_set(dbn, training_images, training_labels, predictor);
    std::cout << "\tError rate (normal): " << 100.0 * error_rate << std::endl;

    std::cout << "Test Set" << std::endl;
    error_rate =  dbn::test_set(dbn, test_images, test_labels, predictor);
    std::cout << "\tError rate (normal): " << 100.0 * error_rate << std::endl;
}

} //end of anonymous namespace

int main(int argc, char* argv[]){
    auto simple = false;
    auto load = false;

    for(int i = 1; i < argc; ++i){
        std::string command(argv[i]);

        if(command == "simple"){
            simple = true;
        }

        if(command == "load"){
            load = true;
        }
    }

    auto training_images = mnist::read_training_images<std::vector, vector, double>();
    auto training_labels = mnist::read_training_labels<std::vector>();

    if(training_images.empty() || training_labels.empty()){
        return 1;
    }

    binarize_each(training_images);

    if(simple){
        typedef dbn::dbn<
            dbn::layer<dbn::conf<true, 50, true>, 28 * 28, 50>,
            dbn::layer<dbn::conf<true, 50, false>, 50, 50>,
            dbn::layer<dbn::conf<true, 50, false>, 60, 100>> dbn_simple_t;

        auto dbn = std::make_shared<dbn_simple_t>();

        dbn->train_with_labels(training_images, training_labels, 10, 5);

        test_all(dbn, training_images, training_labels, dbn::label_predictor());
    } else {
        typedef dbn::dbn<
            dbn::layer<dbn::conf<true, 100, true, true>, 28 * 28, 30>,
            //dbn::layer<dbn::conf<true, 100, false, true>, 300, 300>,
            dbn::layer<dbn::conf<true, 100, false, true>, 30, 30>,
            dbn::layer<dbn::conf<true, 100, false, true, true, dbn::Type::EXP>, 30, 10>> dbn_t;

        auto labels = dbn::make_fake(training_labels);

        auto dbn = std::make_unique<dbn_t>();

        dbn->display();

        if(load){
            std::cout << "Load from file" << std::endl;

            std::ifstream is("dbn.dat", std::ifstream::binary);
            dbn->load(is);
        } else {
            std::cout << "Start pretraining" << std::endl;
            dbn->pretrain(training_images, 5);

            std::cout << "Start fine-tuning" << std::endl;
            dbn->fine_tune(training_images, labels, 5, 1000);

            std::ofstream os("dbn.dat", std::ofstream::binary);
            dbn->store(os);
        }

        test_all(dbn, training_images, training_labels, dbn::predictor());
    }

    return 0;
}
