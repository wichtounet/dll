//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <memory>

#include "dbn.hpp"
#include "layer.hpp"
#include "conf.hpp"
#include "mnist_reader.hpp"
#include "image_utils.hpp"

int main(int argc, char* argv[]){
    auto simple = false;

    if(argc > 1){
        std::string command(argv[1]);

        if(command == "simple"){
            simple = true;
        }
    }

    auto training_images = mnist::read_training_images();
    auto training_labels = mnist::read_training_labels();

    if(training_images.empty() || training_labels.empty()){
        return 1;
    }

    auto test_images = mnist::read_test_images();
    auto test_labels = mnist::read_test_labels();

    if(test_images.empty() || test_labels.empty()){
        return 1;
    }

    binarize_each(training_images);

    if(simple){
        typedef dbn::dbn<
            dbn::layer<dbn::conf<true, 50, true>, 28 * 28, 300>,
            dbn::layer<dbn::conf<true, 50, false>, 300, 300>,
            dbn::layer<dbn::conf<true, 50, false>, 310, 500>> dbn_simple_t;

        auto dbn = std::make_shared<dbn_simple_t>();

        std::cout << "Start pretraining" << std::endl;

        dbn->pretrain_with_labels(training_images, training_labels, 10, 5);

        std::cout << "Start testing" << std::endl;
        stop_watch<std::chrono::milliseconds> watch;

        size_t success = 0;
        for(size_t i = 0; i < training_images.size(); ++i){
            auto& image = training_images[i];
            auto& label = training_labels[i];

            auto predicted = dbn->predict(image, 10);

            if(predicted == label){
                ++success;
            }
        }

        std::cout << "Training Set Error rate: " << 100.0 * ((training_images.size() - success) / static_cast<double>(training_images.size())) << std::endl;

        success = 0;
        for(size_t i = 0; i < test_images.size(); ++i){
            auto& image = test_images[i];
            auto& label = test_labels[i];

            auto predicted = dbn->predict(image, 10);

            if(predicted == label){
                ++success;
            }
        }

        std::cout << "Test Set Error rate: " << 100.0 * ((test_images.size() - success) / static_cast<double>(test_images.size())) << std::endl;

        std::cout << "Testing took " << watch.elapsed() << "ms" << std::endl;
    } else {
        typedef dbn::dbn<
            dbn::layer<dbn::conf<true, 50, true>, 28 * 28, 50>,
            dbn::layer<dbn::conf<true, 50, false>, 50, 50>,
            dbn::layer<dbn::conf<true, 50, false>, 50, 100>,
            dbn::layer<dbn::conf<true, 50, false>, 100, 10>> dbn_t;

        auto dbn = std::make_shared<dbn_t>();

        std::cout << "Start pretraining" << std::endl;

        dbn->pretrain(training_images, 5);
    }

    return 0;
}
