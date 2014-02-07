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

int main(int, char*[]){
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

    typedef dbn::dbn<
        dbn::layer<dbn::conf<true, 50, true>, 28 * 28, 100>,
        //dbn::layer<dbn::conf<true, 50, false>, 500, 500>,
        dbn::layer<dbn::conf<true, 50, false>, 110, 300>> dbn_t;

    auto dbn = std::make_shared<dbn_t>();

    std::cout << "Start training" << std::endl;

    dbn->train_with_labels(training_images, training_labels, 10, 10);

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

    std::cout << "Training Set Error rate: " << ((training_images.size() - success) / static_cast<double>(training_images.size())) << std::endl;

    success = 0;
    for(size_t i = 0; i < test_images.size(); ++i){
        auto& image = test_images[i];
        auto& label = test_labels[i];

        auto predicted = dbn->predict(image, 10);

        if(predicted == label){
            ++success;
        }
    }

    std::cout << "Test Set Error rate: " << ((test_images.size() - success) / static_cast<double>(test_images.size())) << std::endl;

    std::cout << "Training took " << watch.elapsed() << "ms" << std::endl;

    return 0;
}
