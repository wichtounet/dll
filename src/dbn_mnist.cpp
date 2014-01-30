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
    auto prediction = false;

    if(argc > 1){
        std::string command(argv[1]);

        if(command == "predict"){
            prediction = true;
        }
    }

    auto training_images = mnist::read_training_images();

    binarize_each(training_images);

    typedef dbn::dbn<dbn::conf<>, dbn::layer<28 * 28, 500>, dbn::layer<500, 2000>, dbn::layer<2000, 10>> dbn_t;

    auto dbn = std::make_shared<dbn_t>();

    dbn->train(training_images, 5);

    //TODO Train

    if(prediction){
        //TODO
    }

    return 0;
}
