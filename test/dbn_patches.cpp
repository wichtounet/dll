//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

//Test for DBN and patches layers

#include <deque>

#include "catch.hpp"

#include "dll/conv_rbm.hpp"
#include "dll/dbn.hpp"

#include "dll/patches_layer.hpp"
#include "dll/patches_layer_pad.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

TEST_CASE( "dbn/mnist/patches/1", "[dbn][conv][mnist][patches]" ) {
    typedef dll::dbn_desc<
        dll::dbn_layers<
        dll::patches_layer_desc<14, 14, 14, 14>::layer_t,
        dll::conv_rbm_desc_square<1, 14, 20, 10, dll::momentum, dll::batch_size<25>>::rbm_t,
        dll::conv_rbm_desc_square<20, 10, 20, 6, dll::momentum, dll::batch_size<25>>::rbm_t>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(500);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    std::vector<etl::dyn_matrix<double, 3>> converted;
    converted.reserve(dataset.training_images.size());

    for(auto& image : dataset.training_images){
        converted.emplace_back(1, 28, 28);
        converted.back() = image;
    }

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(converted, 20);

    auto probs = dbn->activation_probabilities(converted[0]);
    REQUIRE(probs.size() == 4);
}

TEST_CASE( "dbn/mnist/patches/2", "[dbn][conv][mnist][patches][memory]" ) {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::patches_layer_desc<14, 14, 14, 14>::layer_t,
            dll::conv_rbm_desc_square<1, 14, 20, 10, dll::momentum, dll::batch_size<25>>::rbm_t,
            dll::conv_rbm_desc_square<20, 10, 20, 6, dll::momentum, dll::batch_size<25>>::rbm_t>,
        dll::memory>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(500);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    std::vector<etl::dyn_matrix<double, 3>> converted;
    converted.reserve(dataset.training_images.size());

    for(auto& image : dataset.training_images){
        converted.emplace_back(1, 28, 28);
        converted.back() = image;
    }

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(converted, 20);

    auto probs = dbn->activation_probabilities(converted[0]);
    REQUIRE(probs.size() == 4);
}

TEST_CASE( "dbn/mnist/patches/3", "[dbn][conv][mnist][patches]" ) {
    typedef dll::dbn_desc<
        dll::dbn_layers<
        dll::patches_layer_padh_desc<14, 14, 14, 14, 1>::layer_t,
        dll::conv_rbm_desc_square<1, 14, 20, 10, dll::momentum, dll::batch_size<25>>::rbm_t,
        dll::conv_rbm_desc_square<20, 10, 20, 6, dll::momentum, dll::batch_size<25>>::rbm_t>>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(500);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    std::vector<etl::dyn_matrix<double, 3>> converted;
    converted.reserve(dataset.training_images.size());

    for(auto& image : dataset.training_images){
        converted.emplace_back(1, 28, 28);
        converted.back() = image;
    }

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(converted, 20);

    auto probs = dbn->activation_probabilities(converted[0]);
    REQUIRE(probs.size() == 4);
}

TEST_CASE( "dbn/mnist/patches/4", "[dbn][conv][mnist][patches][memory]" ) {
    typedef dll::dbn_desc<
        dll::dbn_layers<
            dll::patches_layer_padh_desc<14, 14, 14, 14, 1>::layer_t,
            dll::conv_rbm_desc_square<1, 14, 20, 10, dll::momentum, dll::batch_size<25>>::rbm_t,
            dll::conv_rbm_desc_square<20, 10, 20, 6, dll::momentum, dll::batch_size<25>>::rbm_t>,
        dll::memory>::dbn_t dbn_t;

    auto dataset = mnist::read_dataset<std::vector, std::vector, double>(500);

    REQUIRE(!dataset.training_images.empty());

    mnist::binarize_dataset(dataset);

    std::vector<etl::dyn_matrix<double, 3>> converted;
    converted.reserve(dataset.training_images.size());

    for(auto& image : dataset.training_images){
        converted.emplace_back(1, 28, 28);
        converted.back() = image;
    }

    auto dbn = std::make_unique<dbn_t>();

    dbn->pretrain(converted, 20);

    auto probs = dbn->activation_probabilities(converted[0]);
    REQUIRE(probs.size() == 4);
}
