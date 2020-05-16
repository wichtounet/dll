//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "dll_test.hpp"

#include "dll/text_reader.hpp"

DLL_TEST_CASE("unit/text_reader/labels/1", "[unit][reader]") {
    auto labels = dll::text::read_labels<std::vector, uint8_t>("test/text_db/labels", 20);

    REQUIRE(labels.size() == 9);

    REQUIRE(labels[0] == 7);
    REQUIRE(labels[1] == 2);
    REQUIRE(labels[2] == 1);
    REQUIRE(labels[3] == 0);
    REQUIRE(labels[4] == 4);
    REQUIRE(labels[5] == 1);
    REQUIRE(labels[6] == 4);
    REQUIRE(labels[7] == 9);
    REQUIRE(labels[8] == 5);
}

DLL_TEST_CASE("unit/text_reader/labels/2", "[unit][reader]") {
    auto labels = dll::text::read_labels<std::vector, char>("test/text_db/labels", 5);

    REQUIRE(labels.size() == 5);

    REQUIRE(labels[0] == 7);
    REQUIRE(labels[1] == 2);
    REQUIRE(labels[2] == 1);
    REQUIRE(labels[3] == 0);
    REQUIRE(labels[4] == 4);
}

DLL_TEST_CASE("unit/text_reader/images/1", "[unit][reader]") {
    auto samples = dll::text::read_images<std::vector, std::vector<uint8_t>, false>("test/text_db/images", 20);

    REQUIRE(samples.size() == 9);

    for(auto& sample : samples){
        REQUIRE(sample.size() == 28 * 28);
    }

    REQUIRE(samples[0][17 * 28 + 16] == 254);
    REQUIRE(samples[1][15 * 28 + 12] == 189);
    REQUIRE(samples[2][16 * 28 + 13] == 232);
    REQUIRE(samples[3][ 9 * 28 + 13] == 253);
    REQUIRE(samples[4][17 * 28 + 16] == 251);
    REQUIRE(samples[5][16 * 28 + 13] == 254);
    REQUIRE(samples[6][17 * 28 + 15] == 254);
    REQUIRE(samples[7][17 * 28 + 16] == 9);
    REQUIRE(samples[8][17 * 28 + 15] == 253);
}

DLL_TEST_CASE("unit/text_reader/images/2", "[unit][reader]") {
    auto samples = dll::text::read_images<std::vector, std::vector<uint8_t>, false>("test/text_db/images", 4);

    REQUIRE(samples.size() == 4);
}

DLL_TEST_CASE("unit/text_reader/images/3", "[unit][reader]") {
    std::vector<std::vector<uint8_t>> samples;
    dll::text::read_images_direct<false>(samples, "test/text_db/images", 20);

    REQUIRE(samples.size() == 9);

    for(auto& sample : samples){
        REQUIRE(sample.size() == 28 * 28);
    }

    REQUIRE(samples[0][17 * 28 + 16] == 254);
    REQUIRE(samples[1][15 * 28 + 12] == 189);
    REQUIRE(samples[2][16 * 28 + 13] == 232);
    REQUIRE(samples[3][ 9 * 28 + 13] == 253);
    REQUIRE(samples[4][17 * 28 + 16] == 251);
    REQUIRE(samples[5][16 * 28 + 13] == 254);
    REQUIRE(samples[6][17 * 28 + 15] == 254);
    REQUIRE(samples[7][17 * 28 + 16] == 9);
    REQUIRE(samples[8][17 * 28 + 15] == 253);
}

DLL_TEST_CASE("unit/text_reader/images/4", "[unit][reader]") {
    std::vector<etl::dyn_matrix<float, 3>> samples;
    dll::text::read_images_direct<true>(samples, "test/text_db/images", 20);

    REQUIRE(samples.size() == 9);

    for(auto& sample : samples){
        REQUIRE(sample.size() == 1 * 28 * 28);
        REQUIRE(sample.dim(0) == 1);
        REQUIRE(sample.dim(1) == 28);
        REQUIRE(sample.dim(2) == 28);
    }

    REQUIRE(samples[0](0, 17, 16) == 254);
    REQUIRE(samples[1](0, 15, 12) == 189);
    REQUIRE(samples[2](0, 16, 13) == 232);
    REQUIRE(samples[3](0,  9, 13) == 253);
    REQUIRE(samples[4](0, 17, 16) == 251);
    REQUIRE(samples[5](0, 16, 13) == 254);
    REQUIRE(samples[6](0, 17, 15) == 254);
    REQUIRE(samples[7](0, 17, 16) == 9);
    REQUIRE(samples[8](0, 17, 15) == 253);
}

DLL_TEST_CASE("unit/text_reader/images/5", "[unit][reader]") {
    std::vector<etl::fast_dyn_matrix<float, 1, 28, 28>> samples;
    dll::text::read_images_direct<true>(samples, "test/text_db/images", 20);

    REQUIRE(samples.size() == 9);

    REQUIRE(samples[0](0, 17, 16) == 254);
    REQUIRE(samples[1](0, 15, 12) == 189);
    REQUIRE(samples[2](0, 16, 13) == 232);
    REQUIRE(samples[3](0,  9, 13) == 253);
    REQUIRE(samples[4](0, 17, 16) == 251);
    REQUIRE(samples[5](0, 16, 13) == 254);
    REQUIRE(samples[6](0, 17, 15) == 254);
    REQUIRE(samples[7](0, 17, 16) == 9);
    REQUIRE(samples[8](0, 17, 15) == 253);
}
