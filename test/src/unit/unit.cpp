//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "dll_test.hpp"

#define DLL_DETAIL_ONLY

#include "dll/rbm/rbm.hpp"
#include "dll/ocv_visualizer.hpp"

TEST_CASE("unit/unit_1", "[unit]") {
    REQUIRE(dll::detail::ct_sqrt(1) == 1);
    REQUIRE(dll::detail::ct_sqrt(4) == 2);
    REQUIRE(dll::detail::ct_sqrt(9) == 3);
    REQUIRE(dll::detail::ct_sqrt(144) == 12);
    REQUIRE(dll::detail::ct_sqrt(39601) == 199);
}

TEST_CASE("unit/unit_2", "[unit]") {
    REQUIRE(dll::detail::ct_pow(1) == 1);
    REQUIRE(dll::detail::ct_pow(2) == 4);
    REQUIRE(dll::detail::ct_pow(4) == 16);
    REQUIRE(dll::detail::ct_pow(199) == 39601);
}

TEST_CASE("unit/unit_3", "[unit]") {
    REQUIRE(dll::detail::best_height(1) == 1);
    REQUIRE(dll::detail::best_height(4) == 2);
    REQUIRE(dll::detail::best_height(9) == 3);
    REQUIRE(dll::detail::best_height(144) == 12);
    REQUIRE(dll::detail::best_height(400) == 20);
    REQUIRE(dll::detail::best_height(405) == 20);
    REQUIRE(dll::detail::best_height(21 * 21) == 21);
    REQUIRE(dll::detail::best_height(200) == 14);
}

TEST_CASE("unit/unit_4", "[unit]") {
    REQUIRE(dll::detail::best_width(1) == 1);
    REQUIRE(dll::detail::best_width(4) == 2);
    REQUIRE(dll::detail::best_width(9) == 3);
    REQUIRE(dll::detail::best_width(144) == 12);
    REQUIRE(dll::detail::best_width(400) == 20);
    REQUIRE(dll::detail::best_width(200) == 15);
}

TEST_CASE("unit/unit_5", "[unit]") {
    //200 => 15 x 14
    REQUIRE(dll::detail::best_width(200) == 15);
    REQUIRE(dll::detail::best_height(200) == 14);

    //400 => 20 x 20
    REQUIRE(dll::detail::best_width(400) == 20);
    REQUIRE(dll::detail::best_height(400) == 20);

    //2000 => 45 x 45
    REQUIRE(dll::detail::best_width(2000) == 45);
    REQUIRE(dll::detail::best_height(2000) == 45);

    //3000 => 55 x 55
    REQUIRE(dll::detail::best_width(3000) == 55);
    REQUIRE(dll::detail::best_height(3000) == 55);

    //444444 => 666 x 666
    REQUIRE(dll::detail::best_width(444444) == 667);
    REQUIRE(dll::detail::best_height(444444) == 667);
}
