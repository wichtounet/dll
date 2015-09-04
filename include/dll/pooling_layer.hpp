//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_POOLING_LAYER_INL
#define DLL_POOLING_LAYER_INL

#include "etl/etl.hpp"

#include "neural_base.hpp"

namespace dll {

/*!
 * \brief Standard pooling layer
 */
template<typename Parent, typename Desc>
struct pooling_layer_3d : neural_base<Parent> {
    using desc = Desc;
    using weight = typename desc::weight;

    static constexpr const std::size_t I1 = desc::I1;
    static constexpr const std::size_t I2 = desc::I2;
    static constexpr const std::size_t I3 = desc::I3;
    static constexpr const std::size_t C1 = desc::C1;
    static constexpr const std::size_t C2 = desc::C2;
    static constexpr const std::size_t C3 = desc::C3;

    static constexpr const std::size_t O1 = I1 / C1;
    static constexpr const std::size_t O2 = I2 / C2;
    static constexpr const std::size_t O3 = I3 / C3;

    static constexpr const bool is_nop = C1 + C2 + C3 == 1;

    using input_one_t = etl::fast_dyn_matrix<weight, I1, I2, I3>;
    using output_one_t = etl::fast_dyn_matrix<weight, O1, O2, O3>;
    using input_t = std::vector<input_one_t>;
    using output_t = std::vector<output_one_t>;

    template<std::size_t B>
    using input_batch_t = etl::fast_dyn_matrix<weight, B, I1, I2, I3>;

    template<std::size_t B>
    using output_batch_t = etl::fast_dyn_matrix<weight, B, O1, O2, O3>;

    pooling_layer_3d() = default;

    static constexpr std::size_t input_size() noexcept {
        return I1 * I2 * I3;
    }

    static constexpr std::size_t output_size() noexcept {
        return O1 * O2 * O3;
    }

    static constexpr std::size_t parameters() noexcept {
        return 0;
    }

    template<typename Input>
    static output_t prepare_output(std::size_t samples){
        return output_t{samples};
    }

    template<typename Input>
    static output_one_t prepare_one_output(){
        return output_one_t();
    }
};

} //end of dll namespace

#endif
