//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_POOLING_LAYER_INL
#define DLL_POOLING_LAYER_INL

#include "etl/etl.hpp"

namespace dll {

/*!
 * \brief Standard pooling layer
 */
template<typename Desc>
struct pooling_layer_3d {
    using desc = Desc;
    using weight = double; //This should be configurable or TMP computed

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

    //TODO Ideally, the dbn should guess by TMP that Max Pooling don't need any training

    template<bool EnableWatcher = true, typename RW = void, typename Samples, typename... Args>
    double train(const Samples& /*training_data*/, std::size_t /*max_epochs*/, Args... /*args*/){
        return 1.0;
    }

    template<bool EnableWatcher = true, typename RW = void, typename Iterator, typename... Args>
    double train(Iterator&& /*first*/, Iterator&& /*last*/, std::size_t /*max_epochs*/, Args... /*args*/){
        return 1.0;
    }

    using input_one_t = etl::dyn_matrix<weight, 3>;
    using output_one_t = etl::dyn_matrix<weight, 3>;
    using input_t = std::vector<input_one_t>;
    using output_t = std::vector<output_one_t>;

    template<typename Iterator>
    static auto convert_input(Iterator&& first, Iterator&& last){
        input_t input;
        input.reserve(std::distance(std::forward<Iterator>(first), std::forward<Iterator>(last)));

        std::for_each(std::forward<Iterator>(first), std::forward<Iterator>(last), [&input](auto& sample){
            input.emplace_back(I1, I2, I3);
            input.back() = sample;
        });

        return input;
    }

    template<typename Sample>
    static input_one_t convert_sample(const Sample& sample){
        input_one_t result(I1, I2, I3);
        result = sample;
        return result;
    }

    static output_t prepare_output(std::size_t samples){
        output_t output;
        output.reserve(samples);

        for(std::size_t i = 0; i < samples; ++i){
            output.emplace_back(O1, O2, O3);
        }

        return output;
    }

    static output_one_t prepare_one_output(){
        return output_one_t(O1, O2, O3);
    }
};

//Allow odr-use of the constexpr static members

template<typename Desc>
const std::size_t pooling_layer_3d<Desc>::I1;

template<typename Desc>
const std::size_t pooling_layer_3d<Desc>::I2;

template<typename Desc>
const std::size_t pooling_layer_3d<Desc>::I3;

template<typename Desc>
const std::size_t pooling_layer_3d<Desc>::C1;

template<typename Desc>
const std::size_t pooling_layer_3d<Desc>::C2;

template<typename Desc>
const std::size_t pooling_layer_3d<Desc>::C3;

template<typename Desc>
const std::size_t pooling_layer_3d<Desc>::O1;

template<typename Desc>
const std::size_t pooling_layer_3d<Desc>::O2;

template<typename Desc>
const std::size_t pooling_layer_3d<Desc>::O3;

} //end of dll namespace

#endif
