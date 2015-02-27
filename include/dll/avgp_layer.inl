//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_AVGP_LAYER_INL
#define DLL_AVGP_LAYER_INL

#include "etl/etl.hpp"

namespace dll {

/*!
 * \brief Standard average pooling layer
 *
 * This follows the definition of a RBM by Geoffrey Hinton.
 */
template<typename Desc>
struct avgp_layer_3d final {
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

    avgp_layer_3d() = default;

    static constexpr std::size_t input_size() noexcept {
        return I1 * I2 * I3;
    }

    static constexpr std::size_t output_size() noexcept {
        return O1 * O2 * O3;
    }

    static constexpr std::size_t parameters() noexcept {
        return 0;
    }

    static std::string to_short_string(){
        char buffer[1024];
        snprintf(buffer, 1024, "MP(3D): %lux%lux%lu -> (%lux%lux%lu) -> %lux%lux%lu", I1, I2, I3, C1, C2, C3, O1, O2, O3);
        return {buffer};
    }

    static void display(){
        std::cout << to_short_string() << std::endl;
    }

    template<typename H, typename V>
    static void max_pool(H&& h, const V& v){
        for(std::size_t i = 0; i < O1; ++i){
            for(std::size_t j = 0; j < O2; ++j){
                for(std::size_t k = 0; k < O3; ++k){
                    weight avg = 0;

                    for(std::size_t ii = 0; ii < C1; ++ii){
                        for(std::size_t jj = 0; jj < C2; ++jj){
                            for(std::size_t kk = 0; kk < C3; ++kk){
                                avg += v(i * C1 + ii, j * C2 + jj, k * C3 + kk);
                            }
                        }
                    }

                    h(i,j,k) = avg / static_cast<weight>(C1 * C2 * C3);
                }
            }
        }
    }

    //TODO Ideally, the dbn should guess by TMP that Max Pooling don't need any training

    template<typename Samples, bool EnableWatcher = true, typename RW = void, typename... Args>
    double train(const Samples& /*training_data*/, std::size_t /*max_epochs*/, Args... /*args*/){
        return 1.0;
    }

    template<typename Iterator, bool EnableWatcher = true, typename RW = void, typename... Args>
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

    //TODO Ideally, the dbn should guess if h_a/h_a are used or only h_a
    static void activate_one(const input_one_t& input, output_one_t& h, output_one_t& /*h_s*/){
        max_pool(h, input);
    }

    static void activate_many(const input_t& input, output_t& h_a, output_t& h_s){
        for(std::size_t i = 0; i < input.size(); ++i){
            activate_one(input[i], h_a[i], h_s[i]);
        }
    }
};

//Allow odr-use of the constexpr static members

template<typename Desc>
const std::size_t avgp_layer_3d<Desc>::I1;

template<typename Desc>
const std::size_t avgp_layer_3d<Desc>::I2;

template<typename Desc>
const std::size_t avgp_layer_3d<Desc>::I3;

template<typename Desc>
const std::size_t avgp_layer_3d<Desc>::C1;

template<typename Desc>
const std::size_t avgp_layer_3d<Desc>::C2;

template<typename Desc>
const std::size_t avgp_layer_3d<Desc>::C3;

template<typename Desc>
const std::size_t avgp_layer_3d<Desc>::O1;

template<typename Desc>
const std::size_t avgp_layer_3d<Desc>::O2;

template<typename Desc>
const std::size_t avgp_layer_3d<Desc>::O3;

} //end of dll namespace

#endif
