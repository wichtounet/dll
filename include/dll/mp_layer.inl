//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_MP_LAYER_INL
#define DLL_MP_LAYER_INL

#include "etl/etl.hpp"

#include "pooling_layer.hpp"

namespace dll {

/*!
 * \brief Standard max pooling layer
 *
 * This follows the definition of a RBM by Geoffrey Hinton.
 */
template<typename Desc>
struct mp_layer_3d final : pooling_layer_3d<Desc> {
    using desc = Desc;
    using weight = double; //This should be configurable or TMP computed
    using base = pooling_layer_3d<desc>;

    mp_layer_3d() = default;

    static std::string to_short_string(){
        char buffer[1024];
        snprintf(buffer, 1024, "MP(3D): %lux%lux%lu -> (%lux%lux%lu) -> %lux%lux%lu", 
            base::I1, base::I2, base::I3, base::C1, base::C2, base::C3, base::O1, base::O2, base::O3);
        return {buffer};
    }

    static void display(){
        std::cout << to_short_string() << std::endl;
    }

    using input_one_t = etl::dyn_matrix<weight, 3>;
    using output_one_t = etl::dyn_matrix<weight, 3>;
    using input_t = std::vector<input_one_t>;
    using output_t = std::vector<output_one_t>;

    //TODO Ideally, the dbn should guess if h_a/h_a are used or only h_a
    static void activate_one(const input_one_t& v, output_one_t& h, output_one_t& /*h_s*/){
        for(std::size_t i = 0; i < base::O1; ++i){
            for(std::size_t j = 0; j < base::O2; ++j){
                for(std::size_t k = 0; k < base::O3; ++k){
                    auto max = v(i * base::C1, j * base::C2, k * base::C3);

                    for(std::size_t ii = 0; ii < base::C1; ++ii){
                        for(std::size_t jj = 0; jj < base::C2; ++jj){
                            for(std::size_t kk = 0; kk < base::C3; ++kk){
                                max = std::max(max, v(i * base::C1 + ii, j * base::C2 + jj, k * base::C3 + kk));
                            }
                        }
                    }

                    h(i,j,k) = max;
                }
            }
        }
    }

    static void activate_many(const input_t& input, output_t& h_a, output_t& h_s){
        for(std::size_t i = 0; i < input.size(); ++i){
            activate_one(input[i], h_a[i], h_s[i]);
        }
    }
};

} //end of dll namespace

#endif
