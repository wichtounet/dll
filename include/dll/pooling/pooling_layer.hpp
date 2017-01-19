//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/etl.hpp"

#include "dll/layer.hpp"

namespace dll {

/*!
 * \brief Standard pooling layer
 */
template <typename Parent, typename Desc>
struct pooling_layer_3d : layer<Parent> {
    using desc   = Desc;
    using weight = typename desc::weight;

    static constexpr const std::size_t I1 = desc::I1; ///< The first dimension of the input
    static constexpr const std::size_t I2 = desc::I2; ///< The second dimension of the input
    static constexpr const std::size_t I3 = desc::I3; ///< The third dimension of the input
    static constexpr const std::size_t C1 = desc::C1; ///< The first dimension pooling ratio
    static constexpr const std::size_t C2 = desc::C2; ///< The second dimension pooling ratio
    static constexpr const std::size_t C3 = desc::C3; ///< The third dimension pooling ratio

    static constexpr const std::size_t O1 = I1 / C1; ///< The first dimension of the output
    static constexpr const std::size_t O2 = I2 / C2; ///< The second dimension of the output
    static constexpr const std::size_t O3 = I3 / C3; ///< The third dimension of the output

    static constexpr const bool is_nop = C1 * C2 * C3 == 1; ///< Indicate if the operation has no effect

    using input_one_t  = etl::fast_dyn_matrix<weight, I1, I2, I3>;
    using output_one_t = etl::fast_dyn_matrix<weight, O1, O2, O3>;
    using input_t      = std::vector<input_one_t>;
    using output_t     = std::vector<output_one_t>;

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

    /*!
     * \brief Apply the layer to many inputs
     * \param output The set of output
     * \param input The set of input to apply the layer to
     */
    template <typename I, typename O_A>
    void activate_many(const I& input, O_A& output) const {
        for (std::size_t i = 0; i < input.size(); ++i) {
            as_derived().activate_hidden(input[i], output[i]);
        }
    }

    template <typename Input>
    static output_t prepare_output(std::size_t samples) {
        return output_t{samples};
    }

    template <typename Input>
    static output_one_t prepare_one_output() {
        return output_one_t();
    }

private:
    const Parent& as_derived() const {
        return *static_cast<const Parent*>(this);
    }
};

/*!
 * \brief Standard dynamic pooling layer
 */
template <typename Parent, typename Desc>
struct dyn_pooling_layer_3d : layer<Parent> {
    using desc   = Desc;
    using weight = typename desc::weight;

    static constexpr const bool is_nop = false; ///< Indicate if the operation has no effect

    using input_one_t  = etl::dyn_matrix<weight, 3>;
    using output_one_t = etl::dyn_matrix<weight, 3>;
    using input_t      = std::vector<input_one_t>;
    using output_t     = std::vector<output_one_t>;

    std::size_t i1; ///< The first dimension of the input
    std::size_t i2; ///< The second dimension of the input
    std::size_t i3; ///< The third dimension of the input
    std::size_t c1; ///< The first dimension pooling ratio
    std::size_t c2; ///< The second dimension pooling ratio
    std::size_t c3; ///< The third dimension pooling ratio

    std::size_t o1; ///< The first dimension of the output
    std::size_t o2; ///< The second dimension of the output
    std::size_t o3; ///< The third dimension of the output

    dyn_pooling_layer_3d() = default;

    void init_layer(size_t i1, size_t i2, size_t i3, size_t c1, size_t c2, size_t c3){
        this->i1 = i1;
        this->i2 = i2;
        this->i3 = i3;
        this->c1 = c1;
        this->c2 = c2;
        this->c3 = c3;
        this->o1 = i1 / c1;
        this->o2 = i2 / c2;
        this->o3 = i3 / c3;
    }

    std::size_t input_size() const noexcept {
        return i1 * i2 * i3;
    }

    std::size_t output_size() const noexcept {
        return o1 * o2 * o3;
    }

    std::size_t parameters() const noexcept {
        return 0;
    }

    /*!
     * \brief Apply the layer to many inputs
     * \param output The set of output
     * \param input The set of input to apply the layer to
     */
    template <typename I, typename O_A>
    void activate_many(const I& input, O_A& output) const {
        for (std::size_t i = 0; i < input.size(); ++i) {
            as_derived().activate_hidden(input[i], output[i]);
        }
    }

    template <typename Input>
    output_t prepare_output(std::size_t samples) const {
        output_t output;
        output.reserve(samples);
        for(size_t i = 0; i < samples; ++i){
            output.emplace_back(o1, o2, o3);
        }
        return output;
    }

    template <typename Input>
    output_one_t prepare_one_output() const {
        return output_one_t(o1, o2, o3);
    }

private:
    const Parent& as_derived() const {
        return *static_cast<const Parent*>(this);
    }
};

} //end of dll namespace
