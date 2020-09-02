//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
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
struct pooling_2d_layer : layer<Parent> {
    using desc   = Desc; ///< The descriptor of the layer
    using weight = typename desc::weight; ///< The data type for this layer

    static constexpr size_t I1 = desc::I1; ///< The first dimension of the input
    static constexpr size_t I2 = desc::I2; ///< The second dimension of the input
    static constexpr size_t I3 = desc::I3; ///< The third dimension of the input
    static constexpr size_t C1 = desc::C1; ///< The first dimension pooling ratio
    static constexpr size_t C2 = desc::C2; ///< The second dimension pooling ratio
    static constexpr size_t S1 = desc::S1; ///< The first dimension stride
    static constexpr size_t S2 = desc::S2; ///< The second dimension stride
    static constexpr size_t P1 = desc::P1; ///< The first dimension stride
    static constexpr size_t P2 = desc::P2; ///< The second dimension stride

    static constexpr size_t O1 = I1;                      ///< The first dimension of the output
    static constexpr size_t O2 = (I2 - C1 + 2 * P1) / S1 + 1; ///< The second dimension of the output
    static constexpr size_t O3 = (I3 - C2 + 2 * P2) / S2 + 1; ///< The third dimension of the output

    static constexpr bool is_nop = C1 * C2 == 1 && P1 + P2 == 0 && S1 * S1 == 1; ///< Indicate if the operation has no effect

    using input_one_t  = etl::fast_dyn_matrix<weight, I1, I2, I3>; ///< The type of one input
    using output_one_t = etl::fast_dyn_matrix<weight, O1, O2, O3>; ///< The type of one output
    using input_t      = std::vector<input_one_t>; ///< The type of the input
    using output_t     = std::vector<output_one_t>; ///< The type of the output

    pooling_2d_layer() = default;

    /*!
     * \brief Return the size of the input of this layer
     * \return The size of the input of this layer
     */
    static constexpr size_t input_size() noexcept {
        return I1 * I2 * I3;
    }

    /*!
     * \brief Return the size of the output of this layer
     * \return The size of the output of this layer
     */
    static constexpr size_t output_size() noexcept {
        return O1 * O2 * O3;
    }

    /*!
     * \brief Return the number of trainable parameters of this network.
     * \return The the number of trainable parameters of this network.
     */
    static constexpr size_t parameters() noexcept {
        return 0;
    }

    /*!
     * \brief Prepare a set of empty outputs for this layer
     * \param samples The number of samples to prepare the output for
     * \return a container containing empty ETL matrices suitable to store samples output of this layer
     * \tparam Input The type of one input
     */
    template <typename Input>
    static output_t prepare_output(size_t samples) {
        return output_t{samples};
    }

    /*!
     * \brief Prepare one empty output for this layer
     * \return an empty ETL matrix suitable to store one output of this layer
     *
     * \tparam Input The type of one Input
     */
    template <typename Input>
    static output_one_t prepare_one_output() {
        return output_one_t();
    }

private:
    /*!
     * \brief Returns a reference to the derived object, i.e. the object using the CRTP injector.
     * \return a reference to the derived object.
     */
    const Parent& as_derived() const {
        return *static_cast<const Parent*>(this);
    }
};

/*!
 * \brief Standard dynamic pooling layer
 */
template <typename Parent, typename Desc>
struct dyn_pooling_2d_layer : layer<Parent> {
    using desc   = Desc; ///< The descriptor of the layer
    using weight = typename desc::weight; ///< The data type for this layer

    static constexpr bool is_nop = false; ///< Indicate if the operation has no effect

    using input_one_t  = etl::dyn_matrix<weight, 3>; ///< The type of one input
    using output_one_t = etl::dyn_matrix<weight, 3>; ///< The type of one output
    using input_t      = std::vector<input_one_t>; ///< The type of the input
    using output_t     = std::vector<output_one_t>; ///< The type of the output

    size_t i1; ///< The first dimension of the input
    size_t i2; ///< The second dimension of the input
    size_t i3; ///< The third dimension of the input

    size_t c1; ///< The first dimension pooling ratio
    size_t c2; ///< The second dimension pooling ratio

    size_t s1; ///< The first dimension stride
    size_t s2; ///< The second dimension stride

    size_t p1; ///< The first dimension pooling
    size_t p2; ///< The second dimension pooling

    size_t o1; ///< The first dimension of the output
    size_t o2; ///< The second dimension of the output
    size_t o3; ///< The third dimension of the output

    dyn_pooling_2d_layer() = default;

    /*!
     * \brief Initialize the dynamic layer
     */
    void init_layer(size_t i1, size_t i2, size_t i3, size_t c1, size_t c2, size_t s1, size_t s2, size_t p1, size_t p2){
        this->i1 = i1;
        this->i2 = i2;
        this->i3 = i3;
        this->c1 = c1;
        this->c2 = c2;
        this->s1 = s1;
        this->s2 = s2;
        this->p1 = p1;
        this->p2 = p2;
        this->o1 = i1;
        this->o2 = (i2 - c1 + 2 * p1) / s1 + 1;
        this->o3 = (i3 - c2 + 2 * p2) / s2 + 1;
    }

    /*!
     * \brief Return the size of the input of this layer
     * \return The size of the input of this layer
     */
    size_t input_size() const noexcept {
        return i1 * i2 * i3;
    }

    /*!
     * \brief Return the size of the output of this layer
     * \return The size of the output of this layer
     */
    size_t output_size() const noexcept {
        return o1 * o2 * o3;
    }

    /*!
     * \brief Return the number of trainable parameters of this network.
     * \return The the number of trainable parameters of this network.
     */
    size_t parameters() const noexcept {
        return 0;
    }

    /*!
     * \brief Prepare a set of empty outputs for this layer
     * \param samples The number of samples to prepare the output for
     * \return a container containing empty ETL matrices suitable to store samples output of this layer
     * \tparam Input The type of one input
     */
    template <typename Input>
    output_t prepare_output(size_t samples) const {
        output_t output;
        output.reserve(samples);
        for(size_t i = 0; i < samples; ++i){
            output.emplace_back(o1, o2, o3);
        }
        return output;
    }

    /*!
     * \brief Prepare one empty output for this layer
     * \return an empty ETL matrix suitable to store one output of this layer
     *
     * \tparam Input The type of one Input
     */
    template <typename Input>
    output_one_t prepare_one_output() const {
        return output_one_t(o1, o2, o3);
    }

private:
    /*!
     * \brief Returns a reference to the derived object, i.e. the object using the CRTP injector.
     * \return a reference to the derived object.
     */
    const Parent& as_derived() const {
        return *static_cast<const Parent*>(this);
    }
};

/*!
 * \brief Standard pooling layer
 */
template <typename Parent, typename Desc>
struct pooling_3d_layer : layer<Parent> {
    using desc   = Desc; ///< The descriptor of the layer
    using weight = typename desc::weight; ///< The data type for this layer

    static constexpr size_t I1 = desc::I1; ///< The first dimension of the input
    static constexpr size_t I2 = desc::I2; ///< The second dimension of the input
    static constexpr size_t I3 = desc::I3; ///< The third dimension of the input
    static constexpr size_t C1 = desc::C1; ///< The first dimension pooling ratio
    static constexpr size_t C2 = desc::C2; ///< The second dimension pooling ratio
    static constexpr size_t C3 = desc::C3; ///< The third dimension pooling ratio

    static constexpr size_t O1 = I1 / C1; ///< The first dimension of the output
    static constexpr size_t O2 = I2 / C2; ///< The second dimension of the output
    static constexpr size_t O3 = I3 / C3; ///< The third dimension of the output

    static constexpr bool is_nop = C1 * C2 * C3 == 1; ///< Indicate if the operation has no effect

    using input_one_t  = etl::fast_dyn_matrix<weight, I1, I2, I3>; ///< The type of one input
    using output_one_t = etl::fast_dyn_matrix<weight, O1, O2, O3>; ///< The type of one output
    using input_t      = std::vector<input_one_t>; ///< The type of the input
    using output_t     = std::vector<output_one_t>; ///< The type of the output

    pooling_3d_layer() = default;

    /*!
     * \brief Return the size of the input of this layer
     * \return The size of the input of this layer
     */
    static constexpr size_t input_size() noexcept {
        return I1 * I2 * I3;
    }

    /*!
     * \brief Return the size of the output of this layer
     * \return The size of the output of this layer
     */
    static constexpr size_t output_size() noexcept {
        return O1 * O2 * O3;
    }

    /*!
     * \brief Return the number of trainable parameters of this network.
     * \return The the number of trainable parameters of this network.
     */
    static constexpr size_t parameters() noexcept {
        return 0;
    }

    /*!
     * \brief Prepare a set of empty outputs for this layer
     * \param samples The number of samples to prepare the output for
     * \return a container containing empty ETL matrices suitable to store samples output of this layer
     * \tparam Input The type of one input
     */
    template <typename Input>
    static output_t prepare_output(size_t samples) {
        return output_t{samples};
    }

    /*!
     * \brief Prepare one empty output for this layer
     * \return an empty ETL matrix suitable to store one output of this layer
     *
     * \tparam Input The type of one Input
     */
    template <typename Input>
    static output_one_t prepare_one_output() {
        return output_one_t();
    }

private:
    /*!
     * \brief Returns a reference to the derived object, i.e. the object using the CRTP injector.
     * \return a reference to the derived object.
     */
    const Parent& as_derived() const {
        return *static_cast<const Parent*>(this);
    }
};

/*!
 * \brief Standard dynamic pooling layer
 */
template <typename Parent, typename Desc>
struct dyn_pooling_3d_layer : layer<Parent> {
    using desc   = Desc; ///< The descriptor of the layer
    using weight = typename desc::weight; ///< The data type for this layer

    static constexpr bool is_nop = false; ///< Indicate if the operation has no effect

    using input_one_t  = etl::dyn_matrix<weight, 3>; ///< The type of one input
    using output_one_t = etl::dyn_matrix<weight, 3>; ///< The type of one output
    using input_t      = std::vector<input_one_t>; ///< The type of the input
    using output_t     = std::vector<output_one_t>; ///< The type of the output

    size_t i1; ///< The first dimension of the input
    size_t i2; ///< The second dimension of the input
    size_t i3; ///< The third dimension of the input
    size_t c1; ///< The first dimension pooling ratio
    size_t c2; ///< The second dimension pooling ratio
    size_t c3; ///< The third dimension pooling ratio

    size_t o1; ///< The first dimension of the output
    size_t o2; ///< The second dimension of the output
    size_t o3; ///< The third dimension of the output

    dyn_pooling_3d_layer() = default;

    /*!
     * \brief Initialize the dynamic layer
     */
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

    /*!
     * \brief Return the size of the input of this layer
     * \return The size of the input of this layer
     */
    size_t input_size() const noexcept {
        return i1 * i2 * i3;
    }

    /*!
     * \brief Return the size of the output of this layer
     * \return The size of the output of this layer
     */
    size_t output_size() const noexcept {
        return o1 * o2 * o3;
    }

    /*!
     * \brief Return the number of trainable parameters of this network.
     * \return The the number of trainable parameters of this network.
     */
    size_t parameters() const noexcept {
        return 0;
    }

    /*!
     * \brief Prepare a set of empty outputs for this layer
     * \param samples The number of samples to prepare the output for
     * \return a container containing empty ETL matrices suitable to store samples output of this layer
     * \tparam Input The type of one input
     */
    template <typename Input>
    output_t prepare_output(size_t samples) const {
        output_t output;
        output.reserve(samples);
        for(size_t i = 0; i < samples; ++i){
            output.emplace_back(o1, o2, o3);
        }
        return output;
    }

    /*!
     * \brief Prepare one empty output for this layer
     * \return an empty ETL matrix suitable to store one output of this layer
     *
     * \tparam Input The type of one Input
     */
    template <typename Input>
    output_one_t prepare_one_output() const {
        return output_one_t(o1, o2, o3);
    }

private:
    /*!
     * \brief Returns a reference to the derived object, i.e. the object using the CRTP injector.
     * \return a reference to the derived object.
     */
    const Parent& as_derived() const {
        return *static_cast<const Parent*>(this);
    }
};

} //end of dll namespace
