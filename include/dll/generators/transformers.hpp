//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <atomic>
#include <thread>

#include "cpp_utils/data.hpp"

namespace dll {

/*!
 * \brief Transformer to scale the inputs with a scaler
 */
template<typename Desc, typename Enable = void>
struct pre_scaler;

/*!
 * \copydoc pre_scaler
 */
template<typename Desc>
struct pre_scaler <Desc, std::enable_if_t<Desc::ScalePre != 0>> {
    static constexpr size_t S = Desc::ScalePre; ///< The scaling factor

    /*!
     * \brief Apply the transform on the input
     * \param target The input to transform
     */
    template<typename O>
    static void transform(O&& target){
        target /= S;
    }

    /*!
     * \brief Apply the transform on the input
     * \param target The input to transform
     */
    template<typename O>
    static void transform_all(O&& target){
        target /= S;
    }
};

/*!
 * \copydoc pre_scaler
 */
template<typename Desc>
struct pre_scaler <Desc, std::enable_if_t<Desc::ScalePre == 0>> {
    /*!
     * \brief Apply the transform on the input
     * \param target The input to transform
     */
    template <typename O>
    static void transform([[maybe_unused]] O&& target) {}

    /*!
     * \brief Apply the transform on the input
     * \param target The input to transform
     */
    template <typename O>
    static void transform_all([[maybe_unused]] O&& target) {}
};

/*!
 * \brief Transformer to binarize the inputs with a threshold
 */
template<typename Desc, typename Enable = void>
struct pre_binarizer;

/*!
 * \copydoc pre_binarizer
 */
template<typename Desc>
struct pre_binarizer <Desc, std::enable_if_t<Desc::BinarizePre != 0>> {
    static constexpr size_t B = Desc::BinarizePre; ///< The binarization threshold

    /*!
     * \brief Apply the transform on the input
     * \param target The input to transform
     */
    template<typename O>
    static void transform(O&& target){
        for(auto& x : target){
            x = x > B ? 1.0 : 0.0;
        }
    }

    /*!
     * \brief Apply the transform on the input
     * \param target The input to transform
     */
    template<typename O>
    static void transform_all(O&& target){
        etl::binarize(target, B);
    }
};

/*!
 * \copydoc pre_binarizer
 */
template<typename Desc>
struct pre_binarizer <Desc, std::enable_if_t<Desc::BinarizePre == 0>> {
    /*!
     * \brief Apply the transform on the input
     * \param target The input to transform
     */
    template <typename O>
    static void transform([[maybe_unused]] O&& target) {}

    /*!
     * \brief Apply the transform on the input
     * \param target The input to transform
     */
    template <typename O>
    static void transform_all([[maybe_unused]] O&& target) {}
};

/*!
 * \brief Transformer to normalize the inputs
 */
template<typename Desc, typename Enable = void>
struct pre_normalizer;

/*!
 * \copydoc pre_normalizer
 */
template<typename Desc>
struct pre_normalizer <Desc, std::enable_if_t<Desc::NormalizePre>> {
    /*!
     * \brief Apply the transform on the input
     * \param target The input to transform
     */
    template<typename O>
    static void transform(O&& target){
        cpp::normalize(target);
    }

    /*!
     * \brief Apply the transform on the input
     * \param target The input to transform
     */
    template<typename O>
    static void transform_all(O&& target){
        etl::normalize_sub(target);
    }
};

/*!
 * \copydoc pre_normalizer
 */
template<typename Desc>
struct pre_normalizer <Desc, std::enable_if_t<!Desc::NormalizePre>> {
    /*!
     * \brief Apply the transform on the input
     * \param target The input to transform
     */
    template <typename O>
    static void transform([[maybe_unused]] O&& target) {}

    /*!
     * \brief Apply the transform on the input
     * \param target The input to transform
     */
    template <typename O>
    static void transform_all([[maybe_unused]] O&& target) {}
};

} //end of dll namespace
