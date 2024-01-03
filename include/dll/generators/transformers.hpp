//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
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
template<typename Desc>
struct pre_scaler {
    static constexpr size_t S = Desc::ScalePre; ///< The scaling factor

    /*!
     * \brief Apply the transform on the input
     * \param target The input to transform
     */
    template<typename O>
    static void transform(O&& target){
        if constexpr (S) {
            target /= S;
        }
    }

    /*!
     * \brief Apply the transform on the input
     * \param target The input to transform
     */
    template<typename O>
    static void transform_all(O&& target){
        if constexpr (S) {
            target /= S;
        }
    }
};

/*!
 * \brief Transformer to binarize the inputs with a threshold
 */
template<typename Desc>
struct pre_binarizer {
    static constexpr size_t B = Desc::BinarizePre; ///< The binarization threshold

    /*!
     * \brief Apply the transform on the input
     * \param target The input to transform
     */
    template<typename O>
    static void transform(O&& target){
        if constexpr (B) {
            for (auto & x : target) {
                x = x > B ? 1.0 : 0.0;
            }
        }
    }

    /*!
     * \brief Apply the transform on the input
     * \param target The input to transform
     */
    template<typename O>
    static void transform_all(O&& target){
        if constexpr (B) {
            etl::binarize(target, B);
        }
    }
};

/*!
 * \brief Transformer to normalize the inputs
 */
template<typename Desc>
struct pre_normalizer {
    /*!
     * \brief Apply the transform on the input
     * \param target The input to transform
     */
    template<typename O>
    static void transform(O&& target){
        if constexpr (Desc::NormalizePre) {
            cpp::normalize(target);
        }
    }

    /*!
     * \brief Apply the transform on the input
     * \param target The input to transform
     */
    template<typename O>
    static void transform_all(O&& target){
        if constexpr (Desc::NormalizePre) {
            etl::normalize_sub(target);
        }
    }
};

} //end of dll namespace
