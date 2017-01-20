//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "dll/base_traits.hpp"
#include "dll/layer.hpp"
#include "dll/augment/augmenters.hpp"

namespace dll {

/*!
 * \brief Layer to perform data augmentation
 */
template <typename Desc>
struct augment_layer : layer<augment_layer<Desc>> {
    using desc = Desc;

    augment_layer() = default;

    static std::string to_short_string() {
        std::string name = "Augment<";

        concat_all_names(typename desc::parameters(), name);

        name += " >";

        return name;
    }

    template <typename Input, typename Output>
    static void activate_hidden(Output& h_a, const Input& input) {
        h_a.clear();

        // The original is always kept
        h_a.push_back(input);

        apply_all(typename desc::parameters(), h_a, input);
    }

    template <typename Input, typename Output>
    static void test_activate_hidden(Output& h_a, const Input& input) {
        h_a = input;
    }

    template <typename Input, typename Output>
    static void activate_many(Output& h_a, const Input& input) {
        for (std::size_t i = 0; i < input.size(); ++i) {
            activate_one(input[i], h_a[i]);
        }
    }

    template <typename Input>
    static std::vector<std::vector<Input>> prepare_output(std::size_t samples) {
        return std::vector<std::vector<Input>>(samples);
    }

    template <typename Input>
    static std::vector<Input> prepare_one_output() {
        return std::vector<Input>();
    }

    template <typename Input>
    static std::vector<Input> prepare_test_output(std::size_t samples) {
        return std::vector<Input>(samples);
    }

    template <typename Input>
    static Input prepare_one_test_output() {
        return {};
    }

    template<typename DRBM>
    static void dyn_init(DRBM&){
        //Nothing to change
    }

private:
    template <typename... Augmenter>
    static void concat_all_names(const cpp::type_list<Augmenter...>&, std::string& name) {
        int wormhole[] = {(augmenter<Augmenter>::concat_name(name), 0)...};
        cpp_unused(wormhole);
    }

    template <typename... Augmenter, typename Input, typename Output>
    static void apply_all(const cpp::type_list<Augmenter...>&, Output& h_a, const Input& input) {
        int wormhole[] = {(augmenter<Augmenter>::apply(h_a, input), 0)...};
        cpp_unused(wormhole);
    }
};

// Declare the traits for the Layer

template<typename Desc>
struct layer_base_traits<augment_layer<Desc>> {
    static constexpr bool is_neural     = false;  ///< Indicates if the layer is a neural layer
    static constexpr bool is_dense      = false;  ///< Indicates if the layer is dense
    static constexpr bool is_conv       = false; ///< Indicates if the layer is convolutional
    static constexpr bool is_deconv     = false; ///< Indicates if the layer is deconvolutional
    static constexpr bool is_standard   = true;  ///< Indicates if the layer is standard
    static constexpr bool is_rbm        = false;  ///< Indicates if the layer is RBM
    static constexpr bool is_pooling    = false; ///< Indicates if the layer is a pooling layer
    static constexpr bool is_unpooling  = false; ///< Indicates if the layer is an unpooling laye
    static constexpr bool is_transform  = false; ///< Indicates if the layer is a transform layer
    static constexpr bool is_patches    = false; ///< Indicates if the layer is a patches layer
    static constexpr bool is_augment    = true; ///< Indicates if the layer is an augment layer
    static constexpr bool is_dynamic    = false; ///< Indicates if the layer is dynamic
    static constexpr bool pretrain_last = false; ///< Indicates if the layer is dynamic
    static constexpr bool sgd_supported = true;  ///< Indicates if the layer is supported by SGD
};

} //end of dll namespace
