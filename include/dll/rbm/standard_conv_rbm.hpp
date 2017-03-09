//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "dll/base_conf.hpp"    //The configuration helpers
#include "dll/rbm/rbm_base.hpp"     //The base class
#include "dll/layer_traits.hpp" //layer_traits
#include "dll/util/checks.hpp"  //nan_check
#include "dll/util/timers.hpp"  //auto_timer

namespace dll {

/*!
 * \brief Standard version of Convolutional Restricted Boltzmann Machine
 *
 * This follows the definition of a CRBM by Honglak Lee. This is an "abstract" class,
 * using CRTP to inject features into its children.
 */
template <typename Parent, typename Desc>
struct standard_conv_rbm : public rbm_base<Parent, Desc> {
    using desc      = Desc;
    using parent_t  = Parent;
    using this_type = standard_conv_rbm<parent_t, desc>;
    using base_type = rbm_base<parent_t, Desc>;
    using weight    = typename desc::weight;

    using input_one_t         = typename rbm_base_traits<parent_t>::input_one_t;
    using output_one_t        = typename rbm_base_traits<parent_t>::output_one_t;
    using hidden_output_one_t = typename rbm_base_traits<parent_t>::hidden_output_one_t;
    using input_t             = typename rbm_base_traits<parent_t>::input_t;
    using output_t            = typename rbm_base_traits<parent_t>::output_t;

    static constexpr const unit_type visible_unit = desc::visible_unit;
    static constexpr const unit_type hidden_unit  = desc::hidden_unit;

    static_assert(visible_unit == unit_type::BINARY || visible_unit == unit_type::GAUSSIAN,
                  "Only binary and linear visible units are supported");
    static_assert(hidden_unit == unit_type::BINARY || is_relu(hidden_unit),
                  "Only binary hidden units are supported");

    double std_gaussian = 0.2;
    double c_sigm       = 1.0;

    //Constructors

    standard_conv_rbm() {
        //Note: Convolutional RBM needs lower learning rate than standard RBM

        //Better initialization of learning rate
        base_type::learning_rate =
            visible_unit == unit_type::GAUSSIAN ? 1e-5
                                                : is_relu(hidden_unit) ? 1e-4
                                                                       : /* Only Gaussian Units needs lower rate */ 1e-3;
    }

    //Utility functions

    template <typename Sample>
    void reconstruct(const Sample& items) {
        reconstruct(items, as_derived());
    }

    void display_visible_unit_activations() const {
        display_visible_unit_activations(as_derived());
    }

    void display_visible_unit_samples() const {
        display_visible_unit_samples(as_derived());
    }

    void display_hidden_unit_activations() const {
        display_hidden_unit_samples(as_derived());
    }

    void display_hidden_unit_samples() const {
        display_hidden_unit_samples(as_derived());
    }

    //Various functions

    void activate_many(const input_t& input, output_t& h_a, output_t& h_s) const {
        for (std::size_t i = 0; i < input.size(); ++i) {
            as_derived().activate_one(input[i], h_a[i], h_s[i]);
        }
    }

    void activate_many(const input_t& input, output_t& h_a) const {
        for (std::size_t i = 0; i < input.size(); ++i) {
            as_derived().activate_one(input[i], h_a[i]);
        }
    }

    template <typename V, typename H>
    void batch_activate_hidden(H& h_a, const V& input) const {
        as_derived().template batch_activate_hidden<true, false>(h_a, h_a, input, input);
    }

    template<typename Out>
    weight energy(const input_one_t& v, const Out& h) const {
        return as_derived().energy_impl(v, h);
    }

    template<typename Input, typename Out>
    weight energy(const Input& v, const Out& h) const {
        decltype(auto) converted = converter_one<Input, input_one_t>::convert(as_derived(), v);
        return as_derived().energy_impl(converted, h);
    }

    weight free_energy(const input_one_t& v) const {
        return as_derived().free_energy_impl(v);
    }

    template <typename V>
    weight free_energy(const V& v) const {
        decltype(auto) converted = converter_one<V, input_one_t>::convert(as_derived(), v);
        return as_derived().free_energy_impl(converted);
    }

    weight free_energy() const {
        return free_energy(as_derived().v1);
    }

    friend base_type;

private:
    //Since the sub classes do not have the same fields, it is not possible
    //to put the fields in standard_rbm, therefore, it is necessary to use template
    //functions to implement the details

    static double reconstruction_error_impl(const input_one_t& items, parent_t& rbm) {
        cpp_assert(items.size() == input_size(rbm), "The size of the training sample must match visible units");

        //Set the state of the visible units
        rbm.v1 = items;

        rbm.activate_hidden(rbm.h1_a, rbm.h1_s, rbm.v1, rbm.v1);
        rbm.activate_visible(rbm.h1_a, rbm.h1_s, rbm.v2_a, rbm.v2_s);

        return etl::mean((rbm.v1 - rbm.v2_a) >> (rbm.v1 - rbm.v2_a));
    }

    template <typename Sample>
    static void reconstruct(const Sample& items, parent_t& rbm) {
        cpp_assert(items.size() == parent_t::input_size(), "The size of the training sample must match visible units");

        cpp::stop_watch<> watch;

        //Set the state of the visible units
        rbm.v1 = items;

        rbm.activate_hidden(rbm.h1_a, rbm.h1_s, rbm.v1, rbm.v1);

        rbm.activate_visible(rbm.h1_a, rbm.h1_s, rbm.v2_a, rbm.v2_s);
        rbm.activate_hidden(rbm.h2_a, rbm.h2_s, rbm.v2_a, rbm.v2_s);

        std::cout << "Reconstruction took " << watch.elapsed() << "ms" << std::endl;
    }

    static void display_visible_unit_activations(const parent_t& rbm) {
        for (std::size_t channel = 0; channel < parent_t::NC; ++channel) {
            std::cout << "Channel " << channel << std::endl;

            for (size_t i = 0; i < get_nv1(rbm); ++i) {
                for (size_t j = 0; j < get_nv2(rbm); ++j) {
                    std::cout << rbm.v2_a(channel, i, j) << " ";
                }
                std::cout << std::endl;
            }
        }
    }

    static void display_visible_unit_samples(const parent_t& rbm) {
        for (std::size_t channel = 0; channel < parent_t::NC; ++channel) {
            std::cout << "Channel " << channel << std::endl;

            for (size_t i = 0; i < get_nv1(rbm); ++i) {
                for (size_t j = 0; j < get_nv2(rbm); ++j) {
                    std::cout << rbm.v2_s(channel, i, j) << " ";
                }
                std::cout << std::endl;
            }
        }
    }

    static void display_hidden_unit_activations(const parent_t& rbm) {
        for (size_t k = 0; k < get_k(rbm); ++k) {
            for (size_t i = 0; i < get_nv1(rbm); ++i) {
                for (size_t j = 0; j < get_nv2(rbm); ++j) {
                    std::cout << rbm.h2_a(k)(i, j) << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl
                      << std::endl;
        }
    }

    static void display_hidden_unit_samples(const parent_t& rbm) {
        for (size_t k = 0; k < get_k(rbm); ++k) {
            for (size_t i = 0; i < get_nv1(rbm); ++i) {
                for (size_t j = 0; j < get_nv2(rbm); ++j) {
                    std::cout << rbm.h2_s(k)(i, j) << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl
                      << std::endl;
        }
    }

    parent_t& as_derived() {
        return *static_cast<parent_t*>(this);
    }

    const parent_t& as_derived() const {
        return *static_cast<const parent_t*>(this);
    }
};

} //end of dll namespace
