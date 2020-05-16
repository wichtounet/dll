//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
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
    using desc      = Desc;                              ///< The descriptor of the layer
    using parent_t  = Parent;                            ///< The parent type
    using this_type = standard_conv_rbm<parent_t, desc>; ///< The type of this layer
    using base_type = rbm_base<parent_t, Desc>;          ///< The base type
    using weight    = typename desc::weight;             ///< The data type for this layer
    using layer_t     = this_type;                     ///< This layer's type
    using dyn_layer_t = typename desc::dyn_layer_t;    ///< The dynamic version of this layer

    using input_one_t         = typename rbm_base_traits<parent_t>::input_one_t;         ///< The type of one input
    using output_one_t        = typename rbm_base_traits<parent_t>::output_one_t;        ///< The type of one output
    using hidden_output_one_t = typename rbm_base_traits<parent_t>::hidden_output_one_t; ///< The type of an hidden output
    using input_t             = typename rbm_base_traits<parent_t>::input_t;             ///< The type of the input
    using output_t            = typename rbm_base_traits<parent_t>::output_t;            ///< The type of the output

    static constexpr unit_type visible_unit = desc::visible_unit; ///< The visible unit type
    static constexpr unit_type hidden_unit  = desc::hidden_unit;  ///< The hidden unit type

    static_assert(visible_unit == unit_type::BINARY || visible_unit == unit_type::GAUSSIAN,
                  "Only binary and linear visible units are supported");
    static_assert(hidden_unit == unit_type::BINARY || is_relu(hidden_unit),
                  "Only binary hidden units are supported");

    //Constructors

    /*!
     * \brief Construct a new standard convolutional RBM
     */
    standard_conv_rbm() {
        //Note: Convolutional RBM needs lower learning rate than standard RBM

        //Better initialization of learning rate
        base_type::learning_rate =
            visible_unit == unit_type::GAUSSIAN ? 1e-5
                                                : is_relu(hidden_unit) ? 1e-4
                                                                       : /* Only Gaussian Units needs lower rate */ 1e-3;
    }

    //Utility functions

    /*!
     * \brief Reconstruct the given input
     */
    template <typename Sample>
    void reconstruct(const Sample& items) {
        reconstruct(items, as_derived());
    }

    /*!
     * \brief Display the current visible unit activations
     */
    void display_visible_unit_activations() const {
        display_visible_unit_activations(as_derived());
    }

    /*!
     * \brief Display the current visible unit samples
     */
    void display_visible_unit_samples() const {
        display_visible_unit_samples(as_derived());
    }

    /*!
     * \brief Display the current hidden unit activations
     */
    void display_hidden_unit_activations() const {
        display_hidden_unit_samples(as_derived());
    }

    /*!
     * \brief Display the current hidden unit samples
     */
    void display_hidden_unit_samples() const {
        display_hidden_unit_samples(as_derived());
    }

    //Various functions

    /*!
     * \brief Batch activation of inputs
     * \param h_a The output activations
     * \param input The batch of input
     */
    template <typename V, typename H>
    void batch_activate_hidden(H& h_a, const V& input) const {
        decltype(auto) rbm = as_derived();

        if constexpr (etl::dimensions<V>() == 4) {
            rbm.template batch_activate_hidden<true, false>(h_a, h_a, input, input);
        } else {
            rbm.template batch_activate_hidden<true, false>(h_a, h_a,
                                                            etl::reshape(input, etl::dim<0>(input), get_nc(rbm), get_nv1(rbm), get_nv2(rbm)),
                                                            etl::reshape(input, etl::dim<0>(input), get_nc(rbm), get_nv1(rbm), get_nv2(rbm)));
        }
    }

    /*!
     * \brief Return the energy of the given joint configuration
     * \param v The configuration of the inputs
     * \param h The configuration of the outputs
     * \return The scalar energy of the model for the given joint configuration
     */
    template<typename Input, typename Out>
    weight energy(const Input& v, const Out& h) const {
        return as_derived().energy_impl(v, h);
    }

    /*!
     * \brief Return the free energy of the given input
     */
    template <typename V>
    weight free_energy(const V& v) const {
        return as_derived().free_energy_impl(v);
    }

    /*!
     * \brief Return the free energy of the current input
     */
    weight free_energy() const {
        return free_energy(as_derived().v1);
    }

    friend base_type;

private:
    //Since the sub classes do not have the same fields, it is not possible
    //to put the fields in standard_rbm, therefore, it is necessary to use template
    //functions to implement the details

    /*!
     * \brief Compute the reconstruction for the given input and RBM
     */
    template<typename Input>
    static double reconstruction_error_impl(const Input& items, parent_t& rbm) {
        cpp_assert(items.size() == input_size(rbm), "The size of the training sample must match visible units");

        //Set the state of the visible units
        rbm.v1 = items;

        rbm.activate_hidden(rbm.h1_a, rbm.h1_s, rbm.v1, rbm.v1);
        rbm.activate_visible(rbm.h1_a, rbm.h1_s, rbm.v2_a, rbm.v2_s);

        return etl::mean((rbm.v1 - rbm.v2_a) >> (rbm.v1 - rbm.v2_a));
    }

    /*!
     * \brief Reconstruct the given input
     */
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

    /*!
     * \brief Display the current visible unit activations
     */
    static void display_visible_unit_activations(const parent_t& rbm) {
        for (size_t channel = 0; channel < parent_t::NC; ++channel) {
            std::cout << "Channel " << channel << std::endl;

            for (size_t i = 0; i < get_nv1(rbm); ++i) {
                for (size_t j = 0; j < get_nv2(rbm); ++j) {
                    std::cout << rbm.v2_a(channel, i, j) << " ";
                }
                std::cout << std::endl;
            }
        }
    }

    /*!
     * \brief Display the current visible unit samples
     */
    static void display_visible_unit_samples(const parent_t& rbm) {
        for (size_t channel = 0; channel < parent_t::NC; ++channel) {
            std::cout << "Channel " << channel << std::endl;

            for (size_t i = 0; i < get_nv1(rbm); ++i) {
                for (size_t j = 0; j < get_nv2(rbm); ++j) {
                    std::cout << rbm.v2_s(channel, i, j) << " ";
                }
                std::cout << std::endl;
            }
        }
    }

    /*!
     * \brief Display the current hidden unit activations
     */
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

    /*!
     * \brief Display the current hidden unit samples
     */
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

    /*!
     * \brief Returns a reference to the derived object, i.e. the object using the CRTP injector.
     * \return a reference to the derived object.
     */
    parent_t& as_derived() {
        return *static_cast<parent_t*>(this);
    }

    /*!
     * \brief Returns a reference to the derived object, i.e. the object using the CRTP injector.
     * \return a reference to the derived object.
     */
    const parent_t& as_derived() const {
        return *static_cast<const parent_t*>(this);
    }
};

} //end of dll namespace
