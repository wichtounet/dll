//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_STANDARD_CONV_RBM_HPP
#define DLL_STANDARD_CONV_RBM_HPP

#include "base_conf.hpp"          //The configuration helpers
#include "rbm_base.hpp"           //The base class

namespace dll {

/*!
 * \brief Standard version of Convolutional Restricted Boltzmann Machine
 *
 * This follows the definition of a CRBM by Honglak Lee.
 */
template<typename Parent, typename Desc>
struct standard_conv_rbm : public rbm_base<Parent, Desc> {
    using desc = Desc;
    using parent_t = Parent;
    using this_type = standard_conv_rbm<parent_t, desc>;
    using base_type = rbm_base<parent_t, Desc>;
    using weight = typename desc::weight;

    static constexpr const unit_type visible_unit = desc::visible_unit;
    static constexpr const unit_type hidden_unit = desc::hidden_unit;

    static_assert(visible_unit == unit_type::BINARY || visible_unit == unit_type::GAUSSIAN,
        "Only binary and linear visible units are supported");
    static_assert(hidden_unit == unit_type::BINARY || is_relu(hidden_unit),
        "Only binary hidden units are supported");

    //Constructors

    standard_conv_rbm(){
        //Note: Convolutional RBM needs lower learning rate than standard RBM

        //Better initialization of learning rate
        base_type::learning_rate =
                visible_unit == unit_type::GAUSSIAN  ?             1e-5
            :   is_relu(hidden_unit)                 ?             1e-4
            :   /* Only Gaussian Units needs lower rate */         1e-3;
    }

    //Utility functions

    template<typename Sample>
    void reconstruct(const Sample& items){
        reconstruct(items, *static_cast<parent_t*>(this));
    }

    void display_visible_unit_activations() const {
        display_visible_unit_activations(*static_cast<parent_t*>(this));
    }

    void display_visible_unit_samples() const {
        display_visible_unit_samples(*static_cast<parent_t*>(this));
    }

    void display_hidden_unit_activations() const {
        display_hidden_unit_samples(*static_cast<parent_t*>(this));
    }

    void display_hidden_unit_samples() const {
        display_hidden_unit_samples(*static_cast<parent_t*>(this));
    }

private:

    //Since the sub classes do not have the same fields, it is not possible
    //to put the fields in standard_rbm, therefore, it is necessary to use template
    //functions to implement the details

    template<typename Sample, typename RBM>
    static void reconstruct(const Sample& items, RBM& rbm){
        cpp_assert(items.size() == RBM::input_size(), "The size of the training sample must match visible units");

        cpp::stop_watch<> watch;

        //Set the state of the visible units
        rbm.v1 = items;

        rbm.activate_hidden(rbm.h1_a, rbm.h1_s, rbm.v1, rbm.v1);

        rbm.activate_visible(rbm.h1_a, rbm.h1_s, rbm.v2_a, rbm.v2_s);
        rbm.activate_hidden(rbm.h2_a, rbm.h2_s, rbm.v2_a, rbm.v2_s);

        std::cout << "Reconstruction took " << watch.elapsed() << "ms" << std::endl;
    }

    template<typename RBM>
    static void display_visible_unit_activations(const RBM& rbm){
        for(std::size_t channel = 0; channel < RBM::NC; ++channel){
            std::cout << "Channel " << channel << std::endl;

            for(size_t i = 0; i < RBM::NV; ++i){
                for(size_t j = 0; j < RBM::NV; ++j){
                    std::cout << rbm.v2_a(channel, i, j) << " ";
                }
                std::cout << std::endl;
            }
        }
    }

    template<typename RBM>
    static void display_visible_unit_samples(const RBM& rbm){
        for(std::size_t channel = 0; channel < RBM::NC; ++channel){
            std::cout << "Channel " << channel << std::endl;

            for(size_t i = 0; i < RBM::NV; ++i){
                for(size_t j = 0; j < RBM::NV; ++j){
                    std::cout << rbm.v2_s(channel, i, j) << " ";
                }
                std::cout << std::endl;
            }
        }
    }

    template<typename RBM>
    static void display_hidden_unit_activations(const RBM& rbm){
        for(size_t k = 0; k < RBM::K; ++k){
            for(size_t i = 0; i < RBM::NV; ++i){
                for(size_t j = 0; j < RBM::NV; ++j){
                    std::cout << rbm.h2_a(k)(i, j) << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl << std::endl;
        }
    }

    template<typename RBM>
    static void display_hidden_unit_samples(const RBM& rbm){
        for(size_t k = 0; k < RBM::K; ++k){
            for(size_t i = 0; i < RBM::NV; ++i){
                for(size_t j = 0; j < RBM::NV; ++j){
                    std::cout << rbm.h2_s(k)(i, j) << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl << std::endl;
        }
    }
};

} //end of dll namespace

#endif
