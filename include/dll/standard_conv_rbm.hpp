//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_STANDARD_CONV_RBM_HPP
#define DLL_STANDARD_CONV_RBM_HPP

#include "base_conf.hpp"          //The configuration helpers
#include "rbm_base.hpp"           //The base class
#include "rbm_trainer_fwd.hpp"

namespace dll {

/*!
 * \brief Standard version of Convolutional Restricted Boltzmann Machine
 *
 * This follows the definition of a CRBM by Honglak Lee.
 */
template<typename Parent, typename Desc>
class standard_conv_rbm : public rbm_base<Parent, Desc> {
public:
    typedef float weight;

    using desc = Desc;
    using parent_t = Parent;
    using this_type = standard_conv_rbm<parent_t, desc>;
    using base_type = rbm_base<parent_t, Desc>;

    static constexpr const unit_type visible_unit = desc::visible_unit;
    static constexpr const unit_type hidden_unit = desc::hidden_unit;

    static_assert(visible_unit == unit_type::BINARY || visible_unit == unit_type::GAUSSIAN,
        "Only binary and linear visible units are supported");
    static_assert(hidden_unit == unit_type::BINARY || is_relu(hidden_unit),
        "Only binary hidden units are supported");

public:

    //Constructors

    standard_conv_rbm(){
        //Note: Convolutional RBM needs lower learning rate than standard RBM

        //Better initialization of learning rate
        base_type::learning_rate =
                visible_unit == unit_type::GAUSSIAN  ?             1e-5
            :   is_relu(hidden_unit)                 ?             1e-4
            :   /* Only Gaussian Units needs lower rate */         1e-3;
    }

    //Train functions

    template<typename Samples, bool EnableWatcher = true, typename RW = void, typename... Args>
    weight train(Samples& training_data, std::size_t max_epochs, Args... args){
        dll::rbm_trainer<parent_t, EnableWatcher, RW> trainer(args...);
        return trainer.train(*static_cast<parent_t*>(this), training_data.begin(), training_data.end(), max_epochs);
    }

    template<typename Iterator, bool EnableWatcher = true, typename RW = void, typename... Args>
    weight train(Iterator&& first, Iterator&& last, std::size_t max_epochs, Args... args){
        dll::rbm_trainer<parent_t, EnableWatcher, RW> trainer(args...);
        return trainer.train(*static_cast<parent_t*>(this), std::forward<Iterator>(first), std::forward<Iterator>(last), max_epochs);
    }

    template<typename Samples, bool EnableWatcher = true, typename RW = void, typename... Args>
    double train_denoising(Samples& noisy, Samples& clean, std::size_t max_epochs, Args... args){
        dll::rbm_trainer<parent_t, EnableWatcher, RW> trainer(args...);
        return trainer.train(*static_cast<parent_t*>(this), noisy.begin(), noisy.end(), clean.begin(), clean.end(), max_epochs);
    }

    template<typename NIterator, typename CIterator, bool EnableWatcher = true, typename RW = void, typename... Args>
    double train_denoising(NIterator&& noisy_it, NIterator&& noisy_end, CIterator clean_it, CIterator clean_end, std::size_t max_epochs, Args... args){
        dll::rbm_trainer<parent_t, EnableWatcher, RW> trainer(args...);
        return trainer.train(*static_cast<parent_t*>(this),
            std::forward<NIterator>(noisy_it), std::forward<NIterator>(noisy_end),
            std::forward<CIterator>(clean_it), std::forward<CIterator>(clean_end),
            max_epochs);
    }

    //I/O functions

    void store(std::ostream& os) const {
        store(os, *static_cast<parent_t*>(this));
    }

    void load(std::istream& is){
        load(is, *static_cast<parent_t*>(this));
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

    template<typename RBM>
    static void store(std::ostream& os, const RBM& rbm){
        binary_write_all(os, rbm.w);
        binary_write_all(os, rbm.b);
        binary_write(os, rbm.c);
    }

    template<typename RBM>
    static void load(std::istream& is, RBM& rbm){
        binary_load_all(is, rbm.w);
        binary_load_all(is, rbm.b);
        binary_load(is, rbm.c);
    }

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
