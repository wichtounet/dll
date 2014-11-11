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
class standard_conv_rbm : public rbm_base<Desc> {
public:
    typedef float weight;

    using desc = Desc;
    using parent_t = Parent;

    static constexpr const unit_type visible_unit = desc::visible_unit;
    static constexpr const unit_type hidden_unit = desc::hidden_unit;

    static_assert(visible_unit == unit_type::BINARY || visible_unit == unit_type::GAUSSIAN,
        "Only binary and linear visible units are supported");
    static_assert(hidden_unit == unit_type::BINARY || is_relu(hidden_unit),
        "Only binary hidden units are supported");

public:

    standard_conv_rbm(){
        //Note: Convolutional RBM needs lower learning rate than standard RBM

        //Better initialization of learning rate
        rbm_base<desc>::learning_rate =
                visible_unit == unit_type::GAUSSIAN  ?             1e-5
            :   is_relu(hidden_unit)                 ?             1e-4
            :   /* Only Gaussian Units needs lower rate */         1e-3;
    }

    void store(std::ostream& os) const {
        store(os, *static_cast<parent_t*>(this));
    }

    void load(std::istream& is){
        load(is, *static_cast<parent_t*>(this));
    }

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

private:

    template<typename RBM>
    static void store(std::ostream& os, const RBM& rbm){
        binary_write_all(os, rbm.w);
        binary_write_all(os, rbm.b);
        binary_write(os, rbm.c);
    }

    template<typename RBM>
    void load(std::istream& is, RBM& rbm){
        binary_load_all(is, rbm.w);
        binary_load_all(is, rbm.b);
        binary_load(is, rbm.c);
    }

};

} //end of dll namespace

#endif
