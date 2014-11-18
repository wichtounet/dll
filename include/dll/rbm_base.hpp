//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_RBM_BASE_HPP
#define DLL_RBM_BASE_HPP

#include <iostream>
#include <fstream>

#include "io.hpp"
#include "rbm_trainer_fwd.hpp"

namespace dll {

/*!
 * \brief Base class for Restricted Boltzmann Machine.
 *
 * It only contains configurable properties that are used by each
 * version of RBM.
 */
template<typename Parent, typename Layer>
class rbm_base {
public:
    typedef float weight;

    using conf = Layer;
    using parent_t = Parent;

    //Configurable properties
    weight learning_rate = 1e-1;        ///< The learning rate

    weight initial_momentum = 0.5;      ///< The initial momentum
    weight final_momentum = 0.9;        ///< The final momentum applied after *final_momentum_epoch* epoch
    weight final_momentum_epoch = 6;    ///< The epoch at which momentum change

    weight momentum = 0;                ///< The current momentum

    weight l1_weight_cost = 0.0002;     ///< The weight cost for L1 weight decay
    weight l2_weight_cost = 0.0002;     ///< The weight cost for L2 weight decay

    weight sparsity_target = 0.01;      ///< The sparsity target
    weight decay_rate = 0.99;           ///< The sparsity decay rate
    weight sparsity_cost = 1.0;         ///< The sparsity cost (or sparsity multiplier)

    weight pbias = 0.002;
    weight pbias_lambda = 5;

    //No copying
    rbm_base(const rbm_base& rbm) = delete;
    rbm_base& operator=(const rbm_base& rbm) = delete;

    //No moving
    rbm_base(rbm_base&& rbm) = delete;
    rbm_base& operator=(rbm_base&& rbm) = delete;

    rbm_base(){
        //Nothing to do
    }

    //Normal Train functions

    template<typename Samples, bool EnableWatcher = true, typename RW = void, typename... Args>
    double train(Samples& training_data, std::size_t max_epochs, Args... args){
        dll::rbm_trainer<parent_t, EnableWatcher, RW> trainer(args...);
        return trainer.train(*static_cast<parent_t*>(this), training_data.begin(), training_data.end(), max_epochs);
    }

    template<typename Iterator, bool EnableWatcher = true, typename RW = void, typename... Args>
    double train(Iterator&& first, Iterator&& last, std::size_t max_epochs, Args... args){
        dll::rbm_trainer<parent_t, EnableWatcher, RW> trainer(args...);
        return trainer.train(*static_cast<parent_t*>(this), std::forward<Iterator>(first), std::forward<Iterator>(last), max_epochs);
    }

    //Train denoising autoencoder

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

    void store(const std::string& file) const {
        store(file, *static_cast<const parent_t*>(this));
    }

    void store(std::ostream& os) const {
        store(os, *static_cast<const parent_t*>(this));
    }

    void load(const std::string& file){
        load(file, *static_cast<parent_t*>(this));
    }

    void load(std::istream& is){
        load(is, *static_cast<parent_t*>(this));
    }

private:

    //Since the sub classes does not have the same fields, it is not possible
    //to put the fields in standard_rbm, therefore, it is necessary to use template
    //functions to implement the details

    template<typename RBM>
    static void store(std::ostream& os, const RBM& rbm){
        binary_write_all(os, rbm.w);
        binary_write_all(os, rbm.b);
        binary_write_all(os, rbm.c);
    }

    template<typename RBM>
    static void load(std::istream& is, RBM& rbm){
        binary_load_all(is, rbm.w);
        binary_load_all(is, rbm.b);
        binary_load_all(is, rbm.c);
    }

    template<typename RBM>
    static void store(const std::string& file, const RBM& rbm){
        std::ofstream os(file, std::ofstream::binary);
        store(os, rbm);
    }

    template<typename RBM>
    static void load(const std::string& file, RBM& rbm){
        std::ifstream is(file, std::ifstream::binary);
        load(is, rbm);
    }
};

} //end of dll namespace

#endif
