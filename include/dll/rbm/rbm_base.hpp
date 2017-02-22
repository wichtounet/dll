//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <iosfwd>
#include <fstream>

#include "cpp_utils/io.hpp"

#include "dll/layer.hpp"
#include "dll/trainer/rbm_trainer_fwd.hpp"
#include "dll/util/converter.hpp" //converter

namespace dll {

/*!
 * \brief Simple traits to pass information around from the real
 * class to the CRTP class.
 */
template <typename Parent>
struct rbm_base_traits;

/*!
 * \brief Base class for Restricted Boltzmann Machine.
 *
 * It contains configurable properties that are used by each
 * version of RBM. It also contains common functions that are
 * injected using CRTP technique.
 */
template <typename Parent, typename Desc>
struct rbm_base : layer<Parent> {
    using conf     = Desc;
    using parent_t = Parent;
    using weight   = typename conf::weight;

    using input_one_t  = typename rbm_base_traits<parent_t>::input_one_t;
    using output_one_t = typename rbm_base_traits<parent_t>::output_one_t;
    using input_t      = typename rbm_base_traits<parent_t>::input_t;
    using output_t     = typename rbm_base_traits<parent_t>::output_t;

    //Configurable properties
    weight learning_rate = 1e-1; ///< The learning rate

    weight initial_momentum     = 0.5; ///< The initial momentum
    weight final_momentum       = 0.9; ///< The final momentum applied after *final_momentum_epoch* epoch
    weight final_momentum_epoch = 6;   ///< The epoch at which momentum change

    weight momentum = 0; ///< The current momentum

    weight l1_weight_cost = 0.0002; ///< The weight cost for L1 weight decay
    weight l2_weight_cost = 0.0002; ///< The weight cost for L2 weight decay

    weight sparsity_target = 0.01; ///< The sparsity target
    weight decay_rate      = 0.99; ///< The sparsity decay rate
    weight sparsity_cost   = 1.0;  ///< The sparsity cost (or sparsity multiplier)

    weight pbias        = 0.002;
    weight pbias_lambda = 5;

    weight gradient_clip = 5.0; ///< The default gradient clipping value

    rbm_base() {
        //Nothing to do
    }

    void backup_weights() {
        unique_safe_get(as_derived().bak_w) = as_derived().w;
        unique_safe_get(as_derived().bak_b) = as_derived().b;
        unique_safe_get(as_derived().bak_c) = as_derived().c;
    }

    void restore_weights() {
        as_derived().w = *as_derived().bak_w;
        as_derived().b = *as_derived().bak_b;
        as_derived().c = *as_derived().bak_c;
    }

    double reconstruction_error(const input_one_t& item) {
        return parent_t::reconstruction_error_impl(item, as_derived());
    }

    template<typename Input>
    double reconstruction_error(const Input& item) {
        decltype(auto) converted_item = converter_one<Input, input_one_t>::convert(as_derived(), item);
        return parent_t::reconstruction_error_impl(converted_item, as_derived());
    }

    //Normal Train functions

    template <bool EnableWatcher = true, typename RW = void, typename... Args>
    double train(const input_t& training_data, std::size_t max_epochs, Args... args) {
        dll::rbm_trainer<parent_t, EnableWatcher, RW, false> trainer(args...);
        return trainer.train(as_derived(), training_data.begin(), training_data.end(), max_epochs);
    }

    template <bool EnableWatcher = true, typename RW = void, typename Input, typename... Args>
    double train(const Input& training_data, std::size_t max_epochs, Args... args) {
        decltype(auto) converted_samples = converter_many<Input, input_t>::convert(as_derived(), training_data);
        dll::rbm_trainer<parent_t, EnableWatcher, RW, false> trainer(args...);
        return trainer.train(as_derived(), converted_samples.begin(), converted_samples.end(), max_epochs);
    }

    template <bool EnableWatcher = true, typename RW = void, typename Iterator, typename... Args>
    double train(Iterator&& first, Iterator&& last, std::size_t max_epochs, Args... args) {
        dll::rbm_trainer<parent_t, EnableWatcher, RW, false> trainer(args...);
        return trainer.train(as_derived(), std::forward<Iterator>(first), std::forward<Iterator>(last), max_epochs);
    }

    //Train denoising autoencoder

    template <bool EnableWatcher = true, typename RW = void, typename... Args>
    double train_denoising(const input_t& noisy, const input_t& clean, std::size_t max_epochs, Args... args) {
        dll::rbm_trainer<parent_t, EnableWatcher, RW, true> trainer(args...);
        return trainer.train(as_derived(), noisy.begin(), noisy.end(), clean.begin(), clean.end(), max_epochs);
    }

    template <bool EnableWatcher = true, typename RW = void, typename Noisy, typename Clean, typename... Args>
    double train_denoising(const Noisy& noisy, const Clean& clean, std::size_t max_epochs, Args... args) {
        decltype(auto) converted_noisy = converter_many<Noisy, input_t>::convert(as_derived(), noisy);
        decltype(auto) converted_clean = converter_many<Clean, input_t>::convert(as_derived(), clean);
        dll::rbm_trainer<parent_t, EnableWatcher, RW, true> trainer(args...);
        return trainer.train(as_derived(),
                             converted_noisy.begin(), converted_noisy.end(),
                             converted_clean.begin(), converted_clean.end(),
                             max_epochs);
    }

    template <typename NIterator, typename CIterator, bool EnableWatcher = true, typename RW = void, typename... Args>
    double train_denoising(NIterator noisy_it, NIterator noisy_end, CIterator clean_it, CIterator clean_end, std::size_t max_epochs, Args... args) {
        dll::rbm_trainer<parent_t, EnableWatcher, RW, true> trainer(args...);
        return trainer.train(as_derived(),
                             noisy_it, noisy_end,
                             clean_it, clean_end,
                             max_epochs);
    }

    //Train denoising autoencoder (auto)

    template <bool EnableWatcher = true, typename RW = void, typename... Args>
    double train_denoising_auto(const input_t& clean, std::size_t max_epochs, double noise, Args... args) {
        dll::rbm_trainer<parent_t, EnableWatcher, RW, false> trainer(args...);
        return trainer.train_denoising_auto(as_derived(), clean.begin(), clean.end(), max_epochs, noise);
    }

    template <bool EnableWatcher = true, typename RW = void, typename Clean, typename... Args>
    double train_denoising_auto(const Clean& clean, std::size_t max_epochs, double noise, Args... args) {
        decltype(auto) converted_clean = converter_many<Clean, input_t>::convert(as_derived(), clean);
        dll::rbm_trainer<parent_t, EnableWatcher, RW, false> trainer(args...);
        return trainer.train_denoising_auto(as_derived(), converted_clean.begin(), converted_clean.end(), max_epochs, noise);
    }

    template <typename CIterator, bool EnableWatcher = true, typename RW = void, typename... Args>
    double train_denoising_auto(CIterator clean_it, CIterator clean_end, std::size_t max_epochs, double noise, Args... args) {
        dll::rbm_trainer<parent_t, EnableWatcher, RW, false> trainer(args...);
        return trainer.train_denoising_auto(as_derived(), clean_it, clean_end, max_epochs, noise);
    }

    // Features

    output_one_t features(const input_one_t& input){
        return activate_hidden(input);;
    }

    template<typename Input>
    output_one_t features(const Input& input){
        decltype(auto) converted = converter_one<Input, input_one_t>::convert(as_derived(), input);
        return activate_hidden(converted);
    }

    output_one_t activate_hidden(const input_one_t& input){
        auto output = as_derived().template prepare_one_output<input_one_t>();
        as_derived().activate_hidden(output, input);
        return output;
    }

    template<typename Input>
    output_one_t activate_hidden(const Input& input){
        decltype(auto) converted = converter_one<Input, input_one_t>::convert(as_derived(), input);
        auto output = as_derived().template prepare_one_output<input_one_t>();
        as_derived().activate_hidden(output, converted);
        return output;
    }

    //I/O functions

    void store(const std::string& file) const {
        store(file, as_derived());
    }

    void store(std::ostream& os) const {
        store(os, as_derived());
    }

    void load(const std::string& file) {
        load(file, as_derived());
    }

    void load(std::istream& is) {
        load(is, as_derived());
    }

private:
    static void store(std::ostream& os, const parent_t& rbm) {
        cpp::binary_write_all(os, rbm.w);
        cpp::binary_write_all(os, rbm.b);
        cpp::binary_write_all(os, rbm.c);
    }

    static void load(std::istream& is, parent_t& rbm) {
        cpp::binary_load_all(is, rbm.w);
        cpp::binary_load_all(is, rbm.b);
        cpp::binary_load_all(is, rbm.c);
    }

    static void store(const std::string& file, const parent_t& rbm) {
        std::ofstream os(file, std::ofstream::binary);
        store(os, rbm);
    }

    static void load(const std::string& file, parent_t& rbm) {
        std::ifstream is(file, std::ifstream::binary);
        load(is, rbm);
    }

    parent_t& as_derived() {
        return *static_cast<parent_t*>(this);
    }

    const parent_t& as_derived() const {
        return *static_cast<const parent_t*>(this);
    }
};

} //end of dll namespace
