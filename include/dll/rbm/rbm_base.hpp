//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <iosfwd>
#include <fstream>

#include "cpp_utils/io.hpp"

#include "dll/generators.hpp"
#include "dll/layer.hpp"
#include "dll/trainer/rbm_trainer_fwd.hpp"

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
    using conf     = Desc;                  ///< The description of the layer
    using parent_t = Parent;                ///< The parent layer
    using weight   = typename conf::weight; ///< The data type for this layer

    using input_one_t  = typename rbm_base_traits<parent_t>::input_one_t; ///< The type of one input
    using output_one_t = typename rbm_base_traits<parent_t>::output_one_t; ///< The type of one output
    using input_t      = typename rbm_base_traits<parent_t>::input_t; ///< The type of the input
    using output_t     = typename rbm_base_traits<parent_t>::output_t; ///< The type of the output

    using generator_t = inmemory_data_generator_desc<dll::autoencoder, dll::batch_size<Desc::BatchSize>>; ///< The generator to use

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

    weight pbias        = 0.002; ///< The bias for sparsity (LEE)
    weight pbias_lambda = 5;     ///< The lambda for sparsity (LEE)

    weight gradient_clip = 5.0; ///< The default gradient clipping value

    /*!
     * \brief Construct an empty rbm_base
     */
    rbm_base() {
        //Nothing to do
    }

    /*!
     * \brief Backup the weights in the secondary weights matrix
     */
    void backup_weights() {
        unique_safe_get(as_derived().bak_w) = as_derived().w;
        unique_safe_get(as_derived().bak_b) = as_derived().b;
        unique_safe_get(as_derived().bak_c) = as_derived().c;
    }

    /*!
     * \brief Restore the weights from the secondary weights matrix
     */
    void restore_weights() {
        as_derived().w = *as_derived().bak_w;
        as_derived().b = *as_derived().bak_b;
        as_derived().c = *as_derived().bak_c;
    }

    /*!
     * \brief Compute the reconstruction error for the given input
     */
    template<typename Input>
    double reconstruction_error(const Input& item) {
        return parent_t::reconstruction_error_impl(item, as_derived());
    }

    /*!
     * \brief Returns the trainable variables of this layer.
     * \return a tuple containing references to the variables of this layer
     */
    decltype(auto) trainable_parameters(){
        return std::make_tuple(std::ref(as_derived().w), std::ref(as_derived().b));
    }

    /*!
     * \brief Returns the trainable variables of this layer.
     * \return a tuple containing references to the variables of this layer
     */
    decltype(auto) trainable_parameters() const {
        return std::make_tuple(std::cref(as_derived().w), std::cref(as_derived().b));
    }

    //Normal Train functions

    /*!
     * \brief Train the RBM with the data from the generator
     * \param generator The generator to use for data
     */
    template <bool EnableWatcher = true, typename RW = void, typename Generator, typename... Args, cpp_enable_iff(is_generator<Generator>)>
    double train(Generator& generator, size_t max_epochs, Args... args) {
        dll::rbm_trainer<parent_t, EnableWatcher, RW> trainer(args...);
        return trainer.train(as_derived(), generator, max_epochs);
    }

    /*!
     * \brief Train the RBM with the data from the given container
     * \param training_data the training data
     */
    template <bool EnableWatcher = true, typename RW = void, typename Input, typename... Args, cpp_enable_iff(!is_generator<Input>)>
    double train(const Input& training_data, size_t max_epochs, Args... args) {
        // Create a new generator around the data
        auto generator = make_generator(training_data, training_data, training_data.size(), generator_t{});

        generator->set_safe();

        dll::rbm_trainer<parent_t, EnableWatcher, RW> trainer(args...);
        return trainer.train(as_derived(), *generator, max_epochs);
    }

    /*!
     * \brief Train the RBM with the data from the given iterators
     * \param first The iterator to the beginning of the data
     * \param last The iterator to the end of the data
     */
    template <bool EnableWatcher = true, typename RW = void, typename Iterator, typename... Args>
    double train(Iterator&& first, Iterator&& last, size_t max_epochs, Args... args) {
        // Create a new generator around the data
        auto generator = make_generator(first, last, first, last, std::distance(first, last), generator_t{});

        generator->set_safe();

        dll::rbm_trainer<parent_t, EnableWatcher, RW> trainer(args...);
        return trainer.train(as_derived(), *generator, max_epochs);
    }

    //Train denoising autoencoder

    /*!
     * \brief Train the RBM with the data from the generator as denoising auto-encoder
     * \param generator The generator to use for data
     * \param max_epochs The maximum number of epochs for training
     * \params args The args to pass to the trainer
     */
    template <bool EnableWatcher = true, typename RW = void, typename Generator, typename... Args>
    double train_denoising(Generator& generator, size_t max_epochs, Args... args) {
        dll::rbm_trainer<parent_t, EnableWatcher, RW> trainer(args...);
        return trainer.train(as_derived(), generator, max_epochs);
    }

    /*!
     * \brief Train the RBM with the data from the generator as denoising auto-encoder
     * \param noisy The noisy images
     * \param clean The clean images
     * \param max_epochs The maximum number of epochs for training
     * \params args The args to pass to the trainer
     */
    template <bool EnableWatcher = true, typename RW = void, typename Noisy, typename Clean, typename... Args>
    double train_denoising(const Noisy& noisy, const Clean& clean, size_t max_epochs, Args... args) {
        // Create a new generator around the data
        auto generator = make_generator(noisy, clean, noisy.size(), generator_t{});

        generator->set_safe();

        dll::rbm_trainer<parent_t, EnableWatcher, RW> trainer(args...);
        return trainer.train(as_derived(), *generator, max_epochs);
    }

    /*!
     * \brief Train the RBM with the data from the generator as denoising auto-encoder
     * \param noisy_it Iterator pointing to the first element of the noisy images
     * \param noisy_end Iterator pointing to the past-the-end element of the noisy images
     * \param clean_it Iterator pointing to the first element of the clean images
     * \param clean_end Iterator pointing to the past-the-end element of the clean images
     * \params args The args to pass to the trainer
     */
    template <typename NIterator, typename CIterator, bool EnableWatcher = true, typename RW = void, typename... Args>
    double train_denoising(NIterator noisy_it, NIterator noisy_end, CIterator clean_it, CIterator clean_end, size_t max_epochs, Args... args) {
        // Create a new generator around the data
        auto generator = make_generator(
            noisy_it, noisy_end,
            clean_it, clean_end,
            std::distance(clean_it, clean_end),
            generator_t{}
            );

        generator->set_safe();

        dll::rbm_trainer<parent_t, EnableWatcher, RW> trainer(args...);
        return trainer.train(as_derived(), *generator, max_epochs);
    }

    // Features

    /*!
     * \brief Return the features corresponding to the given input
     * \param input The input to extract the features from
     */
    template<typename Input>
    output_one_t features(const Input& input){
        return activate_hidden(input);
    }

    /*!
     * \brief Return the activation probabilities corresponding to the given input
     * \param input The input to extract the features from
     */
    template<typename Input>
    output_one_t activate_hidden(const Input& input){
        auto output = as_derived().template prepare_one_output<Input>();
        as_derived().activate_hidden(output, input);
        return output;
    }

    //I/O functions

    /*!
     * \brief Store the weights in the given file
     */
    void store(const std::string& file) const {
        store(file, as_derived());
    }

    /*!
     * \brief Store the weights using the given stream
     */
    void store(std::ostream& os) const {
        store(os, as_derived());
    }

    /*!
     * \brief Load the weights from the given file
     */
    void load(const std::string& file) {
        load(file, as_derived());
    }

    /*!
     * \brief Load the weights from the given stream
     */
    void load(std::istream& is) {
        load(is, as_derived());
    }

private:
    /*!
     * \brief Load the weigts into the given stream
     */
    static void store(std::ostream& os, const parent_t& rbm) {
        cpp::binary_write_all(os, rbm.w);
        cpp::binary_write_all(os, rbm.b);
        cpp::binary_write_all(os, rbm.c);
    }

    /*!
     * \brief Load the weigts from the given stream
     */
    static void load(std::istream& is, parent_t& rbm) {
        cpp::binary_load_all(is, rbm.w);
        cpp::binary_load_all(is, rbm.b);
        cpp::binary_load_all(is, rbm.c);
    }

    /*!
     * \brief Load the weigts into the given file
     */
    static void store(const std::string& file, const parent_t& rbm) {
        std::ofstream os(file, std::ofstream::binary);
        store(os, rbm);
    }

    /*!
     * \brief Load the weigts from the given file
     */
    static void load(const std::string& file, parent_t& rbm) {
        std::ifstream is(file, std::ifstream::binary);
        load(is, rbm);
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
