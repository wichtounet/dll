//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Deep Belief Network implementation.
 *
 * In this library, a DBN can also be used with standard neural network layers,
 * in which case, it acts as a standard neural network and cannot be
 * pretrained.
 */

#pragma once

#include "cpp_utils/static_if.hpp"

#include "unit_type.hpp"
#include "trainer/dbn_trainer.hpp"
#include "trainer/conjugate_gradient.hpp"
#include "dbn_common.hpp"
#include "svm_common.hpp"
#include "util/flatten.hpp"
#include "util/export.hpp"
#include "util/timers.hpp"
#include "dbn_detail.hpp" //dbn_detail namespace

namespace dll {

/*!
 * \brief A Deep Belief Network implementation
 */
template <typename Desc>
struct dbn final {
    using desc      = Desc;      ///< The network descriptor
    using this_type = dbn<desc>; ///< The network type

    using layers_t = typename desc::layers; ///< The layers container type

    static_assert(!(dbn_traits<this_type>::batch_mode() && layers_t::has_shuffle_layer), "batch_mode dbn does not support shuffle in layers");

    template <std::size_t N>
    using layer_type = detail::layer_type_t<N, layers_t>; ///< The type of the layer at index Nth

    using weight = typename dbn_detail::extract_weight_t<0, this_type>::type; ///< The tpyeof the weights

    using watcher_t = typename desc::template watcher_t<this_type>; ///< The watcher type

    using input_t = typename dbn_detail::layer_input_simple<this_type, 0>::type; ///< The input type of the network

    template <std::size_t B>
    using input_batch_t = typename dbn_detail::layer_input_batch<this_type, 0>::template type<B>; ///< The input batch type of the network for a batch size of B

    template <std::size_t N>
    using layer_input_one_t = dbn_detail::layer_input_one_t<this_type, N>;

    template <std::size_t N>
    using layer_output_one_t = dbn_detail::layer_output_one_t<this_type, N>;

    template <std::size_t N>
    using layer_input_t = dbn_detail::layer_input_t<this_type, N>;

    template <std::size_t N>
    using layer_output_t = dbn_detail::layer_output_t<this_type, N>;

    using label_output_t = layer_input_one_t<layers_t::size - 1>;
    using output_one_t   = layer_output_one_t<layers_t::size - 1>; ///< The type of a single output of the network

    using output_t = std::conditional_t<
        dbn_traits<this_type>::is_multiplex(),
        std::vector<output_one_t>,
        output_one_t>; ///< The output type of the network

    using full_output_t = etl::dyn_vector<weight>;

    using svm_samples_t = std::conditional_t<
        dbn_traits<this_type>::concatenate(),
        std::vector<etl::dyn_vector<weight>>, //In full mode, use a simple 1D vector
        layer_output_t<layers_t::size - 1>>;  //In normal mode, use the output of the last layer

    using for_each_impl_t      = dbn_detail::for_each_impl<this_type, std::make_index_sequence<layers_t::size>>;
    using for_each_pair_impl_t = dbn_detail::for_each_impl<this_type, std::make_index_sequence<layers_t::size - 1>>;

    using const_for_each_impl_t      = dbn_detail::for_each_impl<const this_type, std::make_index_sequence<layers_t::size>>;
    using const_for_each_pair_impl_t = dbn_detail::for_each_impl<const this_type, std::make_index_sequence<layers_t::size - 1>>;

    static constexpr const std::size_t layers         = layers_t::size;     ///< The number of layers
    static constexpr const std::size_t batch_size     = desc::BatchSize;    ///< The batch size (for finetuning)
    static constexpr const std::size_t big_batch_size = desc::BigBatchSize; ///< The number of pretraining batch to do at once

    layers_t tuples; ///< The layers

    weight learning_rate     = 0.1;  ///< The learning rate for finetuning
    weight lr_bold_inc       = 1.05; ///< The multiplicative increase of learning rate for the bold driver
    weight lr_bold_dec       = 0.5;  ///< The multiplicative decrease of learning rate for the bold driver
    weight lr_step_gamma     = 0.5;  ///< The multiplicative decrease of learning rate for the step driver
    std::size_t lr_step_size = 10;   ///< The number of steps after which the step driver decreases the learning rate

    weight initial_momentum     = 0.5; ///< The initial momentum
    weight final_momentum       = 0.9; ///< The final momentum applied after *final_momentum_epoch* epoch
    weight final_momentum_epoch = 6;   ///< The epoch at which momentum change

    weight l1_weight_cost = 0.0002; ///< The weight cost for L1 weight decay
    weight l2_weight_cost = 0.0002; ///< The weight cost for L2 weight decay

    weight momentum = 0; ///< The current momentum

    bool batch_mode_run = false;

#ifdef DLL_SVM_SUPPORT
    //TODO Ideally these fields should be private
    svm::model svm_model;    ///< The learned model
    svm::problem problem;    ///< libsvm is stupid, therefore, you cannot destroy the problem if you want to use the model...
    bool svm_loaded = false; ///< Indicates if a SVM model has been loaded (and therefore must be saved)
#endif                       //DLL_SVM_SUPPORT

private:
    cpp::thread_pool<!dbn_traits<this_type>::is_serial()> pool;

    mutable int fake_resource; ///< Simple field to get a reference from for resource management

public:
    /*!
     * Constructs a DBN and initializes all its members.
     *
     * This is the only way to create a DBN.
     */
    dbn() {
        //Nothing else to init
    }

    //No copying
    dbn(const dbn& dbn) = delete;
    dbn& operator=(const dbn& dbn) = delete;

    //No moving
    dbn(dbn&& dbn) = delete;
    dbn& operator=(dbn&& dbn) = delete;

    /*!
     * \brief Prints a textual representation of the network.
     */
    void display() const {
        std::size_t parameters = 0;

        std::cout << "DBN with " << layers << " layers" << std::endl;

        for_each_layer([&parameters](auto& layer) {
            std::cout << "    ";
            cpp::static_if<decay_layer_traits<decltype(layer)>::is_rbm_layer()>([&](auto f) {
                parameters += f(layer).parameters();
            });
            layer.display();
        });

        std::cout << "Total parameters: " << parameters << std::endl;
    }

    /*!
     * \brief Backup the weights of all the layers into a temporary storage.
     *
     * Only one temporary storage is available, i.e. calling this function
     * twice will erase the first saved weights.
     */
    void backup_weights() {
        for_each_layer([](auto& layer) {
            cpp::static_if<decay_layer_traits<decltype(layer)>::is_trained()>([&layer](auto f) {
                f(layer).backup_weights();
            });
        });
    }

    /*!
     * \brief Restore the weights previously saved.
     *
     * This function has no effect if the weights were not saved before.
     * Calling this function twice will restore the same weights.
     */
    void restore_weights() {
        for_each_layer([](auto& layer) {
            cpp::static_if<decay_layer_traits<decltype(layer)>::is_trained()>([&layer](auto f) {
                f(layer).restore_weights();
            });
        });
    }

    /*!
     * \brief Store the network weights to the given file.
     * \param file The path to the file
     */
    void store(const std::string& file) const {
        std::ofstream os(file, std::ofstream::binary);
        store(os);
    }

    /*!
     * \brief Load the network weights from the given file.
     * \param file The path to the file
     */
    void load(const std::string& file) {
        std::ifstream is(file, std::ifstream::binary);
        load(is);
    }

    /*!
     * \brief Store the network weights using the given output stream.
     * \param os The stream to output the network weights to.
     */
    void store(std::ostream& os) const {
        for_each_layer([&os](auto& layer) {
            cpp::static_if<decay_layer_traits<decltype(layer)>::is_rbm_layer()>([&](auto f) {
                f(layer).store(os);
            });
        });

#ifdef DLL_SVM_SUPPORT
        svm_store(*this, os);
#endif //DLL_SVM_SUPPORT
    }

    /*!
     * \brief Load the network weights using the given output stream.
     * \param is The stream to load the network weights from.
     */
    void load(std::istream& is) {
        for_each_layer([&is](auto& layer) {
            cpp::static_if<decay_layer_traits<decltype(layer)>::is_rbm_layer()>([&](auto f) {
                f(layer).load(is);
            });
        });

#ifdef DLL_SVM_SUPPORT
        svm_load(*this, is);
#endif //DLL_SVM_SUPPORT
    }

    /*!
     * \brief Returns the Nth layer.
     * \return The Nth layer
     * \tparam N The index of the layer to return (from 0)
     */
    template <std::size_t N>
    layer_type<N>& layer_get() {
        return detail::layer_get<N>(tuples);
    }

    /*!
     * \brief Returns the Nth layer.
     * \return The Nth layer
     * \tparam N The index of the layer to return (from 0)
     */
    template <std::size_t N>
    constexpr const layer_type<N>& layer_get() const {
        return detail::layer_get<N>(tuples);
    }

    template <std::size_t N>
    static constexpr std::size_t layer_input_size() noexcept {
        return layer_traits<layer_type<N>>::input_size();
    }

    template <std::size_t N>
    static constexpr std::size_t layer_output_size() noexcept {
        return layer_traits<layer_type<N>>::output_size();
    }

    static constexpr std::size_t input_size() noexcept {
        return layer_traits<layer_type<0>>::input_size();
    }

    static constexpr std::size_t output_size() noexcept {
        return layer_traits<layer_type<layers - 1>>::output_size();
    }

    static std::size_t full_output_size() noexcept {
        std::size_t output = 0;
        detail::for_each_layer_type<this_type>([&output](auto* layer) {
            output += std::decay_t<std::remove_pointer_t<decltype(layer)>>::output_size();
        });
        return output;
    }

    /*!
     * \brief Indicates if training should save memory (true) or run as efficiently as possible (false).
     *
     * This can be configured in the dbn type or using the batch_mode_run field.
     *
     * \return true if the training should save memory, false otherwise.
     */
    [[deprecated("use batch_mode instead")]] bool save_memory() const noexcept {
        return batch_mode();
    }

    /*!
     * \brief Indicates if training should save memory (true) or run as efficiently as possible (false).
     *
     * This can be configured in the dbn type or using the batch_mode_run field.
     *
     * \return true if the training should save memory, false otherwise.
     */
    bool batch_mode() const noexcept {
        return dbn_traits<this_type>::batch_mode() || batch_mode_run;
    }

    /*!
     * \brief Pretrain the network by training all layers in an unsupervised manner.
     *
     * \param first Iterator to the first element of the sequence
     * \param last Iterator to the last element of the sequence
     * \param max_epochs The maximum number of epochs for pretraining.
     *
     * \tparam Iterator the type of iterator
     */
    template <typename Iterator>
    void pretrain(Iterator first, Iterator last, std::size_t max_epochs) {
        dll::auto_timer timer("dbn:pretrain");

        watcher_t watcher;

        watcher.pretraining_begin(*this, max_epochs);

        //Pretrain each layer one-by-one
        if (batch_mode()) {
            std::cout << "DBN: Pretraining done in batch mode" << std::endl;

            if (layers_t::has_shuffle_layer) {
                std::cout << "warning: batch_mode dbn does not support shuffle in layers (will be ignored)";
            }

            pretrain_layer_batch<0>(first, last, watcher, max_epochs);
        } else {
            pretrain_layer<0>(first, last, watcher, max_epochs, fake_resource);
        }

        watcher.pretraining_end(*this);
    }

    /*!
     * \brief Pretrain the network by training all layers in an unsupervised
     * manner.
     */
    template <typename Samples>
    void pretrain(const Samples& training_data, std::size_t max_epochs) {
        pretrain(training_data.begin(), training_data.end(), max_epochs);
    }

    /*!
     * \brief Pretrain the network by training all layers in an unsupervised
     * manner, the network will learn to reconstruct noisy input.
     */
    template <typename NIterator, typename CIterator>
    void pretrain_denoising(NIterator nit, NIterator nend, CIterator cit, CIterator cend, std::size_t max_epochs) {
        dll::auto_timer timer("dbn:pretrain:denoising");

        watcher_t watcher;

        watcher.pretraining_begin(*this, max_epochs);

        cpp_assert(!batch_mode(), "pretrain_denoising has not yet been implemented in memory");

        //Pretrain each layer one-by-one
        pretrain_layer_denoising<0>(nit, nend, cit, cend, watcher, max_epochs, fake_resource, fake_resource);

        watcher.pretraining_end(*this);
    }

    /*!
     * \brief Pretrain the network by training all layers in an unsupervised
     * manner, the network will learn to reconstruct noisy input.
     */
    template <typename Noisy, typename Clean>
    void pretrain_denoising(const Noisy& noisy, const Clean& clean, std::size_t max_epochs) {
        pretrain_denoising(noisy.begin(), noisy.end(), clean.begin(), clean.end(), max_epochs);
    }

    template <typename Iterator, typename LabelIterator>
    void train_with_labels(Iterator&& first, Iterator&& last, LabelIterator&& lfirst, LabelIterator&& llast, std::size_t labels, std::size_t max_epochs) {
        dll::auto_timer timer("dbn:train:labels");

        cpp_assert(std::distance(first, last) == std::distance(lfirst, llast), "There must be the same number of values than labels");
        cpp_assert(dll::input_size(layer_get<layers - 1>()) == dll::output_size(layer_get<layers - 2>()) + labels, "There is no room for the labels units");

        watcher_t watcher;

        watcher.pretraining_begin(*this, max_epochs);

        train_with_labels<0>(first, last, watcher, std::forward<LabelIterator>(lfirst), std::forward<LabelIterator>(llast), labels, max_epochs);

        watcher.pretraining_end(*this);
    }

    template <typename Samples, typename Labels>
    void train_with_labels(const Samples& training_data, const Labels& training_labels, std::size_t labels, std::size_t max_epochs) {
        cpp_assert(training_data.size() == training_labels.size(), "There must be the same number of values than labels");
        cpp_assert(dll::input_size(layer_get<layers - 1>()) == dll::output_size(layer_get<layers - 2>()) + labels, "There is no room for the labels units");

        train_with_labels(training_data.begin(), training_data.end(), training_labels.begin(), training_labels.end(), labels, max_epochs);
    }

    size_t predict_labels(const input_t& item, std::size_t labels) const {
        cpp_assert(dll::input_size(layer_get<layers - 1>()) == dll::output_size(layer_get<layers - 2>()) + labels, "There is no room for the labels units");

        label_output_t output_a = layer_get<layers - 1>().prepare_one_input();

        predict_labels<0>(item, output_a, labels);

        return std::distance(
            std::prev(output_a.end(), labels),
            std::max_element(std::prev(output_a.end(), labels), output_a.end()));
    }

    //activation_probabilities_sub

    /*!
     * \brief Returns the output features of the Ith layer for the given sample and saves them in the given container
     * \param sample The sample to get features from
     * \param result The container where the results will be saved
     * \tparam I The index of the layer for features (starts at 0)
     * \return the output features of the Ith layer of the network
     */
    template <std::size_t I, typename Output, typename T = this_type>
    auto activation_probabilities_sub(const input_t& sample, Output& result) const {
        activation_probabilities<0, I + 1>(sample, result);

        return result;
    }

    /*!
     * \brief Returns the output features of the Ith layer for the given sample
     * \param sample The sample to get features from
     * \tparam I The index of the layer for features (starts at 0)
     * \return the output features of the Ith layer of the network
     */
    template <std::size_t I>
    auto activation_probabilities_sub(const input_t& sample) const {
        auto result = prepare_output<I>();
        return activation_probabilities_sub<I>(sample, result);
    }

    //Note: features_sub are alias functions for activation_probabilities_sub

    /*!
     * \brief Returns the output features of the Ith layer for the given sample and saves them in the given container
     * \param sample The sample to get features from
     * \param result The container where the results will be saved
     * \tparam I The index of the layer for features (starts at 0)
     * \return the output features of the Ith layer of the network
     */
    template <std::size_t I, typename Output, typename T = this_type>
    auto features_sub(const input_t& sample, Output& result) const {
        return activation_probabilities_sub<I>(sample, result);
    }

    /*!
     * \brief Returns the output features of the Ith layer for the given sample
     * \param sample The sample to get features from
     * \tparam I The index of the layer for features (starts at 0)
     * \return the output features of the Ith layer of the network
     */
    template <std::size_t I>
    auto features_sub(const input_t& sample) const {
        return activation_probabilities_sub<I>(sample);
    }

    // activation_probabilities

    /*!
     * \brief Computes the output features for the given sample and saves them in the given container
     * \param sample The sample to get features from
     * \param result The container where to save the features
     * \return result
     */
    auto activation_probabilities(const input_t& sample, output_t& result) const {
        return activation_probabilities_sub<layers - 1>(sample, result);
    }

    /*!
     * \brief Returns the output features for the given sample
     * \param sample The sample to get features from
     * \return the output features of the last layer of the network
     */
    auto activation_probabilities(const input_t& sample) const {
        return activation_probabilities_sub<layers - 1>(sample);
    }

    //Note: features are alias functions for activation_probabilities

    /*!
     * \brief Computes the output features for the given sample and saves them in the given container
     * \param sample The sample to get features from
     * \param result The container where to save the features
     * \return result
     */
    auto features(const input_t& sample, output_t& result) const {
        return activation_probabilities(sample, result);
    }

    /*!
     * \brief Returns the output features for the given sample
     * \param sample The sample to get features from
     * \return the output features of the last layer of the network
     */
    auto features(const input_t& sample) const {
        return activation_probabilities(sample);
    }

    /*!
     * \brief Save the features generated for the given sample in the given file.
     * \param sample The sample to get features from
     * \param file The output file
     * \param f The format of the exported features
     */
    void save_features(const input_t& sample, const std::string& file, format f = format::DLL) const {
        cpp_assert(f == format::DLL, "Only DLL format is supported for now");

        decltype(auto) probs = features(sample);

        if (f == format::DLL) {
            export_features_dll(probs, file);
        }
    }

    void full_activation_probabilities(const input_t& sample, full_output_t& result) const {
        std::size_t i = 0;
        full_activation_probabilities<0>(sample, i, result);
    }

    full_output_t full_activation_probabilities(const input_t& item_data) const {
        full_output_t result(full_output_size());

        full_activation_probabilities(item_data, result);

        return result;
    }

    template <typename DBN = this_type, cpp_enable_if(dbn_traits<DBN>::concatenate())>
    full_output_t get_final_activation_probabilities(const input_t& sample) const {
        return full_activation_probabilities(sample);
    }

    template <typename DBN = this_type, cpp_disable_if(dbn_traits<DBN>::concatenate())>
    output_t get_final_activation_probabilities(const input_t& sample) const {
        return activation_probabilities(sample);
    }

    size_t predict_label(const output_t& result) const {
        return std::distance(result.begin(), std::max_element(result.begin(), result.end()));
    }

    size_t predict(const input_t& item) const {
        auto result = activation_probabilities(item);
        return predict_label(result);
    }

    /*!
     * \brief Fine tune the network for classifcation.
     * \param training_data A container containing all the samples
     * \param labels A container containing all the labels
     * \param max_epochs The maximum number of epochs to train the network for.
     * \return The final classification error
     */
    template <typename Samples, typename Labels>
    weight fine_tune(const Samples& training_data, Labels& labels, size_t max_epochs) {
        return fine_tune(training_data.begin(), training_data.end(), labels.begin(), labels.end(), max_epochs);
    }

    /*!
     * \brief Fine tune the network for classifcation.
     * \param first Iterator to the first sample
     * \param last Iterator to the last sample
     * \param lfirst Iterator the first label
     * \param llast Iterator the last label
     * \param max_epochs The maximum number of epochs to train the network for.
     * \return The final classification error
     */
    template <typename Iterator, typename LIterator>
    weight fine_tune(Iterator&& first, Iterator&& last, LIterator&& lfirst, LIterator&& llast, size_t max_epochs) {
        dll::auto_timer timer("dbn:train:ft");

        dll::dbn_trainer<this_type> trainer;
        return trainer.train(*this,
                             std::forward<Iterator>(first), std::forward<Iterator>(last),
                             std::forward<LIterator>(lfirst), std::forward<LIterator>(llast),
                             max_epochs);
    }

    /*!
     * \brief Fine tune the network for autoencoder.
     * \param training_data A container containing all the samples
     * \param max_epochs The maximum number of epochs to train the network for.
     * \return The final classification error
     */
    template <typename Samples>
    weight fine_tune_ae(const Samples& training_data, size_t max_epochs) {
        return fine_tune_ae(
            training_data.begin(), training_data.end(),
            max_epochs);
    }

    /*!
     * \brief Fine tune the network for autoencoder.
     * \param first Iterator to the first sample
     * \param last Iterator to the last sample
     * \param max_epochs The maximum number of epochs to train the network for.
     * \return The final classification error
     */
    template <typename Iterator>
    weight fine_tune_ae(Iterator&& first, Iterator&& last, size_t max_epochs) {
        dll::auto_timer timer("dbn:train:ft:ae");

        cpp_assert(dll::input_size(layer_get<0>()) == dll::output_size(layer_get<layers - 1>()), "The network is not build as an autoencoder");

        dll::dbn_trainer<this_type> trainer;
        return trainer.train_ae(*this,
                                first, last,
                                max_epochs);
    }

    template <std::size_t I, typename T = this_type, cpp_disable_if(dbn_traits<T>::is_multiplex())>
    auto prepare_output() const {
        return layer_get<I>().template prepare_one_output<layer_input_one_t<I>>();
    }

    template <std::size_t I, typename T = this_type, cpp_enable_if(dbn_traits<T>::is_multiplex())>
    auto prepare_output() const {
        return std::vector<layer_output_one_t<layers - 1>>();
    }

    auto prepare_one_output() const {
        return prepare_output<layers - 1>();
    }

    template <typename Functor>
    void for_each_layer(Functor&& functor) {
        for_each_impl_t(*this).for_each_layer(std::forward<Functor>(functor));
    }

    template <typename Functor>
    void for_each_layer_i(Functor&& functor) {
        for_each_impl_t(*this).for_each_layer_i(std::forward<Functor>(functor));
    }

    template <typename Functor>
    void for_each_layer_pair(Functor&& functor) {
        for_each_pair_impl_t(*this).for_each_layer_pair(std::forward<Functor>(functor));
    }

    template <typename Functor>
    void for_each_layer_pair_i(Functor&& functor) {
        for_each_pair_impl_t(*this).for_each_layer_pair_i(std::forward<Functor>(functor));
    }

    template <typename Functor>
    void for_each_layer_rpair(Functor&& functor) {
        for_each_pair_impl_t(*this).for_each_layer_rpair(std::forward<Functor>(functor));
    }

    template <typename Functor>
    void for_each_layer_rpair_i(Functor&& functor) {
        for_each_pair_impl_t(*this).for_each_layer_rpair_i(std::forward<Functor>(functor));
    }

    template <typename Functor>
    void for_each_layer(Functor&& functor) const {
        const_for_each_impl_t(*this).for_each_layer(std::forward<Functor>(functor));
    }

    template <typename Functor>
    void for_each_layer_i(Functor&& functor) const {
        const_for_each_impl_t(*this).for_each_layer_i(std::forward<Functor>(functor));
    }

    template <typename Functor>
    void for_each_layer_pair(Functor&& functor) const {
        const_for_each_pair_impl_t(*this).for_each_layer_pair(std::forward<Functor>(functor));
    }

    template <typename Functor>
    void for_each_layer_pair_i(Functor&& functor) const {
        const_for_each_pair_impl_t(*this).for_each_layer_pair_i(std::forward<Functor>(functor));
    }

    template <typename Functor>
    void for_each_layer_rpair(Functor&& functor) const {
        const_for_each_pair_impl_t(*this).for_each_layer_rpair(std::forward<Functor>(functor));
    }

    template <typename Functor>
    void for_each_layer_rpair_i(Functor&& functor) const {
        const_for_each_pair_impl_t(*this).for_each_layer_rpair_i(std::forward<Functor>(functor));
    }

#ifdef DLL_SVM_SUPPORT

    template <typename Samples, typename Labels>
    bool svm_train(const Samples& training_data, const Labels& labels, const svm_parameter& parameters = default_svm_parameters()) {
        cpp::stop_watch<std::chrono::seconds> watch;

        make_problem(training_data, labels, dbn_traits<this_type>::scale());

        //Make libsvm quiet
        svm::make_quiet();

        //Make sure parameters are not messed up
        if (!svm::check(problem, parameters)) {
            return false;
        }

        //Train the SVM
        svm_model = svm::train(problem, parameters);

        svm_loaded = true;

        std::cout << "SVM training took " << watch.elapsed() << "s" << std::endl;

        return true;
    }

    template <typename Iterator, typename LIterator>
    bool svm_train(Iterator&& first, Iterator&& last, LIterator&& lfirst, LIterator&& llast, const svm_parameter& parameters = default_svm_parameters()) {
        cpp::stop_watch<std::chrono::seconds> watch;

        make_problem(
            std::forward<Iterator>(first), std::forward<Iterator>(last),
            std::forward<LIterator>(lfirst), std::forward<LIterator>(llast),
            dbn_traits<this_type>::scale());

        //Make libsvm quiet
        svm::make_quiet();

        //Make sure parameters are not messed up
        if (!svm::check(problem, parameters)) {
            return false;
        }

        //Train the SVM
        svm_model = svm::train(problem, parameters);

        svm_loaded = true;

        std::cout << "SVM training took " << watch.elapsed() << "s" << std::endl;

        return true;
    }

    template <typename Samples, typename Labels>
    bool svm_grid_search(const Samples& training_data, const Labels& labels, std::size_t n_fold = 5, const svm::rbf_grid& g = svm::rbf_grid()) {
        make_problem(training_data, labels, dbn_traits<this_type>::scale());

        //Make libsvm quiet
        svm::make_quiet();

        auto parameters = default_svm_parameters();

        //Make sure parameters are not messed up
        if (!svm::check(problem, parameters)) {
            return false;
        }

        //Perform a grid-search
        svm::rbf_grid_search(problem, parameters, n_fold, g);

        return true;
    }

    template <typename It, typename LIt>
    bool svm_grid_search(It&& first, It&& last, LIt&& lfirst, LIt&& llast, std::size_t n_fold = 5, const svm::rbf_grid& g = svm::rbf_grid()) {
        make_problem(
            std::forward<It>(first), std::forward<It>(last),
            std::forward<LIt>(lfirst), std::forward<LIt>(llast),
            dbn_traits<this_type>::scale());

        //Make libsvm quiet
        svm::make_quiet();

        auto parameters = default_svm_parameters();

        //Make sure parameters are not messed up
        if (!svm::check(problem, parameters)) {
            return false;
        }

        //Perform a grid-search
        svm::rbf_grid_search(problem, parameters, n_fold, g);

        return true;
    }

    double svm_predict(const input_t& sample) {
        auto features = get_final_activation_probabilities(sample);
        return svm::predict(svm_model, features);
    }

#endif //DLL_SVM_SUPPORT

private:
    static void release(int&) {}

    template <typename T>
    static void release(std::vector<T>& resource) {
        std::vector<T>().swap(resource);
    }

    //By default all layer are trained
    template <std::size_t I, class Enable = void>
    struct train_next : std::true_type {};

    //The last layer is not always trained (softmax for instance)
    template <std::size_t I>
    struct train_next<I, std::enable_if_t<(I == layers - 1)>> : cpp::bool_constant<layer_traits<layer_type<I>>::pretrain_last()> {};

    template <std::size_t I, typename Enable = void>
    struct inline_next : std::false_type {};

    template <std::size_t I>
    struct inline_next<I, std::enable_if_t<(I < layers)>> : cpp::bool_constant<layer_traits<layer_type<I>>::is_pooling_layer()> {};

    template <std::size_t I, typename Iterator, typename Container>
    void inline_layer(Iterator first, Iterator last, watcher_t& watcher, std::size_t max_epochs, Container& previous) {
        decltype(auto) layer = layer_get<I>();
        decltype(auto) next_layer = layer_get<I + 1>();

        watcher.template pretrain_layer<std::decay_t<decltype(next_layer)>>(*this, I+1, dbn_detail::fast_distance(first, last));

        auto next_a = next_layer.template prepare_output<layer_input_one_t<I>>(std::distance(first, last));

        maybe_parallel_foreach_i(pool, first, last, [&layer, &next_layer, &next_a](auto& v, std::size_t i) {
            auto tmp = layer.template prepare_one_output<layer_input_one_t<I + 1>>();

            layer.activate_hidden(tmp, v);
            next_layer.activate_hidden(next_a[i], tmp);
        });

        this_type::release(previous);

        pretrain_layer<I + 2>(next_a.begin(), next_a.end(), watcher, max_epochs, next_a);
    }

    template <std::size_t I, typename Iterator, typename Container, cpp_enable_if((I < layers))>
    void pretrain_layer(Iterator first, Iterator last, watcher_t& watcher, std::size_t max_epochs, Container& previous) {
        using layer_t = layer_type<I>;

        decltype(auto) layer = layer_get<I>();

        watcher.template pretrain_layer<layer_t>(*this, I, dbn_detail::fast_distance(first, last));

        cpp::static_if<layer_traits<layer_t>::is_pretrained()>([&](auto f) {
            f(layer).template train<!watcher_t::ignore_sub,               //Enable the RBM Watcher or not
                                    dbn_detail::rbm_watcher_t<watcher_t>> //Replace the RBM watcher if not void
                (first, last, max_epochs);
        });

        //When the next layer is a pooling layer, a lot of memory can be saved by directly computing
        //the activations of two layers at once
        cpp::static_if<inline_next<I + 1>::value>([&](auto f) {
            f(this)->template inline_layer<I>(first, last, watcher, max_epochs, previous);
        });

        if (train_next<I + 1>::value && !inline_next<I + 1>::value) {
            auto next_a = layer.template prepare_output<layer_input_one_t<I>>(std::distance(first, last));

            maybe_parallel_foreach_i(pool, first, last, [&layer, &next_a](auto& v, std::size_t i) {
                layer.activate_hidden(next_a[i], v);
            });

            //At this point we don't need the storage of the previous layer
            release(previous);

            //In the standard case, pass the output to the next layer
            cpp::static_if<!layer_traits<layer_t>::is_multiplex_layer()>([&](auto f) {
                f(this)->template pretrain_layer<I + 1>(next_a.begin(), next_a.end(), watcher, max_epochs, next_a);
            });

            //In case of a multiplex layer, the output is flattened
            cpp::static_if<layer_traits<layer_t>::is_multiplex_layer()>([&](auto f) {
                auto flattened_next_a = flatten_clr(f(next_a));
                this_type::release(next_a);
                f(this)->template pretrain_layer<I + 1>(flattened_next_a.begin(), flattened_next_a.end(), watcher, max_epochs, flattened_next_a);
            });
        }
    }

    //Stop template recursion
    template <std::size_t I, typename Iterator, typename Container, cpp_enable_if((I == layers))>
    void pretrain_layer(Iterator, Iterator, watcher_t&, std::size_t, Container&) {}

    /* Pretrain with denoising */

    template <std::size_t I, typename NIterator, typename CIterator, typename NContainer, typename CContainer, cpp_enable_if((I < layers))>
    void pretrain_layer_denoising(NIterator nit, NIterator nend, CIterator cit, CIterator cend, watcher_t& watcher, std::size_t max_epochs, NContainer& previous_n, CContainer& previous_c) {
        using layer_t = layer_type<I>;

        decltype(auto) layer = layer_get<I>();

        watcher.template pretrain_layer<layer_t>(*this, I, dbn_detail::fast_distance(nit, nend));

        cpp::static_if<layer_traits<layer_t>::is_pretrained()>([&](auto f) {
            f(layer).template train_denoising<NIterator, CIterator,
                                              !watcher_t::ignore_sub,               //Enable the RBM Watcher or not
                                              dbn_detail::rbm_watcher_t<watcher_t>> //Replace the RBM watcher if not void
                (nit, nend, cit, cend, max_epochs);
        });

        if (train_next<I + 1>::value) {
            auto next_n = layer.template prepare_output<layer_input_one_t<I>>(std::distance(nit, nend));
            auto next_c = layer.template prepare_output<layer_input_one_t<I>>(std::distance(nit, nend));

            maybe_parallel_foreach_i(pool, nit, nend, [&layer, &next_n](auto& v, std::size_t i) {
                layer.activate_hidden(next_n[i], v);
            });

            maybe_parallel_foreach_i(pool, cit, cend, [&layer, &next_c](auto& v, std::size_t i) {
                layer.activate_hidden(next_c[i], v);
            });

            //At this point we don't need the storage of the previous layer
            release(previous_n);
            release(previous_c);

            //In the standard case, pass the output to the next layer
            pretrain_layer_denoising<I + 1>(next_n.begin(), next_n.end(), next_c.begin(), next_c.end(), watcher, max_epochs, next_n, next_c);

            static_assert(!layer_traits<layer_t>::is_multiplex_layer(), "Denoising pretraining does not support multiplex layer");
        }
    }

    //Stop template recursion
    template <std::size_t I, typename NIterator, typename CIterator, typename NContainer, typename CContainer, cpp_enable_if((I == layers))>
    void pretrain_layer_denoising(NIterator, NIterator, CIterator, CIterator, watcher_t&, std::size_t, NContainer&, CContainer&) {}

    /* Pretrain in batch mode */

    //By default no layer is ignored
    template <std::size_t I, class Enable = void>
    struct batch_layer_ignore : std::false_type {};

    //Transform and pooling layers can safely be skipped
    template <std::size_t I>
    struct batch_layer_ignore<I, std::enable_if_t<(I < layers)>> : cpp::or_u<layer_traits<layer_type<I>>::is_pooling_layer(), layer_traits<layer_type<I>>::is_transform_layer(), layer_traits<layer_type<I>>::is_standard_layer(), !layer_traits<layer_type<I>>::pretrain_last()> {};

    //Special handling for the layer 0
    //data is coming from iterators not from input
    template <std::size_t I, typename Iterator, cpp_enable_if((I == 0 && !batch_layer_ignore<I>::value))>
    void pretrain_layer_batch(Iterator first, Iterator last, watcher_t& watcher, std::size_t max_epochs) {
        using layer_t = layer_type<I>;

        decltype(auto) rbm = layer_get<I>();

        watcher.template pretrain_layer<layer_t>(*this, I, 0);

        using rbm_trainer_t = dll::rbm_trainer<layer_t, !watcher_t::ignore_sub, dbn_detail::rbm_watcher_t<watcher_t>, false>;

        //Initialize the RBM trainer
        rbm_trainer_t r_trainer;

        //Init the RBM and training parameters
        r_trainer.init_training(rbm, first, last); //TODO This may be highly slow...

        //Get the specific trainer (CD)
        auto trainer = rbm_trainer_t::get_trainer(rbm);

        //Several RBM batches are propagated at once
        auto total_batch_size = big_batch_size * get_batch_size(rbm);

        std::vector<typename layer_t::input_one_t> input_cache(total_batch_size);

        //Train for max_epochs epoch
        for (std::size_t epoch = 0; epoch < max_epochs; ++epoch) {
            std::size_t big_batch = 0;

            //Create a new context for this epoch
            rbm_training_context context;

            r_trainer.init_epoch();

            auto it  = first;
            auto end = last;

            while (it != end) {
                //Fill the input cache
                std::size_t i = 0;
                while (it != end && i < total_batch_size) {
                    input_cache[i++] = *it++;
                }

                if (big_batch_size == 1) {
                    //Train the RBM on this batch
                    r_trainer.train_batch(input_cache.begin(), input_cache.end(), input_cache.begin(), input_cache.end(), trainer, context, rbm);
                } else {
                    //Train the RBM on this big batch
                    r_trainer.train_sub(input_cache.begin(), input_cache.begin() + i, input_cache.begin(), trainer, context, rbm);
                }

                if (dbn_traits<this_type>::is_verbose()) {
                    watcher.pretraining_batch(*this, big_batch);
                }

                ++big_batch;
            }

            r_trainer.finalize_epoch(epoch, context, rbm);
        }

        r_trainer.finalize_training(rbm);

        //Train the next layer
        pretrain_layer_batch<I + 1>(first, last, watcher, max_epochs);
    }

    //Special handling for untrained layers
    template <std::size_t I, typename Iterator, cpp_enable_if(batch_layer_ignore<I>::value)>
    void pretrain_layer_batch(Iterator first, Iterator last, watcher_t& watcher, std::size_t max_epochs) {
        //We simply go up one layer on pooling layers
        pretrain_layer_batch<I + 1>(first, last, watcher, max_epochs);
    }

    //Normal version
    template <std::size_t I, typename Iterator, cpp_enable_if((I > 0 && I < layers && !dbn_traits<this_type>::is_multiplex() && !batch_layer_ignore<I>::value))>
    void pretrain_layer_batch(Iterator first, Iterator last, watcher_t& watcher, std::size_t max_epochs) {
        using layer_t = layer_type<I>;

        decltype(auto) rbm = layer_get<I>();

        watcher.template pretrain_layer<layer_t>(*this, I, 0);

        using rbm_trainer_t = dll::rbm_trainer<layer_t, !watcher_t::ignore_sub, dbn_detail::rbm_watcher_t<watcher_t>>;

        //Initialize the RBM trainer
        rbm_trainer_t r_trainer;

        //Init the RBM and training parameters
        r_trainer.init_training(rbm, first, last);

        //Get the specific trainer (CD)
        auto trainer = rbm_trainer_t::get_trainer(rbm);

        auto total_batch_size = big_batch_size * get_batch_size(rbm);

        std::vector<typename Iterator::value_type> input_cache(total_batch_size);

        auto next_input = layer_get<I - 1>().template prepare_output<layer_input_one_t<I - 1>>(total_batch_size);

        //Train for max_epochs epoch
        for (std::size_t epoch = 0; epoch < max_epochs; ++epoch) {
            std::size_t big_batch = 0;

            //Create a new context for this epoch
            rbm_training_context context;

            r_trainer.init_epoch();

            auto it  = first;
            auto end = last;

            while (it != end) {
                //Fill the input cache
                std::size_t i = 0;
                while (it != end && i < total_batch_size) {
                    input_cache[i++] = *it++;
                }

                multi_activation_probabilities<I>(input_cache.begin(), input_cache.begin() + i, next_input);

                if (big_batch_size == 1) {
                    //Train the RBM on this batch
                    r_trainer.train_batch(next_input.begin(), next_input.end(), next_input.begin(), next_input.end(), trainer, context, rbm);
                } else {
                    //Train the RBM on this big batch
                    r_trainer.train_sub(next_input.begin(), next_input.end(), next_input.begin(), trainer, context, rbm);
                }

                if (dbn_traits<this_type>::is_verbose()) {
                    watcher.pretraining_batch(*this, big_batch);
                }

                ++big_batch;
            }

            r_trainer.finalize_epoch(epoch, context, rbm);
        }

        r_trainer.finalize_training(rbm);

        //train the next layer, if any
        pretrain_layer_batch<I + 1>(first, last, watcher, max_epochs);
    }

    //Multiplex version
    template <std::size_t I, typename Iterator, cpp_enable_if((I > 0 && I < layers && dbn_traits<this_type>::is_multiplex() && !batch_layer_ignore<I>::value))>
    void pretrain_layer_batch(Iterator first, Iterator last, watcher_t& watcher, std::size_t max_epochs) {
        using layer_t = layer_type<I>;

        decltype(auto) rbm = layer_get<I>();

        watcher.template pretrain_layer<layer_t>(*this, I, 0);

        using rbm_trainer_t = dll::rbm_trainer<layer_t, !watcher_t::ignore_sub, dbn_detail::rbm_watcher_t<watcher_t>>;

        //Initialize the RBM trainer
        rbm_trainer_t r_trainer;

        //Init the RBM and training parameters
        r_trainer.init_training(rbm, first, last);

        //Get the specific trainer (CD)
        auto trainer = rbm_trainer_t::get_trainer(rbm);

        auto rbm_batch_size   = get_batch_size(rbm);
        auto total_batch_size = big_batch_size * rbm_batch_size;

        std::vector<std::vector<typename layer_type<I - 1>::output_deep_t>> input(total_batch_size);

        std::vector<typename layer_t::input_one_t> input_flat;

        //Train for max_epochs epoch
        for (std::size_t epoch = 0; epoch < max_epochs; ++epoch) {
            std::size_t big_batch = 0;

            //Create a new context for this epoch
            rbm_training_context context;

            r_trainer.init_epoch();

            auto it  = first;
            auto end = last;

            while (it != end) {
                auto batch_start = it;

                dbn_detail::safe_advance(it, end, total_batch_size);

                multi_activation_probabilities<I>(batch_start, it, input);

                flatten_in(input, input_flat);

                for (auto& i : input) {
                    i.clear();
                }

                //Compute the number of batches that have been gathered
                auto batches = input_flat.size() / rbm_batch_size;
                auto offset  = std::min(batches * rbm_batch_size, input_flat.size());

                if (batches <= 1) {
                    //Train the RBM on one batch
                    r_trainer.train_batch(
                        input_flat.begin(), input_flat.begin() + offset,
                        input_flat.begin(), input_flat.begin() + offset, trainer, context, rbm);
                } else if (batches > 1) {
                    //Train the RBM on this big batch
                    r_trainer.train_sub(input_flat.begin(), input_flat.begin() + offset, input_flat.begin(), trainer, context, rbm);
                }

                //Erase what we already passed to the trainer
                input_flat.erase(input_flat.begin(), input_flat.begin() + offset);

                if (dbn_traits<this_type>::is_verbose()) {
                    watcher.pretraining_batch(*this, big_batch);
                }

                ++big_batch;
            }

            r_trainer.finalize_epoch(epoch, context, rbm);
        }

        r_trainer.finalize_training(rbm);

        //train the next layer, if any
        pretrain_layer_batch<I + 1>(first, last, watcher, max_epochs);
    }

    //Stop template recursion
    template <std::size_t I, typename Iterator, cpp_enable_if(I == layers)>
    void pretrain_layer_batch(Iterator, Iterator, watcher_t&, std::size_t) {}

    /* Train with labels */

    template <std::size_t I, typename Iterator, typename LabelIterator>
    std::enable_if_t<(I < layers)> train_with_labels(Iterator first, Iterator last, watcher_t& watcher, LabelIterator lit, LabelIterator lend, std::size_t labels, std::size_t max_epochs) {
        using layer_t = layer_type<I>;

        decltype(auto) layer = layer_get<I>();

        auto input_size = std::distance(first, last);

        watcher.template pretrain_layer<layer_t>(*this, I, input_size);

        cpp::static_if<layer_traits<layer_t>::is_trained()>([&](auto f) {
            f(layer).template train<!watcher_t::ignore_sub,               //Enable the RBM Watcher or not
                                    dbn_detail::rbm_watcher_t<watcher_t>> //Replace the RBM watcher if not void
                (first, last, max_epochs);
        });

        if (I < layers - 1) {
            auto next_a = layer.template prepare_output<layer_input_one_t<I>>(input_size);

            cpp::foreach_i(first, last, [&](auto& sample, std::size_t i) {
                layer.activate_hidden(next_a[i], sample);
            });

            //If the next layer is the last layer
            if (I == layers - 2) {
                auto big_next_a = layer.template prepare_output<layer_input_one_t<I>>(input_size, true, labels);

                //Cannot use std copy since the sub elements have different size
                for (std::size_t i = 0; i < next_a.size(); ++i) {
                    for (std::size_t j = 0; j < next_a[i].size(); ++j) {
                        big_next_a[i][j] = next_a[i][j];
                    }
                }

                std::size_t i = 0;
                while (lit != lend) {
                    decltype(auto) label = *lit;

                    for (size_t l = 0; l < labels; ++l) {
                        big_next_a[i][dll::output_size(layer) + l] = label == l ? 1.0 : 0.0;
                    }

                    ++i;
                    ++lit;
                }

                train_with_labels<I + 1>(big_next_a.begin(), big_next_a.end(), watcher, lit, lend, labels, max_epochs);
            } else {
                train_with_labels<I + 1>(next_a.begin(), next_a.end(), watcher, lit, lend, labels, max_epochs);
            }
        }
    }

    template <std::size_t I, typename Iterator, typename LabelIterator>
    std::enable_if_t<(I == layers)> train_with_labels(Iterator, Iterator, watcher_t&, LabelIterator, LabelIterator, std::size_t, std::size_t) {}

    /* Predict with labels */

    /*!
     * \brief Predict the output labels (only when pretrain with labels)
     */
    template <std::size_t I>
    std::enable_if_t<(I < layers)> predict_labels(const input_t& input, label_output_t& output, std::size_t labels) const {
        decltype(auto) layer = layer_get<I>();

        auto next_a = layer.template prepare_one_output<layer_input_one_t<I>>();
        auto next_s = layer.template prepare_one_output<layer_input_one_t<I>>();

        layer.activate_hidden(next_a, next_s, input, input);

        if (I == layers - 1) {
            auto output_a = layer.prepare_one_input();
            auto output_s = layer.prepare_one_input();

            layer.activate_visible(next_a, next_s, output_a, output_s);

            output = std::move(output_a);
        } else {
            bool is_last = I == layers - 2;

            //If the next layers is the last layer
            if (is_last) {
                auto big_next_a = layer.template prepare_one_output<layer_input_one_t<I>>(is_last, labels);

                for (std::size_t i = 0; i < next_a.size(); ++i) {
                    big_next_a[i] = next_a[i];
                }

                std::fill(big_next_a.begin() + dll::output_size(layer), big_next_a.end(), 0.1);

                predict_labels<I + 1>(big_next_a, output, labels);
            } else {
                predict_labels<I + 1>(next_a, output, labels);
            }
        }
    }

    //Stop recursion
    template <std::size_t I>
    std::enable_if_t<(I == layers)> predict_labels(const input_t&, label_output_t&, std::size_t) const {}

    /* Activation Probabilities */

    template <std::size_t I, typename Iterator, typename Ouput>
    void multi_activation_probabilities(Iterator first, Iterator last, Ouput& output) {
        //Collect an entire batch
        maybe_parallel_foreach_i(pool, first, last, [this, &output](auto& v, std::size_t i) {
            this->activation_probabilities<0, I>(v, output[i]);
        });
    }

    template <std::size_t I, std::size_t S = layers, typename Input, typename Result>
    std::enable_if_t<(I < S)> activation_probabilities(const Input& input, Result& result) const {
        static constexpr const bool multi_layer = layer_traits<layer_type<I>>::is_multiplex_layer();

        auto& layer = layer_get<I>();

        cpp::static_if<(I < S - 1 && !multi_layer)>([&](auto f) {
            auto next_a = layer.template prepare_one_output<Input>();
            f(layer).activate_hidden(next_a, input);
            this->template activation_probabilities<I + 1, S>(next_a, result);
        });

        cpp::static_if<(I < S - 1 && multi_layer)>([&](auto f) {
            auto next_a = layer.template prepare_one_output<Input>();
            layer.activate_hidden(next_a, input);

            cpp_assert(f(result).empty(), "result must be empty on entry of activation_probabilities");

            f(result).reserve(next_a.size());

            for (std::size_t i = 0; i < next_a.size(); ++i) {
                f(result).push_back(this->template layer_get<S - 1>().template prepare_one_output<layer_input_one_t<I>>());
                this->template activation_probabilities<I + 1, S>(next_a[i], f(result)[i]);
            }
        });

        cpp::static_if<(I == S - 1)>([&](auto f) {
            f(layer).activate_hidden(result, input);
        });
    }

    //Stop template recursion
    template <std::size_t I, std::size_t S = layers, typename Input, typename Result>
    std::enable_if_t<(I == S)> activation_probabilities(const Input&, Result&) const {}

    template <std::size_t I, typename Input>
    std::enable_if_t<(I < layers)> full_activation_probabilities(const Input& input, std::size_t& i, full_output_t& result) const {
        auto& layer = layer_get<I>();

        auto next_a = layer.template prepare_one_output<Input>();

        layer.activate_hidden(next_a, input);

        for (auto& value : next_a) {
            result[i++] = value;
        }

        full_activation_probabilities<I + 1>(next_a, i, result);
    }

    //Stop template recursion
    template <std::size_t I, typename Input>
    std::enable_if_t<(I == layers)> full_activation_probabilities(const Input&, std::size_t&, full_output_t&) const {}

#ifdef DLL_SVM_SUPPORT

    template <typename DBN = this_type, cpp_enable_if(dbn_traits<DBN>::concatenate())>
    void add_activation_probabilities(svm_samples_t& result, const input_t& sample) {
        result.emplace_back(full_output_size());
        full_activation_probabilities(sample, result.back());
    }

    template <typename DBN = this_type, cpp_disable_if(dbn_traits<DBN>::concatenate())>
    void add_activation_probabilities(svm_samples_t& result, const input_t& sample) {
        result.push_back(layer_get<layers - 1>().template prepare_one_output<layer_input_one_t<layers - 1>>());
        activation_probabilities(sample, result.back());
    }

    template <typename Samples, typename Labels>
    void make_problem(const Samples& training_data, const Labels& labels, bool scale = false) {
        svm_samples_t svm_samples;

        //Get all the activation probabilities
        for (auto& sample : training_data) {
            add_activation_probabilities(svm_samples, sample);
        }

        //static_cast ensure using the correct overload
        problem = svm::make_problem(labels, static_cast<const svm_samples_t&>(svm_samples), scale);
    }

    /*!
     * \brief Create the svm problem for this dbn
     */
    template <typename Iterator, typename LIterator>
    void make_problem(Iterator first, Iterator last, LIterator&& lfirst, LIterator&& llast, bool scale = false) {
        svm_samples_t svm_samples;

        //Get all the activation probabilities
        std::for_each(first, last, [this, &svm_samples](auto& sample) {
            this->add_activation_probabilities(svm_samples, sample);
        });

        //static_cast ensure using the correct overload
        problem = svm::make_problem(
            std::forward<LIterator>(lfirst), std::forward<LIterator>(llast),
            svm_samples.begin(), svm_samples.end(),
            scale);
    }

#endif //DLL_SVM_SUPPORT
};

} //end of namespace dll
