//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
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
#include "cpp_utils/maybe_parallel.hpp"

#include "unit_type.hpp"
#include "trainer/dbn_trainer.hpp"
#include "trainer/rbm_trainer_fwd.hpp"
#include "dll/trainer/rbm_training_context.hpp"
#include "dbn_common.hpp"
#include "svm_common.hpp"
#include "util/flatten.hpp"
#include "util/converter.hpp" // Input type conversion
#include "util/export.hpp"
#include "util/timers.hpp"
#include "util/random.hpp"
#include "dbn_detail.hpp" // dbn_detail namespace

namespace dll {

template<typename O, typename Enable = void>
struct safe_value_type {
    using type = typename std::decay_t<O>::value_type;
};

template<typename O>
struct safe_value_type <O, std::enable_if_t<etl::is_etl_expr<O>::value>> {
    using type = etl::value_t<O>;
};

template<typename O>
using safe_value_t = typename safe_value_type<O>::type;

template<typename Layer>
struct is_input_layer {
    using traits = decay_layer_traits<Layer>;
    static constexpr const bool value = !traits::is_transform_layer() && !traits::is_augment_layer();
};

template<std::size_t Layer, typename DBN, typename Enable = void>
struct find_input_layer {
    static constexpr const std::size_t L = Layer;
};

template<std::size_t Layer, typename DBN>
struct find_input_layer<Layer, DBN, std::enable_if_t<!is_input_layer<typename DBN::template layer_type<Layer>>::value>> {
    static constexpr const std::size_t L = find_input_layer<Layer + 1, DBN>::L;
};

template<typename Layer>
struct is_output_layer {
    using traits = decay_layer_traits<Layer>;
    static constexpr const bool value = !traits::is_transform_layer();
};

template<std::size_t Layer, typename DBN, typename Enable = void>
struct find_output_layer {
    static constexpr const std::size_t L = Layer;
};

template<std::size_t Layer, typename DBN>
struct find_output_layer<Layer, DBN, std::enable_if_t<!is_output_layer<typename DBN::template layer_type<Layer>>::value>> {
    static constexpr const std::size_t L = find_output_layer<Layer - 1, DBN>::L;
};

/*!
 * \brief A Deep Belief Network implementation
 */
template <typename Desc>
struct dbn final {
    using desc      = Desc;      ///< The network descriptor
    using this_type = dbn<desc>; ///< The network type

    using layers_t = typename desc::layers; ///< The layers container type

    static_assert(!(dbn_traits<this_type>::batch_mode() && layers_t::has_shuffle_layer),
        "batch_mode dbn does not support shuffle in layers");
    static_assert(!dbn_traits<this_type>::shuffle_pretrain() || dbn_traits<this_type>::batch_mode(),
        "shuffle_pre is only compatible with batch mode, for normal mode, use shuffle in layers");

    template <std::size_t N>
    using layer_type = detail::layer_type_t<N, layers_t>; ///< The type of the layer at index Nth

    // The weight is is extracted from the first layer, since all layers have the same type
    using weight = typename dbn_detail::extract_weight_t<0, this_type>::type; ///< The type of the weights

    static_assert(dbn_detail::validate_weight_type<this_type, weight>::value, "Every layer must have consistent weight type");

    using watcher_t = typename desc::template watcher_t<this_type>; ///< The watcher type

    static constexpr const size_t input_layer_n  = find_input_layer<0, this_type>::L;  ///< The index of the input layer
    static constexpr const size_t output_layer_n = find_output_layer<layers_t::size - 1, this_type>::L; ///< The index of the output layer

    using input_layer_t = layer_type<input_layer_n>; ///< The type of the input layer

    using input_one_t = typename input_layer_t::input_one_t; ///< The type of one input
    using input_t     = typename input_layer_t::input_t;     ///< The type of a set of input

private:
    template <std::size_t I, typename Input>
    struct types_helper {
        using input_t = typename types_helper<I - 1, Input>::output_t;
        using output_t = std::decay_t<decltype(std::declval<layer_type<I>>().template prepare_one_output<input_t>())>;
    };

    template <typename Input>
    struct types_helper<0, Input> {
        using input_t = std::decay_t<Input>;
        using output_t = std::decay_t<decltype(std::declval<layer_type<0>>().template prepare_one_output<input_t>())>;
    };

public:
    using full_output_t = etl::dyn_vector<weight>; ///< The type of output for concatenated activation probabilities

    using for_each_impl_t      = dbn_detail::for_each_impl<this_type, layers_t::size, std::make_index_sequence<layers_t::size>>;
    using for_each_pair_impl_t = dbn_detail::for_each_impl<this_type, layers_t::size, std::make_index_sequence<layers_t::size - 1>>;

    using const_for_each_impl_t      = dbn_detail::for_each_impl<const this_type, layers_t::size, std::make_index_sequence<layers_t::size>>;
    using const_for_each_pair_impl_t = dbn_detail::for_each_impl<const this_type, layers_t::size, std::make_index_sequence<layers_t::size - 1>>;

    static constexpr const std::size_t layers         = layers_t::size;     ///< The number of layers
    static constexpr const std::size_t batch_size     = desc::BatchSize;    ///< The batch size (for finetuning)
    static constexpr const std::size_t big_batch_size = desc::BigBatchSize; ///< The number of pretraining batch to do at once

    layers_t tuples; ///< The layers

    weight learning_rate     = 0.1;  ///< The learning rate for finetuning
    weight lr_bold_inc       = 1.05; ///< The multiplicative increase of learning rate for the bold driver
    weight lr_bold_dec       = 0.5;  ///< The multiplicative decrease of learning rate for the bold driver
    weight lr_step_gamma     = 0.5;  ///< The multiplicative decrease of learning rate for the step driver
    std::size_t lr_step_size = 10;   ///< The number of steps after which the step driver decreases the learning rate

    weight initial_momentum     = 0.9; ///< The initial momentum
    weight final_momentum       = 0.9; ///< The final momentum applied after *final_momentum_epoch* epoch
    weight final_momentum_epoch = 6;   ///< The epoch at which momentum change

    weight l1_weight_cost = 0.0002; ///< The weight cost for L1 weight decay
    weight l2_weight_cost = 0.0002; ///< The weight cost for L2 weight decay

    weight momentum = 0; ///< The current momentum

    bool batch_mode_run = false;

    weight goal = 0.0; ///< The learning goal

#ifdef DLL_SVM_SUPPORT
    //TODO Ideally these fields should be private
    svm::model svm_model;    ///< The learned model
    svm::problem problem;    ///< libsvm is stupid, therefore, you cannot destroy the problem if you want to use the model...
    bool svm_loaded = false; ///< Indicates if a SVM model has been loaded (and therefore must be saved)
#endif                       //DLL_SVM_SUPPORT

private:
    cpp::thread_pool<!dbn_traits<this_type>::is_serial()> pool;

    mutable int fake_resource; ///< Simple field to get a reference from for resource management

    template<std::size_t I, cpp_disable_if(I == layers)>
    void dyn_init(){
        using fast_t = detail::layer_type_t<I, typename desc::base_layers>;

        decltype(auto) dyn_rbm = layer_get<I>();

        fast_t::dyn_init(dyn_rbm);

        dyn_init<I+1>();
    }

    template<std::size_t I, cpp_enable_if(I == layers)>
    void dyn_init(){}

public:
    /*!
     * Constructs a DBN and initializes all its members.
     *
     * This is the only way to create a DBN.
     */
    dbn() : pool(etl::threads) {
        //Nothing else to init

        cpp::static_if<!std::is_same<typename desc::base_layers, typename desc::layers>::value>([&](auto f){
            f(this)->template dyn_init<0>();
        });
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
            cpp::static_if<decay_layer_traits<decltype(layer)>::is_neural_layer()>([&](auto f) {
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

    /*!
     * \brief Initialize the Nth layer  with the given args. The Nth layer must
     * be a dynamic layer.
     * \tparam N The index of the layer to return (from 0)
     * \param args The arguments for initialization of the layer.
     */
    template <std::size_t N, typename... Args>
    void init_layer(Args&&... args){
        layer_get<N>().init_layer(std::forward<Args>(args)...);
    }

    template <std::size_t N>
    std::size_t layer_input_size() const noexcept {
        return dll::input_size(layer_get<N>());
    }

    template <std::size_t N>
    std::size_t layer_output_size() const noexcept {
        return dll::output_size(layer_get<N>());
    }

    std::size_t input_size() const noexcept {
        return dll::input_size(layer_get<input_layer_n>());
    }

    std::size_t output_size() const noexcept {
        return dll::output_size(layer_get<output_layer_n>());
    }

    std::size_t full_output_size() const noexcept {
        std::size_t output = 0;
        for_each_layer([&output](auto& layer) {
            output += layer.output_size();
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
    bool batch_mode() const noexcept {
        return dbn_traits<this_type>::batch_mode() || batch_mode_run;
    }

    /* pretrain */

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
    void pretrain(const input_t& training_data, std::size_t max_epochs) {
        pretrain(training_data.begin(), training_data.end(), max_epochs);
    }

    /*!
     * \brief Pretrain the network by training all layers in an unsupervised
     * manner.
     */
    template <typename Input>
    void pretrain(const Input& training_data, std::size_t max_epochs) {
        decltype(auto) converted = converter_many<Input, input_t>::convert(layer_get<input_layer_n>(), training_data);
        pretrain(converted.begin(), converted.end(), max_epochs);
    }

    /* pretrain_denoising */

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
    void pretrain_denoising(const input_t& noisy, const input_t& clean, std::size_t max_epochs) {
        pretrain_denoising(noisy.begin(), noisy.end(), clean.begin(), clean.end(), max_epochs);
    }

    /*!
     * \brief Pretrain the network by training all layers in an unsupervised
     * manner, the network will learn to reconstruct noisy input.
     */
    template <typename Noisy, typename Clean>
    void pretrain_denoising(const Noisy& noisy, const Clean& clean, std::size_t max_epochs) {
        decltype(auto) converted_noisy = converter_many<Noisy, input_t>::convert(layer_get<input_layer_n>(), noisy);
        decltype(auto) converted_clean = converter_many<Clean, input_t>::convert(layer_get<input_layer_n>(), clean);
        pretrain_denoising(converted_noisy.begin(), converted_noisy.end(), converted_clean.begin(), converted_clean.end(), max_epochs);
    }

    /* pretrain_denoising_auto */

    /*!
     * \brief Pretrain the network by training all layers in an unsupervised
     * manner, the network will learn to reconstruct noisy input.
     */
    template <typename CIterator>
    void pretrain_denoising_auto(CIterator cit, CIterator cend, std::size_t max_epochs, double noise) {
        dll::auto_timer timer("dbn:pretrain:denoising:auto");

        watcher_t watcher;

        watcher.pretraining_begin(*this, max_epochs);

        //Pretrain each layer one-by-one
        if (batch_mode()) {
            std::cout << "DBN: Denoising Pretraining done in batch mode" << std::endl;

            if (layers_t::has_shuffle_layer) {
                std::cout << "warning: batch_mode dbn does not support shuffle in layers (will be ignored)";
            }

            //Pretrain each layer one-by-one
            pretrain_layer_denoising_auto_batch<0>(cit, cend, watcher, max_epochs, noise);
        } else {
            std::cout << "DBN: Denoising Pretraining" << std::endl;

            //Pretrain each layer one-by-one
            pretrain_layer_denoising_auto<0>(cit, cend, watcher, max_epochs, noise, fake_resource);
        }

        watcher.pretraining_end(*this);
    }

    /*!
     * \brief Pretrain the network by training all layers in an unsupervised
     * manner, the network will learn to reconstruct noisy input.
     */
    void pretrain_denoising_auto(const input_t& clean, std::size_t max_epochs, double noise) {
        pretrain_denoising_auto(clean.begin(), clean.end(), max_epochs, noise);
    }

    /*!
     * \brief Pretrain the network by training all layers in an unsupervised
     * manner, the network will learn to reconstruct noisy input.
     */
    template <typename Clean>
    void pretrain_denoising_auto(const Clean& clean, std::size_t max_epochs, double noise) {
        decltype(auto) converted_clean = converter_many<Clean, input_t>::convert(layer_get<input_layer_n>(), clean);
        pretrain_denoising_auto(converted_clean.begin(), converted_clean.end(), max_epochs, noise);
    }

    /* train with labels */

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

    template<typename Input>
    size_t predict_labels(const Input& item, std::size_t labels) const {
        cpp_assert(dll::input_size(layer_get<layers - 1>()) == dll::output_size(layer_get<layers - 2>()) + labels, "There is no room for the labels units");

        auto output_a = layer_get<layers - 1>().prepare_one_input();

        predict_labels<0>(item, output_a, labels);

        return std::distance(
            std::prev(output_a.end(), labels),
            std::max_element(std::prev(output_a.end(), labels), output_a.end()));
    }

    //Note: features_sub are alias functions for activation_probabilities_sub

    /*!
     * \brief Returns the output features of the Ith layer for the given sample and saves them in the given container
     * \param sample The sample to get features from
     * \param result The container where the results will be saved
     * \tparam I The index of the layer for features (starts at 0)
     * \return the output features of the Ith layer of the network
     */
    template <std::size_t I, typename Input, typename Output, typename T = this_type>
    auto features_sub(const Input& sample, Output& result) const {
        return result = activation_probabilities_sub<I>(sample);
    }

    /*!
     * \brief Returns the output features of the Ith layer for the given sample
     * \param sample The sample to get features from
     * \tparam I The index of the layer for features (starts at 0)
     * \return the output features of the Ith layer of the network
     */
    template <std::size_t I, typename Input>
    auto features_sub(const Input& sample) const {
        return activation_probabilities_sub<I>(sample);
    }

    //Note: features are alias functions for activation_probabilities

    /*!
     * \brief Computes the output features for the given sample and saves them in the given container
     * \param sample The sample to get features from
     * \param result The container where to save the features
     * \return result
     */
    template <typename Output>
    auto features(const input_one_t& sample, Output& result) const {
        return activation_probabilities(sample, result);
    }

    /*!
     * \brief Returns the output features for the given sample
     * \param sample The sample to get features from
     * \return the output features of the last layer of the network
     */
    auto features(const input_one_t& sample) const {
        return activation_probabilities(sample);
    }

    /*!
     * \brief Returns the output features for the given sample
     * \param sample The sample to get features from
     * \return the output features of the last layer of the network
     */
    template<typename Input>
    auto features(const Input& sample) const {
        decltype(auto) converted = converter_one<Input, input_one_t>::convert(layer_get<input_layer_n>(), sample);
        return activation_probabilities(converted);
    }

    // Forward one batch at a time

    //Note: Ideally, this should be done without using the SGD
    //context, but this would mean a complete overhaul of creation
    //of batches...

    // TODO: Transform layers should be applied inline

    template <size_t L, typename Input, cpp_enable_if((L == 0 && L != layers - 1))>
    decltype(auto) forward_batch_impl(Input&& sample) {
        decltype(auto) layer = layer_get<L>();
        decltype(auto) context = layer.template get_sgd_context<this_type>();

        context.input = sample;
        layer.batch_activate_hidden(context.output, context.input);

        return forward_batch_impl<L+1>(context.output);
    }

    template <size_t L, typename Input, cpp_enable_if((L != 0 && L != layers - 1))>
    decltype(auto) forward_batch_impl(Input&& sample) {
        decltype(auto) layer = layer_get<L>();
        decltype(auto) context = layer.template get_sgd_context<this_type>();

        layer.batch_activate_hidden(context.output, sample);

        return forward_batch_impl<L+1>(context.output);
    }

    template <size_t L, typename Input, cpp_enable_if((L == layers - 1))>
    decltype(auto) forward_batch_impl(Input&& sample) {
        decltype(auto) layer = layer_get<L>();
        decltype(auto) context = layer.template get_sgd_context<this_type>();

        layer.batch_activate_hidden(context.output, sample);

        return context.output;
    }

    template <typename Input>
    decltype(auto) forward_batch(Input&& sample) {
        return forward_batch_impl<0>(sample);
    }

    // Forward one sample at a time
    // This is not as fast as it could be, far from it, but supports
    // larger range of input. The rationale being that time should
    // be spent in forward_batch

    // TODO: Transform layers should be applied inline

    template <size_t L, typename Input, cpp_enable_if((L == 0 && L != layers - 1))>
    decltype(auto) forward_impl(Input&& sample) {
        decltype(auto) layer = layer_get<L>();
        decltype(auto) context = layer.template get_sgd_context<this_type>();

        auto input = force_temporary(context.input(0));
        auto output = force_temporary(context.output(0));

        input = sample;
        layer.activate_hidden(output, input);

        return forward_impl<L+1>(output);
    }

    template <size_t L, typename Input, cpp_enable_if((L != 0 && L != layers - 1))>
    decltype(auto) forward_impl(Input&& sample) {
        decltype(auto) layer = layer_get<L>();
        decltype(auto) context = layer.template get_sgd_context<this_type>();

        auto output = force_temporary(context.output(0));

        layer.activate_hidden(output, sample);

        return forward_impl<L+1>(output);
    }

    template <size_t L, typename Input, cpp_enable_if((L == layers - 1))>
    decltype(auto) forward_impl(Input&& sample) {
        decltype(auto) layer = layer_get<L>();
        decltype(auto) context = layer.template get_sgd_context<this_type>();

        auto output = force_temporary(context.output(0));
        layer.activate_hidden(output, sample);

        return output;
    }

    template <typename Input>
    decltype(auto) forward(Input&& sample) {
        return forward_impl<0>(sample);
    }

    /*!
     * \brief Save the features generated for the given sample in the given file.
     * \param sample The sample to get features from
     * \param file The output file
     * \param f The format of the exported features
     */
    void save_features(const input_one_t& sample, const std::string& file, format f = format::DLL) const {
        cpp_assert(f == format::DLL, "Only DLL format is supported for now");

        decltype(auto) probs = features(sample);

        if (f == format::DLL) {
            export_features_dll(probs, file);
        }
    }

    /*!
     * \brief Save the features generated for the given sample in the given file.
     * \param sample The sample to get features from
     * \param file The output file
     * \param f The format of the exported features
     */
    template<typename Input>
    void save_features(const Input& sample, const std::string& file, format f = format::DLL) const {
        decltype(auto) converted = converter_one<Input, input_one_t>::convert(layer_get<input_layer_n>(), sample);
        save_features(converted, file, f);
    }

    template <typename Output>
    size_t predict_label(const Output& result) const {
        return std::distance(result.begin(), std::max_element(result.begin(), result.end()));
    }

    template <typename Input>
    size_t predict(const Input& item) const {
        auto result = activation_probabilities(item);
        return predict_label(result);
    }

    /*!
     * \brief Create a trainer for custom training of the network
     * \return The trainer for this network
     */
    dll::dbn_trainer<this_type> get_trainer() {
        dll::dbn_trainer<this_type> trainer;
        return trainer;
    }

    /*!
     * \brief Fine tune the network for classifcation.
     * \param training_data A container containing all the samples
     * \param labels A container containing all the labels
     * \param max_epochs The maximum number of epochs to train the network for.
     * \return The final classification error
     */
    template <typename Labels>
    weight fine_tune(const input_t& training_data, Labels& labels, size_t max_epochs) {
        return fine_tune(training_data.begin(), training_data.end(), labels.begin(), labels.end(), max_epochs);
    }

    /*!
     * \brief Fine tune the network for classifcation.
     * \param training_data A container containing all the samples
     * \param labels A container containing all the labels
     * \param max_epochs The maximum number of epochs to train the network for.
     * \return The final classification error
     */
    template <typename Input, typename Labels>
    weight fine_tune(const Input& training_data, Labels& labels, size_t max_epochs) {
        decltype(auto) converted = converter_many<Input, input_t>::convert(layer_get<input_layer_n>(), training_data);
        return fine_tune(converted.begin(), converted.end(), labels.begin(), labels.end(), max_epochs);
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

    /*!
     * \brief Fine tune the network for autoencoder.
     * \param training_data A container containing all the samples
     * \param max_epochs The maximum number of epochs to train the network for.
     * \return The final classification error
     */
    template <typename Samples>
    weight fine_tune_dae(const Samples& training_data, size_t max_epochs, double corrupt) {
        return fine_tune_dae(
            training_data.begin(), training_data.end(),
            max_epochs, corrupt);
    }

    /*!
     * \brief Fine tune the network for autoencoder.
     * \param first Iterator to the first sample
     * \param last Iterator to the last sample
     * \param max_epochs The maximum number of epochs to train the network for.
     * \return The final classification error
     */
    template <typename Iterator>
    weight fine_tune_dae(Iterator&& first, Iterator&& last, size_t max_epochs, double corrupt) {
        dll::auto_timer timer("dbn:train:ft:ae");

        cpp_assert(dll::input_size(layer_get<0>()) == dll::output_size(layer_get<layers - 1>()), "The network is not build as an autoencoder");

        cpp_assert(!batch_mode_run, "DAE does not support batch mode for now");

        dll::dbn_trainer<this_type> trainer;
        return trainer.train_dae(*this,
                                first, last,
                                max_epochs, corrupt);
    }

    template <std::size_t I, typename Input>
    auto prepare_output() const {
        using layer_input_t = typename types_helper<I, Input>::input_t;
        return layer_get<I>().template prepare_one_output<layer_input_t>();
    }

    template <typename Input>
    auto prepare_one_output() const {
        return prepare_output<layers - 1, Input>();
    }

private:
    template<typename T>
    using is_multi_t = etl::matrix_detail::is_vector<std::decay_t<T>>;

    template <typename Output, cpp_enable_if(is_multi_t<Output>::value && is_multi_t<safe_value_t<Output>>::value)>
    safe_value_t<Output> flatten(const Output& output) const {
        safe_value_t<Output> flat;
        flat.reserve(output.size() * output[0].size());
        for(auto& sub : output){
            std::move(sub.begin(), sub.end(), std::back_inserter(flat));
        }
        return flat;
    }

    template <typename Output, cpp_enable_if(!(is_multi_t<Output>::value && is_multi_t<safe_value_t<Output>>::value))>
    Output&& flatten(Output&& output) const {
        return std::forward<Output>(output);
    }

    template <bool Train, std::size_t I, std::size_t S, typename Input, cpp_enable_if(!is_multi_t<Input>::value && I == S)>
    auto activation_probabilities_impl(const Input& input) const {
        decltype(auto) layer = layer_get<I>();
        auto output = layer.template select_prepare_one_output<Train, Input>();
        layer.template select_activate_hidden<Train>(output, input);
        return flatten(output);
    }

    template <bool Train, std::size_t I, std::size_t S, typename Input, cpp_enable_if(is_multi_t<Input>::value && I == S)>
    auto activation_probabilities_impl(const Input& input) const {
        auto n_inputs = input.size();
        decltype(auto) layer = layer_get<I>();
        auto output = layer.template select_prepare_output<Train, safe_value_t<Input>>(n_inputs);
        for(std::size_t i = 0; i < n_inputs; ++i){
            layer.template select_activate_hidden<Train>(output[i], input[i]);
        }
        return flatten(output);
    }

    template <bool Train, std::size_t I, std::size_t S, typename Input, cpp_enable_if(I != S)>
    auto activation_probabilities_impl(const Input& input) const {
        decltype(auto) previous_output = activation_probabilities_impl<Train, I, I>(input);
        return activation_probabilities_impl<Train, I+1, S>(previous_output);
    }

public:
    template <std::size_t I>
    auto train_activation_probabilities_sub(const input_one_t& input) const {
        return activation_probabilities_impl<true, 0, I>(input);
    }

    template <std::size_t I, typename Input>
    auto train_activation_probabilities_sub(const Input& input) const {
        decltype(auto) converted = converter_one<Input, input_one_t>::convert(layer_get<input_layer_n>(), input);
        return activation_probabilities_impl<true, 0, I>(converted);
    }

    template <std::size_t I>
    auto test_activation_probabilities_sub(const input_one_t& input) const {
        return activation_probabilities_impl<false, 0, I>(input);
    }

    template <std::size_t I, typename Input>
    auto test_activation_probabilities_sub(const Input& input) const {
        decltype(auto) converted = converter_one<Input, input_one_t>::convert(layer_get<input_layer_n>(), input);
        return activation_probabilities_impl<false, 0, I>(converted);
    }

    template <std::size_t I>
    auto activation_probabilities_sub(const input_one_t& input) const {
        return train_activation_probabilities_sub<I>(input);
    }

    template <std::size_t I, typename Input>
    auto activation_probabilities_sub(const Input& input) const {
        decltype(auto) converted = converter_one<Input, input_one_t>::convert(layer_get<input_layer_n>(), input);
        return train_activation_probabilities_sub<I>(converted);
    }

    auto train_activation_probabilities(const input_one_t& input) const {
        return activation_probabilities_impl<true, 0, layers - 1>(input);
    }

    template <typename Input>
    auto train_activation_probabilities(const Input& input) const {
        decltype(auto) converted = converter_one<Input, input_one_t>::convert(layer_get<input_layer_n>(), input);
        return activation_probabilities_impl<true, 0, layers - 1>(converted);
    }

    auto test_activation_probabilities(const input_one_t& input) const {
        return activation_probabilities_impl<false, 0, layers - 1>(input);
    }

    template <typename Input>
    auto test_activation_probabilities(const Input& input) const {
        decltype(auto) converted = converter_one<Input, input_one_t>::convert(layer_get<input_layer_n>(), input);
        return activation_probabilities_impl<false, 0, layers - 1>(converted);
    }

    auto activation_probabilities(const input_one_t& input) const {
        return train_activation_probabilities(input);
    }

    template <typename Input>
    auto activation_probabilities(const Input& input) const {
        decltype(auto) converted = converter_one<Input, input_one_t>::convert(layer_get<input_layer_n>(), input);
        return train_activation_probabilities(converted);
    }

    template <std::size_t I, std::size_t S, typename Input, cpp_enable_if(I != S)>
    void full_activation_probabilities(const Input& input, full_output_t& result, std::size_t& i) const {
        auto output = activation_probabilities_impl<false, I, I>(input);
        for(auto& feature : output){
            result[i++] = feature;
        }
        full_activation_probabilities<I+1, S>(output, result, i);
    }

    template <std::size_t I, std::size_t S, typename Input, cpp_enable_if(I == S)>
    void full_activation_probabilities(const Input& input, full_output_t& result, std::size_t& i) const {
        auto output = activation_probabilities_impl<false, I, I>(input);
        for(auto& feature : output){
            result[i++] = feature;
        }
    }

    template <typename Input>
    void full_activation_probabilities(const Input& input, full_output_t& result) const {
        static_assert(!dbn_traits<this_type>::is_multiplex(), "Multiplex DBN does not support full_activation_probabilities");

        std::size_t i = 0;
        full_activation_probabilities<0, layers - 1>(input, result, i);
    }

    auto full_activation_probabilities(const input_one_t& input) const {
        static_assert(!dbn_traits<this_type>::is_multiplex(), "Multiplex DBN does not support full_activation_probabilities");

        full_output_t result(full_output_size());
        full_activation_probabilities(input, result);
        return result;
    }

    template <typename Input>
    auto full_activation_probabilities(const Input& input) const {
        decltype(auto) converted = converter_one<Input, input_one_t>::convert(layer_get<input_layer_n>(), input);
        return full_activation_probabilities(converted);
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

private:
    template <typename DBN = this_type, cpp_enable_if(dbn_traits<DBN>::concatenate())>
    auto get_final_activation_probabilities(const input_one_t& sample) const {
        return full_activation_probabilities(sample);
    }

    template <typename Input, typename DBN = this_type, cpp_enable_if(dbn_traits<DBN>::concatenate())>
    auto get_final_activation_probabilities(const Input& sample) const {
        decltype(auto) converted = converter_one<Input, input_one_t>::convert(layer_get<input_layer_n>(), sample);
        return full_activation_probabilities(converted);
    }

    template <typename DBN = this_type, cpp_disable_if(dbn_traits<DBN>::concatenate())>
    auto get_final_activation_probabilities(const input_one_t& sample) const {
        return activation_probabilities(sample);
    }

    template <typename Input, typename DBN = this_type, cpp_disable_if(dbn_traits<DBN>::concatenate())>
    auto get_final_activation_probabilities(const Input& sample) const {
        decltype(auto) converted = converter_one<Input, input_one_t>::convert(layer_get<input_layer_n>(), sample);
        return activation_probabilities(converted);
    }

public:
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

    template <typename Input>
    double svm_predict(const Input& sample) {
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

        using input_t = safe_value_t<Iterator>;
        using next_input_t = std::decay_t<decltype(layer.template prepare_one_output<input_t>())>;

        watcher.pretrain_layer(*this, I+1, next_layer, dbn_detail::fast_distance(first, last));

        auto next_a = next_layer.template prepare_output<next_input_t>(std::distance(first, last));

        maybe_parallel_foreach_i(pool, first, last, [&layer, &next_layer, &next_a](auto& v, std::size_t i) {
            auto tmp = layer.template prepare_one_output<input_t>();

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

        watcher.pretrain_layer(*this, I, layer, dbn_detail::fast_distance(first, last));

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
            auto next_a = layer.template prepare_output<safe_value_t<Iterator>>(std::distance(first, last));

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

        watcher.pretrain_layer(*this, I, layer, dbn_detail::fast_distance(nit, nend));

        cpp::static_if<layer_traits<layer_t>::is_pretrained()>([&](auto f) {
            f(layer).template train_denoising<NIterator, CIterator,
                                              !watcher_t::ignore_sub,               //Enable the RBM Watcher or not
                                              dbn_detail::rbm_watcher_t<watcher_t>> //Replace the RBM watcher if not void
                (nit, nend, cit, cend, max_epochs);
        });

        if (train_next<I + 1>::value) {
            auto next_n = layer.template prepare_output<safe_value_t<NIterator>>(std::distance(nit, nend));
            auto next_c = layer.template prepare_output<safe_value_t<CIterator>>(std::distance(nit, nend));

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

    /* Pretrain with denoising (auto) */

    template <std::size_t I, typename CIterator, typename CContainer, cpp_enable_if((I < layers))>
    void pretrain_layer_denoising_auto(CIterator cit, CIterator cend, watcher_t& watcher, std::size_t max_epochs, double noise, CContainer& previous_c) {
        using layer_t = layer_type<I>;

        decltype(auto) layer = layer_get<I>();

        watcher.pretrain_layer(*this, I, layer, dbn_detail::fast_distance(cit, cend));

        cpp::static_if<layer_traits<layer_t>::is_pretrained()>([&](auto f) {
            f(layer).template train_denoising_auto<CIterator,
                                              !watcher_t::ignore_sub,               //Enable the RBM Watcher or not
                                              dbn_detail::rbm_watcher_t<watcher_t>> //Replace the RBM watcher if not void
                (cit, cend, max_epochs, noise);
        });

        if (train_next<I + 1>::value) {
            auto next_c = layer.template prepare_output<safe_value_t<CIterator>>(std::distance(cit, cend));

            maybe_parallel_foreach_i(pool, cit, cend, [&layer, &next_c](auto& v, std::size_t i) {
                layer.activate_hidden(next_c[i], v);
            });

            //At this point we don't need the storage of the previous layer
            release(previous_c);

            //In the standard case, pass the output to the next layer
            pretrain_layer_denoising_auto<I + 1>(next_c.begin(), next_c.end(), watcher, max_epochs, noise, next_c);

            static_assert(!layer_traits<layer_t>::is_multiplex_layer(), "Denoising pretraining does not support multiplex layer");
        }
    }

    //Stop template recursion
    template <std::size_t I, typename CIterator, typename CContainer, cpp_enable_if((I == layers))>
    void pretrain_layer_denoising_auto(CIterator, CIterator, watcher_t&, std::size_t, double, CContainer&) {}

    /* Pretrain in batch mode */

    //By default no layer is ignored
    template <std::size_t I, class Enable = void>
    struct batch_layer_ignore : std::false_type {};

    //Transform and pooling layers can safely be skipped
    template <std::size_t I>
    struct batch_layer_ignore<I, std::enable_if_t<(I < layers)>> : cpp::or_u<layer_traits<layer_type<I>>::is_pooling_layer(), layer_traits<layer_type<I>>::is_transform_layer(), layer_traits<layer_type<I>>::is_standard_layer(), !layer_traits<layer_type<I>>::pretrain_last()> {};

    template <bool Bypass, typename Iterator, typename Container, typename T = this_type, cpp_enable_if(Bypass && dbn_traits<T>::shuffle_pretrain())>
    auto prepare_it(Iterator it, Iterator end, Container& container){
        std::copy(it, end, std::back_inserter(container));
        return std::make_tuple(container.begin(), container.end());
    }

    template <bool Bypass, typename Iterator, typename Container, typename T = this_type, cpp_disable_if(Bypass && dbn_traits<T>::shuffle_pretrain())>
    auto prepare_it(Iterator it, Iterator end, Container& container){
        cpp_unused(container);
        return std::make_tuple(it, end);
    }

    template <typename Container>
    void shuffle(Container& container){
        decltype(auto) g = dll::rand_engine();
        std::shuffle(container.begin(), container.end(), g);
    }

    //Special handling for the layer 0
    //data is coming from iterators not from input
    template <std::size_t I, typename Iterator, cpp_enable_if((I == 0 && !batch_layer_ignore<I>::value))>
    void pretrain_layer_batch(Iterator orig_first, Iterator orig_last, watcher_t& watcher, std::size_t max_epochs) {
        std::vector<std::remove_cv_t<typename std::iterator_traits<Iterator>::value_type>> input_copy;

        auto iterators = prepare_it<false>(orig_first, orig_last, input_copy);

        decltype(auto) first = std::get<0>(iterators);
        decltype(auto) last = std::get<1>(iterators);

        using layer_t = layer_type<I>;

        decltype(auto) rbm = layer_get<I>();

        watcher.pretrain_layer(*this, I, rbm, 0);

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

            // Sort before training
            shuffle(input_copy);

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
    void pretrain_layer_batch(Iterator orig_first, Iterator orig_last, watcher_t& watcher, std::size_t max_epochs) {
        std::vector<typename std::iterator_traits<Iterator>::value_type> input_copy;

        auto iterators = prepare_it<!batch_layer_ignore<0>::value>(orig_first, orig_last, input_copy);

        decltype(auto) first = std::get<0>(iterators);
        decltype(auto) last = std::get<1>(iterators);

        using layer_t = layer_type<I>;

        decltype(auto) rbm = layer_get<I>();

        watcher.pretrain_layer(*this, I, rbm, 0);

        using rbm_trainer_t = dll::rbm_trainer<layer_t, !watcher_t::ignore_sub, dbn_detail::rbm_watcher_t<watcher_t>>;

        //Initialize the RBM trainer
        rbm_trainer_t r_trainer;

        //Init the RBM and training parameters
        r_trainer.init_training(rbm, first, last);

        //Get the specific trainer (CD)
        auto trainer = rbm_trainer_t::get_trainer(rbm);

        auto total_batch_size = big_batch_size * get_batch_size(rbm);

        std::vector<safe_value_t<Iterator>> input_cache(total_batch_size);

        using input_t = typename types_helper<I - 1, safe_value_t<Iterator>>::input_t;
        auto next_input = layer_get<I - 1>().template prepare_output<input_t>(total_batch_size);

        //Train for max_epochs epoch
        for (std::size_t epoch = 0; epoch < max_epochs; ++epoch) {
            std::size_t big_batch = 0;

            // Sort before training
            shuffle(input_copy);

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

                multi_activation_probabilities<I - 1>(input_cache.begin(), input_cache.begin() + i, next_input);

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

    // TODO THis should not be necessary at all

    template <std::size_t I, typename Input, typename Enable = void>
    struct output_deep_t;

    template <std::size_t I, typename Input>
    struct output_deep_t<I, Input, std::enable_if_t<layer_traits<layer_type<I>>::is_multiplex_layer()>> {
        using type = safe_value_t<typename types_helper<I, Input>::output_t>;
    };

    template <std::size_t I, typename Input>
    struct output_deep_t<I, Input, std::enable_if_t<!layer_traits<layer_type<I>>::is_multiplex_layer()>> {
        using type = typename types_helper<I, Input>::output_t;
    };

    //Multiplex version
    template <std::size_t I, typename Iterator, cpp_enable_if((I > 0 && I < layers && dbn_traits<this_type>::is_multiplex() && !batch_layer_ignore<I>::value))>
    void pretrain_layer_batch(Iterator orig_first, Iterator orig_last, watcher_t& watcher, std::size_t max_epochs) {
        std::vector<typename std::iterator_traits<Iterator>::value_type> input_copy;

        auto iterators = prepare_it<!batch_layer_ignore<0>::value>(orig_first, orig_last, input_copy);

        decltype(auto) first = std::get<0>(iterators);
        decltype(auto) last = std::get<1>(iterators);

        using layer_t = layer_type<I>;

        decltype(auto) rbm = layer_get<I>();

        watcher.pretrain_layer(*this, I, rbm, 0);

        using rbm_trainer_t = dll::rbm_trainer<layer_t, !watcher_t::ignore_sub, dbn_detail::rbm_watcher_t<watcher_t>>;

        //Initialize the RBM trainer
        rbm_trainer_t r_trainer;

        //Init the RBM and training parameters
        r_trainer.init_training(rbm, first, last);

        //Get the specific trainer (CD)
        auto trainer = rbm_trainer_t::get_trainer(rbm);

        auto rbm_batch_size   = get_batch_size(rbm);
        auto total_batch_size = big_batch_size * rbm_batch_size;

        std::vector<std::vector<typename output_deep_t<I - 1, decltype(*first)>::type>> input(total_batch_size);

        std::vector<typename layer_t::input_one_t> input_flat;

        //Train for max_epochs epoch
        for (std::size_t epoch = 0; epoch < max_epochs; ++epoch) {
            std::size_t big_batch = 0;

            // Sort before training
            shuffle(input_copy);

            //Create a new context for this epoch
            rbm_training_context context;

            r_trainer.init_epoch();

            auto it  = first;
            auto end = last;

            while (it != end) {
                auto batch_start = it;

                dbn_detail::safe_advance(it, end, total_batch_size);

                multi_activation_probabilities<I - 1>(batch_start, it, input);

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

    /* pretrain_layer_denoising_auto_batch */

    template<typename T>
    inline void noise_transform(T&& value, double noise){
        decltype(auto) g = dll::rand_engine();

        std::uniform_real_distribution<double> dist(0.0, 1000.0);

        for(auto& v :  value){
            v *= dist(g) < noise * 1000.0 ? 0.0 : 1.0;
        }
    }

    //Special handling for the layer 0
    //data is coming from iterators not from input
    template <std::size_t I, typename Iterator, cpp_enable_if((I == 0 && !batch_layer_ignore<I>::value))>
    void pretrain_layer_denoising_auto_batch(Iterator orig_first, Iterator orig_last, watcher_t& watcher, std::size_t max_epochs, double noise) {
        std::vector<std::remove_cv_t<typename std::iterator_traits<Iterator>::value_type>> input_copy;

        auto iterators = prepare_it<false>(orig_first, orig_last, input_copy);

        decltype(auto) first = std::get<0>(iterators);
        decltype(auto) last = std::get<1>(iterators);

        using layer_t = layer_type<I>;

        decltype(auto) rbm = layer_get<I>();

        watcher.pretrain_layer(*this, I, rbm, 0);

        using rbm_trainer_t = dll::rbm_trainer<layer_t, !watcher_t::ignore_sub, dbn_detail::rbm_watcher_t<watcher_t>, false>;

        //Initialize the RBM trainer
        rbm_trainer_t r_trainer;

        //Init the RBM and training parameters
        r_trainer.init_training(rbm, first, last); //TODO This may be highly slow...

        //Get the specific trainer (CD)
        auto trainer = rbm_trainer_t::get_trainer(rbm);

        //Several RBM batches are propagated at once
        const auto total_batch_size = big_batch_size * get_batch_size(rbm);

        std::vector<typename layer_t::input_one_t> clean_cache(total_batch_size);
        std::vector<typename layer_t::input_one_t> noisy_cache(total_batch_size);

        //Train for max_epochs epoch
        for (std::size_t epoch = 0; epoch < max_epochs; ++epoch) {
            std::size_t big_batch = 0;

            // Sort before training
            shuffle(input_copy);

            //Create a new context for this epoch
            rbm_training_context context;

            r_trainer.init_epoch();

            auto it  = first;
            auto end = last;

            while (it != end) {
                //Fill the input cache
                std::size_t i = 0;
                while (it != end && i < total_batch_size) {
                    auto index = i++;
                    clean_cache[index] = *it++;
                    noisy_cache[index] = clean_cache[index];
                    noise_transform(noisy_cache[index], noise);
                }

                if (big_batch_size == 1) {
                    //Train the RBM on this batch
                    r_trainer.train_batch(noisy_cache.begin(), noisy_cache.end(), clean_cache.begin(), clean_cache.end(), trainer, context, rbm);
                } else {
                    //Train the RBM on this big batch
                    r_trainer.train_sub(noisy_cache.begin(), noisy_cache.begin() + i, clean_cache.begin(), trainer, context, rbm);
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
        pretrain_layer_denoising_auto_batch<I + 1>(first, last, watcher, max_epochs, noise);
    }

    //Special handling for untrained layers
    template <std::size_t I, typename Iterator, cpp_enable_if(batch_layer_ignore<I>::value)>
    void pretrain_layer_denoising_auto_batch(Iterator first, Iterator last, watcher_t& watcher, std::size_t max_epochs, double noise) {
        //We simply go up one layer on pooling layers
        pretrain_layer_denoising_auto_batch<I + 1>(first, last, watcher, max_epochs, noise);
    }

    //Normal version
    template <std::size_t I, typename Iterator, cpp_enable_if((I > 0 && I < layers && !batch_layer_ignore<I>::value))>
    void pretrain_layer_denoising_auto_batch(Iterator orig_first, Iterator orig_last, watcher_t& watcher, std::size_t max_epochs, double noise) {
        std::vector<typename std::iterator_traits<Iterator>::value_type> input_copy;

        auto iterators = prepare_it<!batch_layer_ignore<0>::value>(orig_first, orig_last, input_copy);

        decltype(auto) first = std::get<0>(iterators);
        decltype(auto) last = std::get<1>(iterators);

        using layer_t = layer_type<I>;

        decltype(auto) rbm = layer_get<I>();

        watcher.pretrain_layer(*this, I, rbm, 0);

        using rbm_trainer_t = dll::rbm_trainer<layer_t, !watcher_t::ignore_sub, dbn_detail::rbm_watcher_t<watcher_t>>;

        //Initialize the RBM trainer
        rbm_trainer_t r_trainer;

        //Init the RBM and training parameters
        r_trainer.init_training(rbm, first, last);

        //Get the specific trainer (CD)
        auto trainer = rbm_trainer_t::get_trainer(rbm);

        auto total_batch_size = big_batch_size * get_batch_size(rbm);

        std::vector<safe_value_t<Iterator>> clean_cache(total_batch_size);

        using input_t = typename types_helper<I - 1, safe_value_t<Iterator>>::input_t;
        auto next_clean = layer_get<I - 1>().template prepare_output<input_t>(total_batch_size);
        auto next_noisy = layer_get<I - 1>().template prepare_output<input_t>(total_batch_size);

        //Train for max_epochs epoch
        for (std::size_t epoch = 0; epoch < max_epochs; ++epoch) {
            std::size_t big_batch = 0;

            // Sort before training
            shuffle(input_copy);

            //Create a new context for this epoch
            rbm_training_context context;

            r_trainer.init_epoch();

            auto it  = first;
            auto end = last;

            while (it != end) {
                //Fill the input cache
                std::size_t i = 0;
                while (it != end && i < total_batch_size) {
                    auto index = i++;
                    clean_cache[index] = *it++;
                }

                multi_activation_probabilities<I - 1>(clean_cache.begin(), clean_cache.begin() + i, next_clean);

                next_noisy = next_clean;
                for(auto& noisy : next_noisy){
                    noise_transform(noisy, noise);
                }

                if (big_batch_size == 1) {
                    //Train the RBM on this batch
                    r_trainer.train_batch(next_noisy.begin(), next_noisy.end(), next_clean.begin(), next_clean.end(), trainer, context, rbm);
                } else {
                    //Train the RBM on this big batch
                    r_trainer.train_sub(next_noisy.begin(), next_noisy.end(), next_clean.begin(), trainer, context, rbm);
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
        pretrain_layer_denoising_auto_batch<I + 1>(first, last, watcher, max_epochs, noise);
    }

    //Stop template recursion
    template <std::size_t I, typename Iterator, cpp_enable_if(I == layers)>
    void pretrain_layer_denoising_auto_batch(Iterator, Iterator, watcher_t&, std::size_t, double) {}

    /* Train with labels */

    template <std::size_t I, typename Iterator, typename LabelIterator>
    std::enable_if_t<(I < layers)> train_with_labels(Iterator first, Iterator last, watcher_t& watcher, LabelIterator lit, LabelIterator lend, std::size_t labels, std::size_t max_epochs) {
        using layer_t = layer_type<I>;

        decltype(auto) layer = layer_get<I>();

        auto input_size = std::distance(first, last);

        watcher.pretrain_layer(*this, I, layer, input_size);

        cpp::static_if<layer_traits<layer_t>::is_trained()>([&](auto f) {
            f(layer).template train<!watcher_t::ignore_sub,               //Enable the RBM Watcher or not
                                    dbn_detail::rbm_watcher_t<watcher_t>> //Replace the RBM watcher if not void
                (first, last, max_epochs);
        });

        if (I < layers - 1) {
            using input_t = std::decay_t<decltype(*first)>;
            auto next_a = layer.template prepare_output<input_t>(input_size);

            cpp::foreach_i(first, last, [&](auto& sample, std::size_t i) {
                layer.activate_hidden(next_a[i], sample);
            });

            //If the next layer is the last layer
            if (I == layers - 2) {
                auto big_next_a = layer.template prepare_output<input_t>(input_size, true, labels);

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
    template <std::size_t I, typename Input, typename Output>
    std::enable_if_t<(I < layers)> predict_labels(const Input& input, Output& output, std::size_t labels) const {
        decltype(auto) layer = layer_get<I>();

        auto next_a = layer.template prepare_one_output<Input>();
        auto next_s = layer.template prepare_one_output<Input>();

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
                auto big_next_a = layer.template prepare_one_output<Input>(is_last, labels);

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
    template <std::size_t I, typename Input, typename Output>
    std::enable_if_t<(I == layers)> predict_labels(const Input&, Output&, std::size_t) const {}

    /* Activation Probabilities */

    template <std::size_t I, typename Iterator, typename Output>
    void multi_activation_probabilities(Iterator first, Iterator last, Output& output) {
        //Collect an entire batch
        maybe_parallel_foreach_i(pool, first, last, [this, &output](auto& v, std::size_t i) {
            output[i] = this->activation_probabilities_sub<I>(v);
        });
    }

#ifdef DLL_SVM_SUPPORT

    template <typename Samples, typename Input, typename DBN = this_type, cpp_enable_if(dbn_traits<DBN>::concatenate())>
    void add_activation_probabilities(Samples& result, const Input& sample) {
        result.emplace_back(full_output_size());
        full_activation_probabilities(sample, result.back());
    }

    template <typename Samples,typename Input, typename DBN = this_type, cpp_disable_if(dbn_traits<DBN>::concatenate())>
    void add_activation_probabilities(Samples& result, const Input& sample) {
        result.push_back(activation_probabilities(sample));
    }

    template <typename Input>
    using svm_sample_t = std::conditional_t<
        dbn_traits<this_type>::concatenate(),
        etl::dyn_vector<weight>,                             //In full mode, use a simple 1D vector
        typename types_helper<layers - 1, Input>::output_t>; //In normal mode, use the output of the last layer

    template <typename Input>
    using svm_samples_t = std::vector<svm_sample_t<Input>>;

    template <typename Samples, typename Labels>
    void make_problem(const Samples& training_data, const Labels& labels, bool scale = false) {
        svm_samples_t<safe_value_t<Samples>> svm_samples;

        //Get all the activation probabilities
        for (auto& sample : training_data) {
            add_activation_probabilities(svm_samples, sample);
        }

        //static_cast ensure using the correct overload
        problem = svm::make_problem(labels, static_cast<const svm_samples_t<safe_value_t<Samples>>&>(svm_samples), scale);
    }

    /*!
     * \brief Create the svm problem for this dbn
     */
    template <typename Iterator, typename LIterator>
    void make_problem(Iterator first, Iterator last, LIterator&& lfirst, LIterator&& llast, bool scale = false) {
        svm_samples_t<safe_value_t<Iterator>> svm_samples;

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
