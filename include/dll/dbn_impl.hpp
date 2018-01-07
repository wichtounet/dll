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
#include "cpp_utils/tuple_utils.hpp"

#include "generators.hpp"
#include "unit_type.hpp"
#include "trainer/dbn_trainer.hpp"
#include "trainer/rbm_trainer_fwd.hpp"
#include "dll/trainer/rbm_training_context.hpp"
#include "dbn_common.hpp"
#include "svm_common.hpp"
#include "util/export.hpp"
#include "util/timers.hpp"
#include "util/random.hpp"
#include "util/ready.hpp"
#include "dbn_detail.hpp" // dbn_detail namespace

namespace dll {
	
struct nullbuffer : std::streambuf {
	int overflow(int c) { return c; }
};

class nullstream : public std::ostream {
	nullbuffer m_sb;
public: 
	nullstream() : std::ostream(&m_sb) {}
};

template<typename O, typename Enable = void>
struct safe_value_type {
    using type = typename std::decay_t<O>::value_type;
};

template<typename O>
struct safe_value_type <O, std::enable_if_t<etl::is_etl_expr<O>>> {
    using type = etl::value_t<O>;
};

template<typename O>
using safe_value_t = typename safe_value_type<O>::type;

template<typename Layer>
struct is_output_layer {
    using traits = decay_layer_traits<Layer>;
    static constexpr bool value = !traits::is_transform_layer();
};

template<size_t Layer, typename DBN, typename Enable = void>
struct find_output_layer {
    static constexpr size_t L = Layer;
};

template<size_t Layer, typename DBN>
struct find_output_layer<Layer, DBN, std::enable_if_t<!is_output_layer<typename DBN::template layer_type<Layer>>::value>> {
    static constexpr size_t L = find_output_layer<Layer - 1, DBN>::L;
};

template<size_t Layer, typename DBN, typename Enable = void>
struct find_rbm_layer {
    static constexpr size_t L = Layer;
};

template<size_t Layer, typename DBN>
struct find_rbm_layer<Layer, DBN, std::enable_if_t<(Layer < DBN::layers_t::size)>> {
    static constexpr bool RBM = decay_layer_traits<typename DBN::template layer_type<Layer>>::is_rbm_layer();
    static constexpr size_t L = RBM ? Layer : find_rbm_layer<Layer + 1, DBN>::L;
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

    template <size_t N>
    using layer_type = detail::layer_type_t<N, layers_t>; ///< The type of the layer at index Nth

    // The weight is is extracted from the first layer, since all layers have the same type
    using weight = typename dbn_detail::extract_weight_t<0, this_type>::type; ///< The type of the weights

    static_assert(dbn_detail::validate_weight_type<this_type, weight>::value, "Every layer must have consistent weight type");

    using watcher_t = typename desc::template watcher_t<this_type>; ///< The watcher type

    static constexpr size_t input_layer_n   = 0;                                                   ///< The index of the input layer
    static constexpr size_t output_layer_n  = find_output_layer<layers_t::size - 1, this_type>::L; ///< The index of the output layer
    static constexpr size_t rbm_layer_n     = find_rbm_layer<0, this_type>::L;                     ///< The index of the first RBM layer
    static constexpr bool pretrain_possible = rbm_layer_n < layers_t::size;                        ///< Indicates if pretraining is possible

    using input_layer_t = layer_type<input_layer_n>;           ///< The type of the input layer
    using input_one_t   = typename input_layer_t::input_one_t; ///< The type of one input
    using input_t       = std::vector<input_one_t>;            ///< The type of a set of input

private:
    template <size_t I, typename Input>
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

    static constexpr size_t layers         = layers_t::size;     ///< The number of layers
    static constexpr size_t batch_size     = desc::BatchSize;    ///< The batch size (for finetuning)
    static constexpr size_t big_batch_size = desc::BigBatchSize; ///< The number of pretraining batch to do at once
    static constexpr auto loss             = desc::Loss;         ///< The loss function
    static constexpr auto updater          = desc::Updater;      ///< The Updater type
    static constexpr auto early            = desc::Early;        ///< The Early Stopping stragy

    layers_t tuples; ///< The layers

    weight learning_rate       = 0.1; ///< The learning rate for finetuning
    weight learning_rate_decay = 0.0; ///< The learning rate decay

    weight initial_momentum     = 0.9; ///< The initial momentum
    weight final_momentum       = 0.9; ///< The final momentum applied after *final_momentum_epoch* epoch
    weight final_momentum_epoch = 6;   ///< The epoch at which momentum change
    weight momentum             = 0;   ///< The current momentum

    weight l1_weight_cost = 0.0002; ///< The weight cost for L1 weight decay
    weight l2_weight_cost = 0.0002; ///< The weight cost for L2 weight decay

    weight rmsprop_decay        = 0.9;   ///< The decay rate for RMSPROP
    weight adadelta_beta        = 0.95;  ///< Adadelta beta factor
    weight adam_beta1           = 0.9;   ///< Adam's beta1 factor
    weight adam_beta2           = 0.999; ///< Adam's beta1 factor
    weight nadam_schedule_decay = 0.004; ///< NAdam's schedule decay

    weight gradient_clip = 5.0; ///< The gradient clipping

    weight goal     = 0.0; ///< The learning goal
    size_t patience = 1;   ///< The patience for early stopping goals
	
	std::ostream *log = new nullstream();// &std::cout;

#ifdef DLL_SVM_SUPPORT
    //TODO Ideally these fields should be private
    svm::model svm_model;    ///< The learned model
    svm::problem problem;    ///< libsvm is stupid, therefore, you cannot destroy the problem if you want to use the model...
    bool svm_loaded = false; ///< Indicates if a SVM model has been loaded (and therefore must be saved)
#endif                       //DLL_SVM_SUPPORT

    using categorical_generator_t = std::conditional_t<
        !dbn_traits<this_type>::batch_mode(),
        inmemory_data_generator_desc<dll::batch_size<batch_size>, dll::big_batch_size<big_batch_size>, dll::categorical, dll::scale_pre<desc::ScalePre>, dll::binarize_pre<desc::BinarizePre>, dll::normalize_pre_cond<desc::NormalizePre>>,
        outmemory_data_generator_desc<dll::batch_size<batch_size>, dll::big_batch_size<big_batch_size>, dll::categorical, dll::scale_pre<desc::ScalePre>, dll::binarize_pre<desc::BinarizePre>, dll::normalize_pre_cond<desc::NormalizePre>>>;

    using ae_generator_t = std::conditional_t<
        !dbn_traits<this_type>::batch_mode(),
        inmemory_data_generator_desc<dll::batch_size<batch_size>, dll::big_batch_size<big_batch_size>, dll::scale_pre<desc::ScalePre>, dll::autoencoder, dll::noise<desc::Noise>, dll::binarize_pre<desc::BinarizePre>, dll::normalize_pre_cond<desc::NormalizePre>>,
        outmemory_data_generator_desc<dll::batch_size<batch_size>, dll::big_batch_size<big_batch_size>, dll::scale_pre<desc::ScalePre>, dll::autoencoder, dll::noise<desc::Noise>, dll::binarize_pre<desc::BinarizePre>, dll::normalize_pre_cond<desc::NormalizePre>>>;

    template<size_t B>
    using rbm_generator_fast_t = std::conditional_t<
        !dbn_traits<this_type>::batch_mode(),
        inmemory_data_generator_desc<dll::batch_size<B>, dll::big_batch_size<big_batch_size>, dll::scale_pre<desc::ScalePre>, dll::autoencoder, dll::binarize_pre<desc::BinarizePre>, dll::normalize_pre_cond<desc::NormalizePre>>,
        outmemory_data_generator_desc<dll::batch_size<B>, dll::big_batch_size<big_batch_size>, dll::scale_pre<desc::ScalePre>, dll::autoencoder, dll::binarize_pre<desc::BinarizePre>, dll::normalize_pre_cond<desc::NormalizePre>>>;

    template<size_t B>
    using rbm_ingenerator_fast_inner_t = inmemory_data_generator_desc<dll::batch_size<B>, dll::big_batch_size<big_batch_size>, dll::autoencoder>;

    template<size_t B>
    using rbm_generator_fast_inner_t = std::conditional_t<
        !dbn_traits<this_type>::batch_mode(),
        inmemory_data_generator_desc<dll::batch_size<B>, dll::big_batch_size<big_batch_size>, dll::autoencoder>,
        outmemory_data_generator_desc<dll::batch_size<B>, dll::big_batch_size<big_batch_size>, dll::autoencoder>>;

    template<size_t B>
    using rbm_denoising_generator_fast_t = std::conditional_t<
        !dbn_traits<this_type>::batch_mode(),
        inmemory_data_generator_desc<dll::batch_size<B>, dll::big_batch_size<big_batch_size>, dll::scale_pre<desc::ScalePre>, dll::autoencoder, dll::noise<desc::Noise>, dll::binarize_pre<desc::BinarizePre>, dll::normalize_pre_cond<desc::NormalizePre>>,
        outmemory_data_generator_desc<dll::batch_size<B>, dll::big_batch_size<big_batch_size>, dll::scale_pre<desc::ScalePre>, dll::autoencoder, dll::noise<desc::Noise>, dll::binarize_pre<desc::BinarizePre>, dll::normalize_pre_cond<desc::NormalizePre>>>;

    template<size_t B>
    using rbm_denoising_ingenerator_fast_inner_t = inmemory_data_generator_desc<dll::batch_size<B>, dll::big_batch_size<big_batch_size>, dll::autoencoder, dll::noise<desc::Noise>>;

    template<size_t B>
    using rbm_denoising_generator_fast_inner_t = std::conditional_t<
        !dbn_traits<this_type>::batch_mode(),
        inmemory_data_generator_desc<dll::batch_size<B>, dll::big_batch_size<big_batch_size>, dll::autoencoder, dll::noise<desc::Noise>>,
        outmemory_data_generator_desc<dll::batch_size<B>, dll::big_batch_size<big_batch_size>, dll::autoencoder, dll::noise<desc::Noise>>>;

private:
    cpp::thread_pool<!dbn_traits<this_type>::is_serial()> pool;

    template<size_t I, cpp_disable_if(I == layers)>
    void dyn_init(){
        using fast_t = detail::layer_type_t<I, typename desc::base_layers>;

        decltype(auto) dyn_rbm = layer_get<I>();

        fast_t::dyn_init(dyn_rbm);

        dyn_init<I+1>();
    }

    template<size_t I, cpp_enable_iff(I == layers)>
    void dyn_init(){}

    template<size_t L = rbm_layer_n>
    auto get_rbm_generator_desc(){
        static_assert(decay_layer_traits<layer_type<L>>::is_rbm_layer(), "Invalid use of get_rbm_generator_desc");

        return rbm_generator_fast_t<layer_type<L>::batch_size>{};
    }

    template<size_t L = rbm_layer_n>
    auto get_rbm_denoising_generator_desc(){
        static_assert(decay_layer_traits<layer_type<L>>::is_rbm_layer(), "Invalid use of get_rbm_denoising_generator_desc");

        return rbm_denoising_generator_fast_t<layer_type<L>::batch_size>{};
    }

    template<size_t L = rbm_layer_n>
    auto get_rbm_generator_inner_desc(){
        static_assert(decay_layer_traits<layer_type<L>>::is_rbm_layer(), "Invalid use of get_rbm_generator_inner_desc");

        return rbm_generator_fast_inner_t<layer_type<L>::batch_size>{};
    }

    template<size_t L = rbm_layer_n>
    auto get_rbm_ingenerator_inner_desc(){
        static_assert(decay_layer_traits<layer_type<L>>::is_rbm_layer(), "Invalid use of get_rbm_generator_inner_desc");

        return rbm_ingenerator_fast_inner_t<layer_type<L>::batch_size>{};
    }

    template <size_t L, cpp_enable_iff((L < layers - 1) && decay_layer_traits<layer_type<L>>::is_rbm_layer())>
    void validate_pretraining_base() const {
        static_assert(layer_type<L>::batch_size == layer_type<rbm_layer_n>::batch_size, "Incoherent batch sizes in network");

        validate_pretraining_base<L + 1>();
    }

    template <size_t L, cpp_enable_iff((L == layers - 1) && decay_layer_traits<layer_type<L>>::is_rbm_layer())>
    void validate_pretraining_base() const {
        static_assert(layer_type<L>::batch_size == layer_type<rbm_layer_n>::batch_size, "Incoherent batch sizes in network");
    }

    template <size_t L, cpp_enable_iff((L < layers - 1) && !decay_layer_traits<layer_type<L>>::is_rbm_layer())>
    void validate_pretraining_base() const {
        validate_pretraining_base<L + 1>();
    }

    template <size_t L, cpp_enable_iff((L == layers - 1) && !decay_layer_traits<layer_type<L>>::is_rbm_layer())>
    void validate_pretraining_base() const {
        // Nothing to do
    }

    void validate_pretraining() const {
        validate_pretraining_base<0>();
    }

    template<typename Generator>
    void validate_generator(const Generator& generator){
        static_assert(batch_size == Generator::batch_size, "Invalid batch size for generator");

        cpp_unused(generator);
    }

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

        // Update defaults for each updater type

        if(updater == updater_type::RMSPROP){
            learning_rate = 0.001;
        }

        if(updater == updater_type::ADAGRAD){
            learning_rate = 0.01;
        }

        if(updater == updater_type::ADAM){
            learning_rate = 0.001;
        }

        if(updater == updater_type::ADAMAX){
            learning_rate = 0.002;
        }

        if(updater == updater_type::NADAM){
            learning_rate = 0.002;
        }
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
        size_t parameters = 0;

        std::cout << "Network with " << layers << " layers" << std::endl;

        for_each_layer([&parameters](auto& layer) {
            std::string pre = "    ";
            std::cout << pre;
            std::cout << layer.to_full_string(pre) << std::endl;

            cpp::static_if<decay_layer_traits<decltype(layer)>::is_neural_layer()>([&](auto f) {
                parameters += f(layer).parameters();
            });
        });

        std::cout << "Total parameters: " << parameters << std::endl;
    }

    static std::string shape_to_string(const std::vector<size_t>& shape){
        std::string s = "[B";

        for(auto& d : shape){
            s += "x" + std::to_string(d);
        }

        s += "]";

        return s;
    }

    template<typename Layer>
    void sub_display_pretty(const std::vector<size_t>& output, const std::string& parent, const std::string& pre, Layer& layer, std::vector<std::array<std::string, 4>>& rows) const {
        std::vector<size_t> sub_output = output;

        cpp::for_each_i(layer.layers, [&](size_t i, auto& sub_layer) {
            rows.emplace_back();
            auto& row = rows.back();

            std::string sub_pre = pre + "  ";
            std::string sub_parameters_str = "0";

            // Extract the number of parameters
            cpp::static_if<decay_layer_traits<decltype(sub_layer)>::is_neural_layer()>([&](auto f) {
                sub_parameters_str = std::to_string(f(sub_layer).parameters());
            });

            // Extract the output shape if possible
            sub_output = sub_layer.output_shape(sub_output);

            std::string number = parent + ":" + std::to_string(i);

            row[0] = number;
            row[1] = sub_pre + sub_layer.to_short_string(sub_pre);
            row[2] = sub_parameters_str;
            row[3] = this_type::shape_to_string(sub_output);

            cpp::static_if<decay_layer_traits<decltype(sub_layer)>::base_traits::is_multi>([&](auto f) {
                sub_display_pretty(sub_output, number, sub_pre, f(sub_layer), rows);
            });
        });
    }

    /*!
     * \brief Prints a textual representation of the network.
     */
    void display_pretty() const {
        constexpr size_t columns = 4;

        std::cout << '\n';

        std::array<std::string, columns> column_name;
        column_name[0] = "Index";
        column_name[1] = "Layer";
        column_name[2] = "Parameters";
        column_name[3] = "Output Shape";

        std::vector<std::array<std::string, columns>> rows;

        size_t parameters = 0;

        std::vector<size_t> output;

        for_each_layer_i([&](size_t I, auto& layer) {
            std::string parameters_str = "0";

            // Extract the number of parameters
            cpp::static_if<decay_layer_traits<decltype(layer)>::is_neural_layer()>([&](auto f) {
                parameters_str = std::to_string(f(layer).parameters());

                parameters += f(layer).parameters();
            });

            // Extract the output shape if possible
            output = layer.output_shape(output);

            rows.emplace_back();
            auto& row = rows.back();

            row[0] = std::to_string(I);
            row[1] = layer.to_short_string("");
            row[2] = parameters_str;
            row[3] = this_type::shape_to_string(output);

            cpp::static_if<decay_layer_traits<decltype(layer)>::base_traits::is_multi>([&](auto f) {
                sub_display_pretty(output, std::to_string(I), "", f(layer), rows);
            });
        });

        std::array<size_t, columns> column_length;
        column_length[0] = column_name[0].size();
        column_length[1] = column_name[1].size();
        column_length[2] = column_name[2].size();
        column_length[3] = column_name[3].size();

        for(auto& row : rows){
            column_length[0] = std::max(column_length[0], row[0].size());
            column_length[1] = std::max(column_length[1], row[1].size());
            column_length[2] = std::max(column_length[2], row[2].size());
            column_length[3] = std::max(column_length[3], row[3].size());
        }

        const size_t line_length = (columns + 1) * 1 + 2 + (columns - 1) * 2 + std::accumulate(column_length.begin(), column_length.end(), 0);

        std::cout << " " << std::string(line_length, '-') << '\n';

        printf(" | %-*s | %-*s | %-*s | %-*s |\n",
               int(column_length[0]), column_name[0].c_str(),
               int(column_length[1]), column_name[1].c_str(),
               int(column_length[2]), column_name[2].c_str(),
               int(column_length[3]), column_name[3].c_str());

        std::cout << " " << std::string(line_length, '-') << '\n';

        for(auto& row : rows){
            // Print the layer line
            printf(" | %-*s | %-*s | %*s | %-*s |\n",
                   int(column_length[0]), row[0].c_str(),
                   int(column_length[1]), row[1].c_str(),
                   int(column_length[2]), row[2].c_str(),
                   int(column_length[3]), row[3].c_str());
        }

        std::cout << " " << std::string(line_length, '-') << '\n';

        printf("  %*s: %*lu\n", int(column_length[0] + column_length[1] + 5), "Total Parameters", int(column_length[2]), parameters);
    }

    /*!
     * \brief Backup the weights of all the layers into a temporary storage.
     *
     * Only one temporary storage is available, i.e. calling this function
     * twice will erase the first saved weights.
     */
    void backup_weights() {
        for_each_layer([](auto& layer) {
            layer.backup_weights();
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
            layer.restore_weights();
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
            cpp::static_if<decay_layer_traits<decltype(layer)>::is_neural_layer()>([&](auto f) {
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
            cpp::static_if<decay_layer_traits<decltype(layer)>::is_neural_layer()>([&](auto f) {
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
    template <size_t N>
    layer_type<N>& layer_get() {
        return detail::layer_get<N>(tuples);
    }

    /*!
     * \brief Returns the Nth layer.
     * \return The Nth layer
     * \tparam N The index of the layer to return (from 0)
     */
    template <size_t N>
    constexpr const layer_type<N>& layer_get() const {
        return detail::layer_get<N>(tuples);
    }

    /*!
     * \brief Initialize the Nth layer  with the given args. The Nth layer must
     * be a dynamic layer.
     * \tparam N The index of the layer to return (from 0)
     * \param args The arguments for initialization of the layer.
     */
    template <size_t N, typename... Args>
    void init_layer(Args&&... args){
        layer_get<N>().init_layer(std::forward<Args>(args)...);
    }

    template <size_t N>
    size_t layer_input_size() const noexcept {
        return dll::input_size(layer_get<N>());
    }

    template <size_t N>
    size_t layer_output_size() const noexcept {
        return dll::output_size(layer_get<N>());
    }

    /*!
     * \brief Returns the input size expected by the network
     * \return The input size of the network
     */
    size_t input_size() const noexcept {
        return dll::input_size(layer_get<input_layer_n>());
    }

    /*!
     * \brief Returns the output size generated by the network
     * \return The output size of the network
     */
    size_t output_size() const noexcept {
        return dll::output_size(layer_get<output_layer_n>());
    }

    size_t full_output_size() const noexcept {
        size_t output = 0;
        for_each_layer([&output](auto& layer) {
            output += layer.output_size();
        });
        return output;
    }

    /*!
     * \brief Indicates if training should save memory (true) or run as efficiently as possible (false).
     *
     * \return true if the training should save memory, false otherwise.
     */
    constexpr bool batch_mode() const noexcept {
        return dbn_traits<this_type>::batch_mode();
    }

    /* pretrain */

    /*!
     * \brief Pretrain the network by training all layers in an unsupervised
     * manner.
     */
    template <typename Generator, cpp_enable_iff(is_generator<Generator>)>
    void pretrain(Generator& generator, size_t max_epochs) {
        static_assert(pretrain_possible, "Only networks with RBM can be pretrained");

        validate_pretraining();

        dll::auto_timer timer("net:pretrain");

        watcher_t watcher;

        watcher.pretraining_begin(*this, max_epochs);

        //Pretrain each layer one-by-one
        if /*constexpr*/ (batch_mode()) {
            std::cout << "DBN: Pretraining done in batch mode" << std::endl;

            if (layers_t::has_shuffle_layer) {
                std::cout << "warning: batch_mode dbn does not support shuffle in layers (will be ignored)";
            }

            pretrain_layer_batch<0>(generator, watcher, max_epochs);
        } else {
            pretrain_layer<0>(generator, watcher, max_epochs);
        }

        watcher.pretraining_end(*this);
    }

    /*!
     * \brief Pretrain the network by training all layers in an unsupervised
     * manner.
     */
    template <typename Input, cpp_enable_iff(!is_generator<Input>)>
    void pretrain(const Input& training_data, size_t max_epochs) {
        static_assert(pretrain_possible, "Only networks with RBM can be pretrained");

        validate_pretraining();

        // Create generator around the data
        auto generator = make_generator(
            training_data, training_data,
            training_data.size(), output_size(),
            get_rbm_generator_desc());

        generator->set_safe();

        pretrain(*generator, max_epochs);
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
    void pretrain(Iterator first, Iterator last, size_t max_epochs) {
        static_assert(pretrain_possible, "Only networks with RBM can be pretrained");

        validate_pretraining();

        // Create generator around the data
        auto generator = make_generator(
            first, last,
            first, last,
            std::distance(first, last), output_size(),
            get_rbm_generator_desc());

        generator->set_safe();

        pretrain(*generator, max_epochs);
    }

    /* pretrain_denoising */

    /*!
     * \brief Pretrain the network by training all layers in an unsupervised
     * manner, the network will learn to reconstruct noisy input.
     */
    template <typename Generator, cpp_enable_if(is_generator<Generator>)>
    void pretrain_denoising(Generator& generator, size_t max_epochs) {
        static_assert(pretrain_possible, "Only networks with RBM can be pretrained");

        validate_pretraining();

        dll::auto_timer timer("net:pretrain:denoising");

        watcher_t watcher;

        watcher.pretraining_begin(*this, max_epochs);

        //Pretrain each layer one-by-one
        if /*constexpr*/ (batch_mode()) {
            std::cout << "DBN: Denoising Pretraining done in batch mode" << std::endl;

            if (layers_t::has_shuffle_layer) {
                std::cout << "warning: batch_mode dbn does not support shuffle in layers (will be ignored)";
            }

            pretrain_layer_denoising_batch<0>(generator, watcher, max_epochs);
        } else {
            std::cout << "DBN: Denoising Pretraining" << std::endl;

            pretrain_layer_denoising<0>(generator, watcher, max_epochs);
        }

        watcher.pretraining_end(*this);
    }

    /*!
     * \brief Pretrain the network by training all layers in an unsupervised
     * manner, the network will learn to reconstruct noisy input.
     */
    template <typename Clean, cpp_disable_if(is_generator<Clean>)>
    void pretrain_denoising(const Clean& clean, size_t max_epochs) {
        static_assert(pretrain_possible, "Only networks with RBM can be pretrained");

        validate_pretraining();

        // Create generator around the data
        auto generator = make_generator(
            clean, clean,
            clean.size(), output_size(),
            get_rbm_denoising_generator_desc());

        generator->set_safe();

        pretrain_denoising(*generator, max_epochs);
    }

    /*!
     * \brief Pretrain the network by training all layers in an unsupervised
     * manner, the network will learn to reconstruct noisy input.
     */
    template <typename Noisy, typename Clean>
    void pretrain_denoising(const Noisy& noisy, const Clean& clean, size_t max_epochs) {
        static_assert(pretrain_possible, "Only networks with RBM can be pretrained");

        validate_pretraining();

        // Create generator around the data
        auto generator = make_generator(
            noisy, clean,
            noisy.size(), output_size(),
            get_rbm_generator_desc());

        generator->set_safe();

        pretrain_denoising(*generator, max_epochs);
    }

    /*!
     * \brief Pretrain the network by training all layers in an unsupervised
     * manner, the network will learn to reconstruct noisy input.
     */
    template <typename NIterator, typename CIterator>
    void pretrain_denoising(NIterator nit, NIterator nend, CIterator cit, CIterator cend, size_t max_epochs) {
        static_assert(pretrain_possible, "Only networks with RBM can be pretrained");

        validate_pretraining();

        // Create generator around the data
        auto generator = make_generator(
            nit, nend,
            cit, cend,
            std::distance(cit, cend), output_size(),
            get_rbm_generator_desc());

        generator->set_safe();

        pretrain_denoising(*generator, max_epochs);
    }

    /* train with labels */

    template <typename Iterator, typename LabelIterator>
    void train_with_labels(Iterator&& first, Iterator&& last, LabelIterator&& lfirst, LabelIterator&& llast, size_t labels, size_t max_epochs) {
        static_assert(pretrain_possible, "Only networks with RBM can be pretrained");

        dll::auto_timer timer("net:train:labels");

        cpp_assert(std::distance(first, last) == std::distance(lfirst, llast), "There must be the same number of values than labels");
        cpp_assert(dll::input_size(layer_get<layers - 1>()) == dll::output_size(layer_get<layers - 2>()) + labels, "There is no room for the labels units");

        watcher_t watcher;

        watcher.pretraining_begin(*this, max_epochs);

        train_with_labels<0>(first, last, watcher, std::forward<LabelIterator>(lfirst), std::forward<LabelIterator>(llast), labels, max_epochs);

        watcher.pretraining_end(*this);
    }

    template <typename Samples, typename Labels>
    void train_with_labels(const Samples& training_data, const Labels& training_labels, size_t labels, size_t max_epochs) {
        static_assert(pretrain_possible, "Only networks with RBM can be pretrained");

        cpp_assert(training_data.size() == training_labels.size(), "There must be the same number of values than labels");
        cpp_assert(dll::input_size(layer_get<layers - 1>()) == dll::output_size(layer_get<layers - 2>()) + labels, "There is no room for the labels units");

        train_with_labels(training_data.begin(), training_data.end(), training_labels.begin(), training_labels.end(), labels, max_epochs);
    }

    template<typename Input>
    size_t predict_labels(const Input& item, size_t labels) const {
        static_assert(pretrain_possible, "Only networks with RBM can be pretrained");

        cpp_assert(dll::input_size(layer_get<layers - 1>()) == dll::output_size(layer_get<layers - 2>()) + labels, "There is no room for the labels units");

        auto output_a = layer_get<layers - 1>().prepare_one_input();

        predict_labels<0>(item, output_a, labels);

        return std::distance(
            std::prev(output_a.end(), labels),
            std::max_element(std::prev(output_a.end(), labels), output_a.end()));
    }

    //Note: features_sub are alias functions for forward_one

    /*!
     * \brief Returns the output features of the Ith layer for the given sample and saves them in the given container
     * \param sample The sample to get features from
     * \param result The container where the results will be saved
     * \tparam I The index of the layer for features (starts at 0)
     * \return the output features of the Ith layer of the network
     */
    template <size_t I, typename Input, typename Output, typename T = this_type>
    auto features_sub(const Input& sample, Output& result) const {
        return result = forward_one<I>(sample);
    }

    /*!
     * \brief Returns the output features of the Ith layer for the given sample
     * \param sample The sample to get features from
     * \tparam I The index of the layer for features (starts at 0)
     * \return the output features of the Ith layer of the network
     */
    template <size_t I, typename Input>
    auto features_sub(const Input& sample) const {
        return forward_one<I>(sample);
    }

    //Note: features functions are alias functions for forward_one

    /*!
     * \brief Computes the output features for the given sample and saves them in the given container
     * \param sample The sample to get features from
     * \param result The container where to save the features
     * \return result
     */
    template <typename Output>
    auto features(const input_one_t& sample, Output& result) const {
        return result = forward_one(sample);
    }

    /*!
     * \brief Returns the output features for the given sample
     * \param sample The sample to get features from
     * \return the output features of the last layer of the network
     */
    template<typename Input>
    auto features(const Input& sample) const {
        return forward_one(sample);
    }

    // Forward one batch at a time

    // Forward functions are not perfect:
    // TODO: Transform layers should be applied inline

    /*
     * \brief Return the test representation for the given input batch.
     *
     * \tparam LS The layer from which the representation is extracted
     * \tparam L The layer to which the input is given
     *
     * \param sample The input batch to the layer L
     *
     * \return The test representation of the LS layer forwarded from L
     */
    template <size_t LS, size_t L, typename Input, cpp_enable_iff((L != LS))>
    decltype(auto) test_forward_batch_impl(Input&& sample) const {
        decltype(auto) next = layer_get<L>().test_forward_batch(sample);
        return test_forward_batch_impl<LS, L+1>(next);
    }

    /*
     * \brief Return the test representation for the given input batch.
     *
     * \tparam LS The layer from which the representation is extracted
     * \tparam L The layer to which the input is given
     *
     * \param sample The input batch to the layer L
     *
     * \return The test representation of the LS layer forwarded from L
     */
    template <size_t LS, size_t L, typename Input, cpp_enable_iff((L == LS))>
    decltype(auto) test_forward_batch_impl(Input&& sample) const {
        return layer_get<L>().test_forward_batch(sample);
    }

    /*
     * \brief Return the train representation for the given input batch.
     *
     * \tparam LS The layer from which the representation is extracted
     * \tparam L The layer to which the input is given
     *
     * \param sample The input batch to the layer L
     *
     * \return The train representation of the LS layer forwarded from L
     */
    template <size_t LS, size_t L, typename Input, cpp_enable_iff((L != LS))>
    decltype(auto) train_forward_batch_impl(Input&& sample) {
        decltype(auto) next = layer_get<L>().train_forward_batch(sample);
        return train_forward_batch_impl<LS, L+1>(next);
    }

    /*
     * \brief Return the train representation for the given input batch.
     *
     * \tparam LS The layer from which the representation is extracted
     * \tparam L The layer to which the input is given
     *
     * \param sample The input batch to the layer L
     *
     * \return The train representation of the LS layer forwarded from L
     */
    template <size_t LS, size_t L, typename Input, cpp_enable_iff((L == LS))>
    decltype(auto) train_forward_batch_impl(Input&& sample) {
        return layer_get<L>().train_forward_batch(sample);
    }

    /*
     * \brief Return the test representation for the given input batch.
     *
     * \tparam LS The layer from which the representation is extracted
     * \tparam L The layer to which the input is given
     *
     * \param sample The input batch to the layer L
     *
     * \return The test representation of the LS layer forwarded from L
     */
    template <size_t LS = layers - 1, size_t L = 0, typename Input>
    decltype(auto) test_forward_batch(Input&& sample) const {
        return test_forward_batch_impl<LS, L>(sample);
    }

    /*
     * \brief Return the train representation for the given input batch.
     *
     * \tparam LS The layer from which the representation is extracted
     * \tparam L The layer to which the input is given
     *
     * \param sample The input batch to the layer L
     *
     * \return The train representation of the LS layer forwarded from L
     */
    template <size_t LS = layers - 1, size_t L = 0, typename Input>
    decltype(auto) train_forward_batch(Input&& sample) {
        return train_forward_batch_impl<LS, L>(sample);
    }

    /*
     * \brief Return the test representation for the given input batch.
     *
     * \tparam LS The layer from which the representation is extracted
     * \tparam L The layer to which the input is given
     *
     * \param sample The input batch to the layer L
     *
     * \return The test representation of the LS layer forwarded from L
     */
    template <size_t LS = layers - 1, size_t L = 0, typename Input>
    decltype(auto) forward_batch(Input&& sample) const {
        return test_forward_batch_impl<LS, L>(sample);
    }

    // Forward one sample at a time
    // This is not as fast as it could be, far from it, but supports
    // larger range of input. The rationale being that time should
    // be spent in forward_batch

    /*
     * \brief Return the test representation for the given input sample.
     *
     * \tparam LS The layer from which the representation is extracted
     * \tparam L The layer to which the input is given
     *
     * \param sample The input sample to the layer L
     *
     * \return The test representation of the LS layer forwarded from L
     */
    template <size_t LS, size_t L, typename Input, cpp_enable_iff((L != LS))>
    decltype(auto) test_forward_one_impl(Input&& sample) const {
        decltype(auto) next = layer_get<L>().test_forward_one(sample);
        return test_forward_one_impl<LS, L+1>(next);
    }

    /*
     * \brief Return the test representation for the given input sample.
     *
     * \tparam LS The layer from which the representation is extracted
     * \tparam L The layer to which the input is given
     *
     * \param sample The input sample to the layer L
     *
     * \return The test representation of the LS layer forwarded from L
     */
    template <size_t LS, size_t L, typename Input, cpp_enable_iff((L == LS))>
    decltype(auto) test_forward_one_impl(Input&& sample) const {
        return layer_get<L>().test_forward_one(sample);
    }

    /*
     * \brief Return the train representation for the given input sample.
     *
     * \tparam LS The layer from which the representation is extracted
     * \tparam L The layer to which the input is given
     *
     * \param sample The input sample to the layer L
     *
     * \return The train representation of the LS layer forwarded from L
     */
    template <size_t LS, size_t L, typename Input, cpp_enable_iff((L != LS))>
    decltype(auto) train_forward_one_impl(Input&& sample) {
        decltype(auto) next = layer_get<L>().train_forward_one(sample);
        return train_forward_one_impl<LS, L+1>(next);
    }

    /*
     * \brief Return the train representation for the given input sample.
     *
     * \tparam LS The layer from which the representation is extracted
     * \tparam L The layer to which the input is given
     *
     * \param sample The input sample to the layer L
     *
     * \return The train representation of the LS layer forwarded from L
     */
    template <size_t LS, size_t L, typename Input, cpp_enable_iff((L == LS))>
    decltype(auto) train_forward_one_impl(Input&& sample) {
        return layer_get<L>().train_forward_one(sample);
    }

    /*
     * \brief Return the test representation for the given input sample.
     *
     * \tparam LS The layer from which the representation is extracted
     * \tparam L The layer to which the input is given
     *
     * \param sample The input sample to the layer L
     *
     * \return The test representation of the LS layer forwarded from L
     */
    template <size_t LS = layers - 1, size_t L = 0, typename Input>
    decltype(auto) test_forward_one(Input&& sample) const {
        return test_forward_one_impl<LS, L>(sample);
    }

    /*
     * \brief Return the train representation for the given input sample.
     *
     * \tparam LS The layer from which the representation is extracted
     * \tparam L The layer to which the input is given
     *
     * \param sample The input sample to the layer L
     *
     * \return The train representation of the LS layer forwarded from L
     */
    template <size_t LS = layers - 1, size_t L = 0, typename Input>
    decltype(auto) train_forward_one(Input&& sample) {
        return train_forward_one_impl<LS, L>(sample);
    }

    /*
     * \brief Return the test representation for the given input sample.
     *
     * \tparam LS The layer from which the representation is extracted
     * \tparam L The layer to which the input is given
     *
     * \param sample The input sample to the layer L
     *
     * \return The test representation of the LS layer forwarded from L
     */
    template <size_t LS = layers - 1, size_t L = 0, typename Input>
    decltype(auto) forward_one(Input&& sample) const {
        return test_forward_one_impl<LS, L>(sample);
    }

    // Forward a collection of samples at a time
    // This is not as fast as it could be, far from it, but supports
    // larger range of input. The rationale being that time should
    // be spent in forward_batch

    /*
     * \brief Return the test representation for the given collection of inputs.
     *
     * \tparam LS The layer from which the representation is extracted
     * \tparam L The layer to which the input is given
     *
     * \param samples The collection of inputs to the layer L
     *
     * \return The test representation of the LS layer forwarded from L
     */
    template <size_t LS, size_t L, typename Inputs, cpp_enable_iff((L != LS))>
    decltype(auto) test_forward_many_impl(Inputs&& samples) const {
        decltype(auto) layer = layer_get<L>();

        auto next = prepare_many_ready_output(layer, samples[0], samples.size());

        layer.test_forward_many(next, samples);

        return test_forward_many_impl<LS, L+1>(next);
    }

    /*
     * \brief Return the test representation for the given collection of inputs.
     *
     * \tparam LS The layer from which the representation is extracted
     * \tparam L The layer to which the input is given
     *
     * \param samples The collection of inputs to the layer L
     *
     * \return The test representation of the LS layer forwarded from L
     */
    template <size_t LS, size_t L, typename Inputs, cpp_enable_iff((L == LS))>
    decltype(auto) test_forward_many_impl(Inputs&& samples) const {
        decltype(auto) layer = layer_get<L>();

        auto out = prepare_many_ready_output(layer, samples[0], samples.size());

        layer.test_forward_many(out, samples);

        return out;
    }

    /*
     * \brief Return the train representation for the given collection of inputs.
     *
     * \tparam LS The layer from which the representation is extracted
     * \tparam L The layer to which the input is given
     *
     * \param samples The collection of inputs to the layer L
     *
     * \return The train representation of the LS layer forwarded from L
     */
    template <size_t LS, size_t L, typename Inputs, cpp_enable_iff((L != LS))>
    decltype(auto) train_forward_many_impl(Inputs&& samples) {
        decltype(auto) layer = layer_get<L>();

        auto next = prepare_many_ready_output(layer, samples[0], samples.size());

        layer.train_forward_many(next, samples);

        return train_forward_many_impl<LS, L+1>(next);
    }

    /*
     * \brief Return the train representation for the given collection of inputs.
     *
     * \tparam LS The layer from which the representation is extracted
     * \tparam L The layer to which the input is given
     *
     * \param samples The collection of inputs to the layer L
     *
     * \return The train representation of the LS layer forwarded from L
     */
    template <size_t LS, size_t L, typename Inputs, cpp_enable_iff((L == LS))>
    decltype(auto) train_forward_many_impl(Inputs&& samples) {
        decltype(auto) layer = layer_get<L>();

        auto out = prepare_many_ready_output(layer, samples[0], samples.size());

        layer.train_forward_many(out, samples);

        return out;
    }

    /*
     * \brief Return the test representation for the given collection of inputs.
     *
     * \tparam LS The layer from which the representation is extracted
     * \tparam L The layer to which the input is given
     *
     * \param samples The collection of inputs to the layer L
     *
     * \return The test representation of the LS layer forwarded from L
     */
    template <size_t LS = layers - 1, size_t L = 0, typename Inputs>
    decltype(auto) test_forward_many(Inputs&& samples) const {
        return test_forward_many_impl<LS, L>(samples);
    }

    /*
     * \brief Return the train representation for the given collection of inputs.
     *
     * \tparam LS The layer from which the representation is extracted
     * \tparam L The layer to which the input is given
     *
     * \param samples The collection of inputs to the layer L
     *
     * \return The train representation of the LS layer forwarded from L
     */
    template <size_t LS = layers - 1, size_t L = 0, typename Inputs>
    decltype(auto) train_forward_many(Inputs&& samples) {
        return train_forward_many_impl<LS, L>(samples);
    }

    /*
     * \brief Return the test representation for the given collection of inputs.
     *
     * \tparam LS The layer from which the representation is extracted
     * \tparam L The layer to which the input is given
     *
     * \param samples The collection of inputs to the layer L
     *
     * \return The test representation of the LS layer forwarded from L
     */
    template <size_t LS = layers - 1, size_t L = 0, typename Inputs>
    decltype(auto) forward_many(Inputs&& samples) const {
        return test_forward_many_impl<LS, L>(samples);
    }

    // Forward a collection of samples (iterators) at a time
    // This is not as fast as it could be, far from it, but supports
    // larger range of input. The rationale being that time should
    // be spent in forward_batch

    /*
     * \brief Return the test representation for the given collection of inputs.
     *
     * \tparam LS The layer from which the representation is extracted
     * \tparam L The layer to which the input is given
     *
     * \param first Iterator to the first element of the collection of inputs
     * \param last Iterator to the past-the-end element of the collection of inputs
     *
     * \return The test representation of the LS layer forwarded from L
     */
    template <size_t LS, size_t L, typename Iterator, cpp_enable_iff((L != LS))>
    decltype(auto) test_forward_many_impl(const Iterator& first, const Iterator& last) const {
        decltype(auto) layer = layer_get<L>();

        auto n = std::distance(first, last);

        auto next = prepare_many_ready_output(layer, *first, n);

        cpp::foreach_i(first, last, [&](auto& sample, size_t i) {
            layer.test_forward_one(next[i], sample);
        });

        return test_forward_many_impl<LS, L+1>(next);
    }

    /*
     * \brief Return the test representation for the given collection of inputs.
     *
     * \tparam LS The layer from which the representation is extracted
     * \tparam L The layer to which the input is given
     *
     * \param first Iterator to the first element of the collection of inputs
     * \param last Iterator to the past-the-end element of the collection of inputs
     *
     * \return The test representation of the LS layer forwarded from L
     */
    template <size_t LS, size_t L, typename Iterator, cpp_enable_iff((L == LS))>
    decltype(auto) test_forward_many_impl(const Iterator& first, const Iterator& last) const {
        decltype(auto) layer = layer_get<L>();

        auto n = std::distance(first, last);

        auto out = prepare_many_ready_output(layer, *first, n);

        cpp::foreach_i(first, last, [&](auto& sample, size_t i) {
            layer.test_forward_one(out[i], sample);
        });

        return out;
    }

    /*
     * \brief Return the train representation for the given collection of inputs.
     *
     * \tparam LS The layer from which the representation is extracted
     * \tparam L The layer to which the input is given
     *
     * \param first Iterator to the first element of the collection of inputs
     * \param last Iterator to the past-the-end element of the collection of inputs
     *
     * \return The train representation of the LS layer forwarded from L
     */
    template <size_t LS, size_t L, typename Iterator, cpp_enable_iff((L != LS))>
    decltype(auto) train_forward_many_impl(const Iterator& first, const Iterator& last) {
        decltype(auto) layer = layer_get<L>();

        auto n = std::distance(first, last);

        auto next = prepare_many_ready_output(layer, *first, n);

        cpp::foreach_i(first, last, [&](auto& sample, size_t i) {
            layer.train_forward_one(next[i], sample);
        });

        return train_forward_many_impl<LS, L+1>(next);
    }

    /*
     * \brief Return the train representation for the given collection of inputs.
     *
     * \tparam LS The layer from which the representation is extracted
     * \tparam L The layer to which the input is given
     *
     * \param first Iterator to the first element of the collection of inputs
     * \param last Iterator to the past-the-end element of the collection of inputs
     *
     * \return The train representation of the LS layer forwarded from L
     */
    template <size_t LS, size_t L, typename Iterator, cpp_enable_iff((L == LS))>
    decltype(auto) train_forward_many_impl(const Iterator& first, const Iterator& last) {
        decltype(auto) layer = layer_get<L>();

        auto n = std::distance(first, last);

        auto out = prepare_many_ready_output(layer, *first, n);

        cpp::foreach_i(first, last, [&](auto& sample, size_t i) {
            layer.train_forward_one(out[i], sample);
        });

        return out;
    }

    /*
     * \brief Return the test representation for the given collection of inputs.
     *
     * \tparam LS The layer from which the representation is extracted
     * \tparam L The layer to which the input is given
     *
     * \param first Iterator to the first element of the collection of inputs
     * \param last Iterator to the past-the-end element of the collection of inputs
     *
     * \return The test representation of the LS layer forwarded from L
     */
    template <size_t LS = layers - 1, size_t L = 0, typename Iterator>
    decltype(auto) test_forward_many(const Iterator& first, const Iterator& last) const {
        return test_forward_many_impl<LS, L>(first, last);
    }

    /*
     * \brief Return the train representation for the given collection of inputs.
     *
     * \tparam LS The layer from which the representation is extracted
     * \tparam L The layer to which the input is given
     *
     * \param first Iterator to the first element of the collection of inputs
     * \param last Iterator to the past-the-end element of the collection of inputs
     *
     * \return The train representation of the LS layer forwarded from L
     */
    template <size_t LS = layers - 1, size_t L = 0, typename Iterator>
    decltype(auto) train_forward_many(const Iterator& first, const Iterator& last) {
        return train_forward_many_impl<LS, L>(first, last);
    }

    /*
     * \brief Return the test representation for the given collection of inputs.
     *
     * \tparam LS The layer from which the representation is extracted
     * \tparam L The layer to which the input is given
     *
     * \param first Iterator to the first element of the collection of inputs
     * \param last Iterator to the past-the-end element of the collection of inputs
     *
     * \return The test representation of the LS layer forwarded from L
     */
    template <size_t LS = layers - 1, size_t L = 0, typename Iterator>
    decltype(auto) forward_many(const Iterator& first, const Iterator& last) const {
        return test_forward_many_impl<LS, L>(first, last);
    }

    /*!
     * \brief Save the features generated for the given sample in the given file.
     * \param sample The sample to get features from
     * \param file The output file
     * \param f The format of the exported features
     */
    template<typename Input>
    void save_features(const Input& sample, const std::string& file, format f = format::DLL) const {
        cpp_assert(f == format::DLL, "Only DLL format is supported for now");

        decltype(auto) probs = features(sample);

        if (f == format::DLL) {
            export_features_dll(probs, file);
        }
    }

    template <typename Output>
    size_t predict_label(const Output& result) const {
        return std::distance(result.begin(), std::max_element(result.begin(), result.end()));
    }

    template <typename Input>
    size_t predict(const Input& item) const {
        auto result = forward_one(item);
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

    // Fine-tune for classification

    /*!
     * \brief Fine tune the network for classifcation with a generator.
     *
     * \param generator A generator for data and labels
     * \param max_epochs The maximum number of epochs to train the network for.
     *
     * \return The final classification error
     */
    template <typename Generator>
    weight fine_tune(Generator& generator, size_t max_epochs) {
        dll::auto_timer timer("net:train:ft");

        validate_generator(generator);

        dll::dbn_trainer<this_type> trainer;
        return trainer.train(*this, generator, max_epochs);
    }

    /*!
     * \brief Fine tune the network for classifcation with a generator.
     *
     * \param train_generator A generator for training data and labels
     * \param val_generator A generator for validation data and labels
     * \param max_epochs The maximum number of epochs to train the network for.
     *
     * \return The final classification error
     */
    template <typename Generator, typename ValGenerator>
    weight fine_tune_val(Generator& train_generator, ValGenerator& val_generator, size_t max_epochs) {
        dll::auto_timer timer("net:train:ft");

        validate_generator(train_generator);
        validate_generator(val_generator);

        dll::dbn_trainer<this_type> trainer;
        return trainer.train(*this, train_generator, val_generator, max_epochs);
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
        // Create generator around the containers
        auto generator = dll::make_generator(
            training_data, labels,
            training_data.size(), output_size(), categorical_generator_t{});

        generator->set_safe();

        return fine_tune(*generator, max_epochs);
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
        // Create generator around the iterators
        auto generator = dll::make_generator(
            std::forward<Iterator>(first), std::forward<Iterator>(last),
            std::forward<LIterator>(lfirst), std::forward<LIterator>(llast),
            std::distance(lfirst, llast), output_size(), categorical_generator_t{});

        generator->set_safe();

        return fine_tune(*generator, max_epochs);
    }

    // Fine-tune for auto-encoder

    /*!
     * \brief Fine tune the network for autoencoder.
     * \param training_data A container containing all the samples
     * \param max_epochs The maximum number of epochs to train the network for.
     * \return The final classification error
     */
    template <typename Generator, cpp_enable_iff(is_generator<Generator>)>
    weight fine_tune_ae(Generator& generator, size_t max_epochs) {
        dll::auto_timer timer("net:train:ft:ae");

        validate_generator(generator);

        cpp_assert(dll::input_size(layer_get<0>()) == dll::output_size(layer_get<layers - 1>()), "The network is not build as an autoencoder");

        dll::dbn_trainer<this_type> trainer;
        return trainer.train(*this, generator, max_epochs);
    }

    /*!
     * \brief Fine tune the network for autoencoder.
     * \param training_data A container containing all the samples
     * \param max_epochs The maximum number of epochs to train the network for.
     * \return The final classification error
     */
    template <typename Samples, cpp_disable_if(is_generator<Samples>)>
    weight fine_tune_ae(const Samples& training_data, size_t max_epochs) {
        // Create generator around the containers
        auto generator = dll::make_generator(
            training_data, training_data,
            training_data.size(), output_size(), ae_generator_t{});

        generator->set_safe();

        return fine_tune_ae(*generator, max_epochs);
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
        // Create generator around the iterators
        auto generator = make_generator(
            std::forward<Iterator>(first), std::forward<Iterator>(last),
            std::forward<Iterator>(first), std::forward<Iterator>(last),
            std::distance(first, last), output_size(), ae_generator_t{});

        generator->set_safe();

        return fine_tune_ae(*generator, max_epochs);
    }
    
    // Fine tune for regression
    
    /*!
     * \brief Fine tune the network for regression.
     * \param generator Generator for samples and outputs
     * \param max_epochs The maximum number of epochs to train the network for.
     * \return The final average loss
     */
    template <typename Generator, cpp_enable_iff(is_generator<Generator>)>
    weight fine_tune_reg(Generator& generator, size_t max_epochs) {
        dll::auto_timer timer("net:train:ft:reg");

        validate_generator(generator);

        dll::dbn_trainer<this_type> trainer;
        return trainer.train(*this, generator, max_epochs);
    }
    
    /*!
     * \brief Fine tune the network for regression.
     * \param inputs A container containing all the samples
     * \param outputs A container containing the correct results
     * \param max_epochs The maximum number of epochs to train the network for.
     * \return The final average loss
     */
    template <typename Inputs, typename Outputs>
    weight fine_tune_reg(const Inputs& inputs, const Outputs& outputs, size_t max_epochs) {
        // Create generator around the containers
        cpp_assert(inputs.size() == outputs.size(), "The number of inputs does not match the number of outputs for training.");
        auto generator = dll::make_generator(
            inputs, outputs,
            inputs.size(), output_size(), ae_generator_t{});
        generator->set_safe();

        return fine_tune_reg(*generator, max_epochs);
    }
    
    /*!
     * \brief Fine tune the network for regression.
     * \param in_first Iterator to the first sample
     * \param in_last Iterator to the last sample
     * \param out_first Iterator to the first output
     * \param out_last Iterator to the last output
     * \param max_epochs The maximum number of epochs to train the network for.
     * \return The final average loss
     */
    template <typename InIterator, typename OutIterator>
    weight fine_tune_reg(InIterator&& in_first, InIterator&& in_last, OutIterator&& out_first, OutIterator&& out_last, size_t max_epochs) {
        // Create generator around the iterators
        cpp_assert(std::distance(in_first, in_last) == std::distance(out_first, out_last), "The number of inputs does not match the number of outputs for training.");
        auto generator = make_generator(
            std::forward<InIterator>(in_first), std::forward<InIterator>(in_last),
            std::forward<OutIterator>(out_first), std::forward<OutIterator>(out_last),
            std::distance(in_first, in_last), output_size(), ae_generator_t{});

        generator->set_safe();

        return fine_tune_reg(*generator, max_epochs);
    }

    template <size_t I, typename Input>
    auto prepare_output() const {
        using layer_input_t = typename types_helper<I, Input>::input_t;
        return layer_get<I>().template prepare_one_output<layer_input_t>();
    }

    /*!
     * \brief Prepare one empty output for this layer
     * \return an empty ETL matrix suitable to store one output of this layer
     *
     * \tparam Input The type of one Input
     */
    template <typename Input>
    auto prepare_one_output() const {
        return prepare_output<layers - 1, Input>();
    }

    /*!
     * \brief Evaluate the network on the given classification task.
     *
     * The result of the evaluation will be printed on the console.
     *
     * \param generator The data generator
     */
    template <typename Generator>
    void evaluate(Generator& generator){
        cpp::stop_watch<std::chrono::milliseconds> watch;

        validate_generator(generator);

        auto metrics = evaluate_metrics(generator);

        printf("\nEvaluation Results\n");
        printf("   error: %.5f \n", std::get<0>(metrics));
        printf("    loss: %.5f \n", std::get<1>(metrics));
        printf("evaluation took %dms \n", int(watch.elapsed()));
    }

    /*!
     * \brief Evaluate the network on the given classification task.
     *
     * The result of the evaluation will be printed on the console.
     *
     * \param samples The container containing the samples
     * \param labels The container containing the labels
     */
    template <typename Samples, typename Labels>
    void evaluate(const Samples&  samples, const Labels& labels){
        auto generator = make_generator(samples, labels, samples.size(), output_size(), categorical_generator_t{});

        generator->set_safe();

        return evaluate(*generator);
    }

    /*!
     * \brief Evaluate the network on the given classification task.
     *
     * The result of the evaluation will be printed on the console.
     *
     * \param iit The beginning of the range of the samples
     * \param iend The end of the range of the samples
     * \param lit The beginning of the range of the labels
     * \param lend The end of the range of the labels
     */
    template <typename InputIterator, typename LabelIterator>
    void evaluate(InputIterator&& iit, InputIterator&& iend, LabelIterator&& lit, LabelIterator&& lend){
        auto generator = make_generator(iit, iend, lit, lend, std::distance(lit, lend), output_size(), categorical_generator_t{});

        generator->set_safe();

        evaluate(*generator);
    }

    /*!
     * \brief Evaluate the network on the given auto-encoder task.
     *
     * The result of the evaluation will be printed on the console.
     *
     * \param generator The data generator
     */
    template <typename Generator, cpp_enable_iff(is_generator<Generator>)>
    void evaluate_ae(Generator& generator){
        validate_generator(generator);

        evaluate(generator);
    }

    /*!
     * \brief Evaluate the network on the given auto-encoder task.
     *
     * The result of the evaluation will be printed on the console.
     *
     * \param samples The container containing the samples
     * \param labels The container containing the labels
     */
    template <typename Samples, cpp_enable_iff(!is_generator<Samples>)>
    void evaluate_ae(const Samples&  samples){
        auto generator = make_generator(samples, samples, samples.size(), output_size(), ae_generator_t{});

        generator->set_safe();

        return evaluate(*generator);
    }

    /*!
     * \brief Evaluate the network on the given auto-encoder task.
     *
     * The result of the evaluation will be printed on the console.
     *
     * \param iit The beginning of the range of the samples
     * \param iend The end of the range of the samples
     * \param lit The beginning of the range of the labels
     * \param lend The end of the range of the labels
     */
    template <typename InputIterator>
    void evaluate_ae(InputIterator&& iit, InputIterator&& iend){
        auto generator = make_generator(iit, iend, iit, iend, std::distance(iit, iend), output_size(), ae_generator_t{});

        generator->set_safe();

        evaluate(*generator);
    }

    /*!
     * \brief Evaluate the network on the given classification task
     * and return the classification error.
     *
     * The result of the evaluation will be printed on the console.
     *
     * \param generator The data generator
     */
    template <typename Generator>
    double evaluate_error(Generator& generator){
        validate_generator(generator);

        auto metrics = evaluate_metrics(generator);

        return std::get<0>(metrics);
    }

    /*!
     * \brief Evaluate the network on the given classification task
     * and return the classification error.
     *
     * \param samples The container containing the samples
     * \param labels The container containing the labels
     *
     * \return The classification error
     */
    template <typename Samples, typename Labels>
    double evaluate_error(const Samples&  samples, const Labels& labels){
        auto generator = make_generator(samples, labels, samples.size(), output_size(), categorical_generator_t{});

        generator->set_safe();

        return evaluate_error(*generator);
    }

    /*!
     * \brief Evaluate the network on the given classification task
     * and return the classification error.
     *
     * \param iit The beginning of the range of the samples
     * \param iend The end of the range of the samples
     * \param lit The beginning of the range of the labels
     * \param lend The end of the range of the labels
     *
     * \return The classification error
     */
    template <typename InputIterator, typename LabelIterator>
    double evaluate_error(InputIterator&& iit, InputIterator&& iend, LabelIterator&& lit, LabelIterator&& lend){
        auto generator = make_generator(iit, iend, lit, lend, std::distance(lit, lend), output_size(), categorical_generator_t{});

        generator->set_safe();

        return evaluate_error(*generator);
    }

    /*!
     * \brief Evaluate the network on the given classification task
     * and return the classification error.
     *
     * The result of the evaluation will be printed on the console.
     *
     * \param generator The data generator
     */
    template <typename Generator, cpp_enable_iff(is_generator<Generator>)>
    double evaluate_error_ae(Generator& generator){
        validate_generator(generator);

        auto metrics = evaluate_metrics(generator);

        return std::get<0>(metrics);
    }

    /*!
     * \brief Evaluate the network on the given classification task
     * and return the classification error.
     *
     * \param samples The container containing the samples
     * \param labels The container containing the labels
     *
     * \return The classification error
     */
    template <typename Samples, cpp_enable_iff(!is_generator<Samples>)>
    double evaluate_error_ae(const Samples&  samples){
        auto generator = make_generator(samples, samples, samples.size(), output_size(), ae_generator_t{});

        generator->set_safe();

        return evaluate_error(*generator);
    }

    /*!
     * \brief Evaluate the network on the given classification task
     * and return the classification error.
     *
     * \param iit The beginning of the range of the samples
     * \param iend The end of the range of the samples
     * \param lit The beginning of the range of the labels
     * \param lend The end of the range of the labels
     *
     * \return The classification error
     */
    template <typename InputIterator>
    double evaluate_error_ae(InputIterator&& iit, InputIterator&& iend){
        auto generator = make_generator(iit, iend, iit, iend, std::distance(iit, iend), output_size(), ae_generator_t{});

        generator->set_safe();

        return evaluate_error(*generator);
    }

    using metrics_t = std::tuple<double, double>; ///< The metrics returned by evaluate_metrics

    template <loss_function F, typename Output, typename Labels, cpp_enable_iff((F == loss_function::CATEGORICAL_CROSS_ENTROPY))>
    std::tuple<double, double> compute_loss(size_t n, bool full_batch, double s, Output&& output, Labels&& labels){
        dll::auto_timer timer("net:compute_loss:CCE");

        double batch_loss;
        double batch_error;

        if (cpp_unlikely(!full_batch)) {
            auto soutput = slice(output, 0, n);

            batch_loss  = etl::ml::cce_loss(soutput, labels, -1.0 / s);
            batch_error = etl::ml::cce_error(soutput, labels, 1.0 / s);
        } else {
            batch_loss  = etl::ml::cce_loss(output, labels, -1.0 / s);
            batch_error = etl::ml::cce_error(output, labels, 1.0 / s);
        }

        return std::make_tuple(batch_error, batch_loss);
    }

    template <loss_function F, typename Output, typename Labels, cpp_enable_iff((F == loss_function::BINARY_CROSS_ENTROPY))>
    std::tuple<double, double> compute_loss(size_t n, bool full_batch, double s, Output&& output, Labels&& labels){
        dll::auto_timer timer("net:compute_loss:BCE");

        double batch_loss;
        double batch_error;

        // Avoid Nan in log(out) or log(1-out)
        auto out = etl::force_temporary(etl::clip(output, 0.001, 0.999));

        if (cpp_unlikely(!full_batch)) {
            auto sout = slice(out, 0, n);

            batch_loss  = (-1.0 / (s * output_size())) * sum((labels >> log(sout)) + ((1.0 - labels) >> log(1.0 - sout)));
            batch_error = (1.0 / (s * output_size())) * asum(labels - sout);
        } else {
            batch_loss  = (-1.0 / (s * output_size())) * sum((labels >> log(out)) + ((1.0 - labels) >> log(1.0 - out)));
            batch_error = (1.0 / (s * output_size())) * asum(labels - output);
        }

        return std::make_tuple(batch_error, batch_loss);
    }

    template <loss_function F, typename Output, typename Labels, cpp_enable_iff((F == loss_function::MEAN_SQUARED_ERROR))>
    std::tuple<double, double> compute_loss(size_t n, bool full_batch, double s, Output&& output, Labels&& labels){
        dll::auto_timer timer("net:compute_loss:MSE");

        double batch_loss;
        double batch_error;

        if (cpp_unlikely(!full_batch)) {
            auto soutput = slice(output, 0, n);

            batch_loss  = (1.0 / (2.0 * s)) * sum((soutput - labels) >> (soutput - labels));
            batch_error = (1.0 / s) * asum(labels - soutput);
        } else {
            batch_loss  = (1.0 / (2.0 * s)) * sum((output - labels) >> (output - labels));
            batch_error = (1.0 / s) * asum(labels - output);
        }

        return std::make_tuple(batch_error, batch_loss);
    }

    /*!
     * \brief Evaluate the network on the given output batch and labels and return the metrics.
     *
     * \param output The output of the network
     * \param labels The expected labels
     * \param n The size of the batch
     * \param normalize Indicates if the metrics must be normalized by the size of the batch
     *
     * \return A tuple contains the error and the loss
     */
    template <typename Output, typename Labels>
    metrics_t evaluate_metrics_batch(Output&& output, Labels&& labels, size_t n, bool normalize){
        const bool full_batch = n == etl::dim<0>(output);

        double s = 1.0;

        if(normalize){
            s = n;
        }

        // TODO Detect if labels are categorical already or not
        // And change the way this is done

        // CPP17 Use if constexpr instaed of SFINAE
        return compute_loss<loss>(n, full_batch, s, output, labels);
    }

    /*!
     * \brief Evaluate the network on the given classification task
     * and return the evaluation metrics.
     *
     * \param generator The data generator
     *
     * \return The evaluation metrics
     */
    template <typename Generator>
    metrics_t evaluate_metrics(Generator& generator){
        validate_generator(generator);

        auto forward_helper = [this](auto&& input_batch){
            return this->forward_batch(input_batch);
        };

        return evaluate_metrics(generator, forward_helper);
    }

    /*!
     * \brief Evaluate the network on the given classification task
     * and return the evaluation metrics.
     *
     * \param generator The data generator
     * \param helper The function to use to compute a batch of output
     *
     * \return The evaluation metrics
     */
    template <typename Generator, typename Helper>
    metrics_t evaluate_metrics(Generator& generator, Helper&& helper){
        validate_generator(generator);

        // Starts a new
        generator.reset();

        // Set the generator in test mode
        generator.set_test();

        double error = 0.0;
        double loss  = 0.0;

        while(generator.has_next_batch()){
            auto input_batch = generator.data_batch();
            auto label_batch = generator.label_batch();

            decltype(auto) output = helper(input_batch);

            double batch_error;
            double batch_loss;

            std::tie(batch_error, batch_loss) = evaluate_metrics_batch(output, label_batch, etl::dim<0>(input_batch), false);

            error += batch_error;
            loss += batch_loss;

            generator.next_batch();
        }

        error /= generator.size();
        loss /= generator.size();

        return std::make_tuple(error, loss);
    }

public:
    template <size_t I, size_t S, typename Input, cpp_enable_iff(I != S)>
    void full_activation_probabilities(const Input& input, full_output_t& result, size_t& i) const {
        auto output = forward_one<I, I>(input);
        for(auto& feature : output){
            result[i++] = feature;
        }
        full_activation_probabilities<I+1, S>(output, result, i);
    }

    template <size_t I, size_t S, typename Input, cpp_enable_iff(I == S)>
    void full_activation_probabilities(const Input& input, full_output_t& result, size_t& i) const {
        auto output = forward_one<I, I>(input);
        for(auto& feature : output){
            result[i++] = feature;
        }
    }

    template <typename Input>
    void full_activation_probabilities(const Input& input, full_output_t& result) const {
        size_t i = 0;
        full_activation_probabilities<0, layers - 1>(input, result, i);
    }

    template <typename Input>
    auto full_activation_probabilities(const Input& input) const {
        full_output_t result(full_output_size());
        full_activation_probabilities(input, result);
        return result;
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
    template <typename Input, typename DBN = this_type, cpp_enable_iff(dbn_traits<DBN>::concatenate())>
    auto get_final_activation_probabilities(const Input& sample) const {
        return full_activation_probabilities(sample);
    }

    template <typename Input, typename DBN = this_type, cpp_disable_if(dbn_traits<DBN>::concatenate())>
    auto get_final_activation_probabilities(const Input& sample) const {
        return forward_one(sample);
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
    bool svm_grid_search(const Samples& training_data, const Labels& labels, size_t n_fold = 5, const svm::rbf_grid& g = svm::rbf_grid()) {
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
    bool svm_grid_search(It&& first, It&& last, LIt&& lfirst, LIt&& llast, size_t n_fold = 5, const svm::rbf_grid& g = svm::rbf_grid()) {
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
    //By default all layer are trained
    template <size_t I, class Enable = void>
    struct train_next : std::true_type {};

    //The last layer is not always trained (softmax for instance)
    template <size_t I>
    struct train_next<I, std::enable_if_t<(I == layers - 1)>> : cpp::bool_constant<layer_traits<layer_type<I>>::pretrain_last()> {};

    template <size_t I, typename Enable = void>
    struct inline_next : std::false_type {};

    template <size_t I>
    struct inline_next<I, std::enable_if_t<(I < layers)>> : cpp::bool_constant<layer_traits<layer_type<I>>::is_pooling_layer()> {};

    template <size_t I, typename Generator>
    void inline_layer_pretrain(Generator& generator, watcher_t& watcher, size_t max_epochs) {
        decltype(auto) layer = layer_get<I>();
        decltype(auto) next_layer = layer_get<I + 1>();

        watcher.pretrain_layer(*this, I+1, next_layer, generator.size());

        // Reset correctly the generator
        generator.reset();
        generator.set_test();

        // Need one output in order to create the generator
        auto one = prepare_one_ready_output(layer, generator.data_batch()(0));
        auto two = prepare_one_ready_output(next_layer, one);

        // Prepare a generator to hold the data
        auto next_generator = prepare_generator(
            two, two,
            generator.size(), output_size(),
            get_rbm_ingenerator_inner_desc());

        next_generator->set_safe();

        // Compute the input of the next layer
        // using batch activation

        size_t i = 0;
        while(generator.has_next_batch()){
            auto batch = layer.train_forward_batch(generator.data_batch());
            auto next_batch = next_layer.train_forward_batch(batch);

            next_generator->set_data_batch(i, next_batch);
            next_generator->set_label_batch(i, next_batch);

            i += etl::dim<0>(next_batch);

            generator.next_batch();
        }

        // Release the memory if possible
        generator.clear();

        pretrain_layer<I + 2>(*next_generator, watcher, max_epochs);
    }

    template <size_t I, typename Generator, cpp_enable_iff((I < layers))>
    void pretrain_layer(Generator& generator, watcher_t& watcher, size_t max_epochs) {
        using layer_t = layer_type<I>;

        decltype(auto) layer = layer_get<I>();

        watcher.pretrain_layer(*this, I, layer, generator.size());

        cpp::static_if<layer_traits<layer_t>::is_pretrained()>([&](auto f) {
            // Train the RBM
            f(layer).template train<!watcher_t::ignore_sub,               //Enable the RBM Watcher or not
                                    dbn_detail::rbm_watcher_t<watcher_t>> //Replace the RBM watcher if not void
                (generator, max_epochs);
        });

        //When the next layer is a pooling layer, a lot of memory can be saved by directly computing
        //the activations of two layers at once
        cpp::static_if<inline_next<I + 1>::value>([&](auto f) {
            f(this)->template inline_layer_pretrain<I>(generator, watcher, max_epochs);
        });

        if /*constexpr*/ (train_next<I + 1>::value && !inline_next<I + 1>::value) {
            // Reset correctly the generator
            generator.reset();
            generator.set_test();

            // Need one output in order to create the generator
            auto one = prepare_one_ready_output(layer, generator.data_batch()(0));

            // Prepare a generator to hold the data
            auto next_generator = prepare_generator(
                one, one,
                generator.size(), output_size(),
                get_rbm_ingenerator_inner_desc());

            next_generator->set_safe();

            // Compute the input of the next layer
            // using batch activation

            size_t i = 0;
            while(generator.has_next_batch()){
                auto next_batch = layer.train_forward_batch(generator.data_batch());

                next_generator->set_data_batch(i, next_batch);
                next_generator->set_label_batch(i, next_batch);

                i += etl::dim<0>(next_batch);

                generator.next_batch();
            }

            // Release the memory if possible
            generator.clear();

            //Pass the output to the next layer
            this->template pretrain_layer<I + 1>(*next_generator, watcher, max_epochs);
        }
    }

    //Stop template recursion
    template <size_t I, typename Generator, cpp_enable_iff((I == layers))>
    void pretrain_layer(Generator&, watcher_t&, size_t) {}

    /* Pretrain with denoising */

    template <size_t I, typename Generator, cpp_enable_iff((I < layers))>
    void pretrain_layer_denoising(Generator& generator, watcher_t& watcher, size_t max_epochs) {
        using layer_t = layer_type<I>;

        decltype(auto) layer = layer_get<I>();

        watcher.pretrain_layer(*this, I, layer, generator.size());

        cpp::static_if<layer_traits<layer_t>::is_pretrained()>([&](auto f) {
            // Train the RBM
            f(layer).template train_denoising<
                                              !watcher_t::ignore_sub,               //Enable the RBM Watcher or not
                                              dbn_detail::rbm_watcher_t<watcher_t>> //Replace the RBM watcher if not void
                (generator, max_epochs);
        });

        if /*constexpr*/ (train_next<I + 1>::value) {
            // Reset correctly the generator
            generator.reset();
            generator.set_test();

            // Need one output in order to create the generator
            auto one_n = prepare_one_ready_output(layer, generator.data_batch()(0));
            auto one_c = prepare_one_ready_output(layer, generator.label_batch()(0));

            // Prepare a generator to hold the data
            auto next_generator = prepare_generator(
                one_n, one_c,
                generator.size(), output_size(),
                get_rbm_ingenerator_inner_desc());

            next_generator->set_safe();

            // Compute the input of the next layer
            // using batch activation

            size_t i = 0;
            while(generator.has_next_batch()){
                auto next_batch_n = layer.train_forward_batch(generator.data_batch());
                auto next_batch_c = layer.train_forward_batch(generator.label_batch());

                next_generator->set_data_batch(i, next_batch_n);
                next_generator->set_label_batch(i, next_batch_c);

                i += etl::dim<0>(next_batch_n);

                generator.next_batch();
            }

            // Release the memory if possible
            generator.clear();

            //In the standard case, pass the output to the next layer
            pretrain_layer_denoising<I + 1>(*next_generator, watcher, max_epochs);
        }
    }

    //Stop template recursion
    template <size_t I, typename Generator, cpp_enable_iff((I == layers))>
    void pretrain_layer_denoising(Generator&, watcher_t&, size_t) {}

    /* Pretrain in batch mode */

    //By default no layer is ignored
    template <size_t I, class Enable = void>
    struct batch_layer_ignore : std::false_type {};

    //Transform and pooling layers can safely be skipped
    template <size_t I>
    struct batch_layer_ignore<I, std::enable_if_t<(I < layers)>> : cpp::or_u<
        layer_traits<layer_type<I>>::is_pooling_layer(),
        layer_traits<layer_type<I>>::is_transform_layer(),
        layer_traits<layer_type<I>>::is_standard_layer(),
        !layer_traits<layer_type<I>>::pretrain_last()> {};

    //Special handling for the layer 0
    //data is coming from iterators not from input
    template <size_t I, typename Generator, cpp_enable_iff((I == 0 && !batch_layer_ignore<I>::value))>
    void pretrain_layer_batch(Generator& generator, watcher_t& watcher, size_t max_epochs) {
        decltype(auto) rbm = layer_get<I>();

        watcher.pretrain_layer(*this, I, rbm, 0);

        // The train function can be used directly because the
        // batch mode will be done by the generator itself
        rbm.template train<
                !watcher_t::ignore_sub,               //Enable the RBM Watcher or not
                dbn_detail::rbm_watcher_t<watcher_t>> //Replace the RBM watcher if not void
            (generator, max_epochs);

        //Train the next layer
        pretrain_layer_batch<I + 1>(generator, watcher, max_epochs);
    }

    //Special handling for untrained layers
    template <size_t I, typename Generator, cpp_enable_iff(batch_layer_ignore<I>::value)>
    void pretrain_layer_batch(Generator& generator, watcher_t& watcher, size_t max_epochs) {
        //We simply go up one layer on untrained layers
        pretrain_layer_batch<I + 1>(generator, watcher, max_epochs);
    }

    //Normal version
    template <size_t I, typename Generator, cpp_enable_iff((I > 0 && I < layers && !batch_layer_ignore<I>::value))>
    void pretrain_layer_batch(Generator& generator, watcher_t& watcher, size_t max_epochs) {
        using layer_t = layer_type<I>;

        decltype(auto) rbm = layer_get<I>();

        watcher.pretrain_layer(*this, I, rbm, 0);

        using rbm_trainer_t = dll::rbm_trainer<layer_t, !watcher_t::ignore_sub, dbn_detail::rbm_watcher_t<watcher_t>>;

        //Initialize the RBM trainer
        rbm_trainer_t r_trainer;

        //Init the RBM and training parameters
        r_trainer.init_training(rbm, generator);

        //Get the specific trainer (CD)
        auto trainer = rbm_trainer_t::get_trainer(rbm);

        //Train for max_epochs epoch
        for (size_t epoch = 0; epoch < max_epochs; ++epoch) {
            size_t big_batch = 0;

            //Create a new context for this epoch
            rbm_training_context context;

            r_trainer.init_epoch();

            generator.reset();
            generator.set_train();

            while (generator.has_next_batch()) {
                auto next_batch = forward_batch<I - 1>(generator.data_batch());

                r_trainer.train_batch(next_batch, next_batch, trainer, context, rbm);

                if (dbn_traits<this_type>::is_verbose()) {
                    watcher.pretraining_batch(*this, big_batch);
                }

                generator.next_batch();
            }

            r_trainer.finalize_epoch(epoch, context, rbm);
        }

        r_trainer.finalize_training(rbm);

        //train the next layer, if any
        pretrain_layer_batch<I + 1>(generator, watcher, max_epochs);
    }

    //Stop template recursion
    template <size_t I, typename Generator, cpp_enable_iff(I == layers)>
    void pretrain_layer_batch(Generator&, watcher_t&, size_t) {}

    /* Pretrain layer denoising batch  */

    //Special handling for the layer 0
    //data is coming from iterators not from input
    template <size_t I, typename Generator, cpp_enable_iff((I == 0 && !batch_layer_ignore<I>::value))>
    void pretrain_layer_denoising_batch(Generator& generator, watcher_t& watcher, size_t max_epochs) {
        decltype(auto) rbm = layer_get<I>();

        watcher.pretrain_layer(*this, I, rbm, 0);

        // The train function can be used directly because the
        // batch mode will be done by the generator itself
        rbm.template train_denoising<
                !watcher_t::ignore_sub,               //Enable the RBM Watcher or not
                dbn_detail::rbm_watcher_t<watcher_t>> //Replace the RBM watcher if not void
            (generator, max_epochs);

        //Train the next layer
        pretrain_layer_denoising_batch<I + 1>(generator, watcher, max_epochs);
    }

    //Special handling for untrained layers
    template <size_t I, typename Generator, cpp_enable_iff(batch_layer_ignore<I>::value)>
    void pretrain_layer_denoising_batch(Generator& generator, watcher_t& watcher, size_t max_epochs) {
        //We simply go up one layer on untrained layers
        pretrain_layer_denoising_batch<I + 1>(generator, watcher, max_epochs);
    }

    //Normal version
    template <size_t I, typename Generator, cpp_enable_iff((I > 0 && I < layers && !batch_layer_ignore<I>::value))>
    void pretrain_layer_denoising_batch(Generator& generator, watcher_t& watcher, size_t max_epochs) {
        using layer_t = layer_type<I>;

        decltype(auto) rbm = layer_get<I>();

        watcher.pretrain_layer(*this, I, rbm, 0);

        using rbm_trainer_t = dll::rbm_trainer<layer_t, !watcher_t::ignore_sub, dbn_detail::rbm_watcher_t<watcher_t>>;

        //Initialize the RBM trainer
        rbm_trainer_t r_trainer;

        //Init the RBM and training parameters
        r_trainer.init_training(rbm, generator);

        //Get the specific trainer (CD)
        auto trainer = rbm_trainer_t::get_trainer(rbm);

        //Train for max_epochs epoch
        for (size_t epoch = 0; epoch < max_epochs; ++epoch) {
            size_t big_batch = 0;

            //Create a new context for this epoch
            rbm_training_context context;

            r_trainer.init_epoch();

            generator.reset();
            generator.set_train();

            while (generator.has_next_batch()) {
                auto next_batch_n = forward_batch<I - 1>(generator.data_batch());
                auto next_batch_c = forward_batch<I - 1>(generator.label_batch());

                r_trainer.train_batch(next_batch_n, next_batch_c, trainer, context, rbm);

                if (dbn_traits<this_type>::is_verbose()) {
                    watcher.pretraining_batch(*this, big_batch);
                }

                generator.next_batch();
            }

            r_trainer.finalize_epoch(epoch, context, rbm);
        }

        r_trainer.finalize_training(rbm);

        //train the next layer, if any
        pretrain_layer_denoising_batch<I + 1>(generator, watcher, max_epochs);
    }

    //Stop template recursion
    template <size_t I, typename Generator, cpp_enable_iff(I == layers)>
    void pretrain_layer_denoising_batch(Generator&, watcher_t&, size_t) {}

    /* Train with labels */

    template <size_t I, typename Iterator, typename LabelIterator>
    std::enable_if_t<(I < layers)> train_with_labels(Iterator first, Iterator last, watcher_t& watcher, LabelIterator lit, LabelIterator lend, size_t labels, size_t max_epochs) {
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
            auto next_a = this_type::template forward_many<I, I>(first, last);

            //If the next layer is the last layer
            if (I == layers - 2) {
                using input_t = std::decay_t<decltype(*first)>;
                auto big_next_a = layer.template prepare_output<input_t>(input_size, true, labels);

                //Cannot use std copy since the sub elements have different size
                for (size_t i = 0; i < next_a.size(); ++i) {
                    for (size_t j = 0; j < next_a[i].size(); ++j) {
                        big_next_a[i][j] = next_a[i][j];
                    }
                }

                size_t i = 0;
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

    template <size_t I, typename Iterator, typename LabelIterator>
    std::enable_if_t<(I == layers)> train_with_labels(Iterator, Iterator, watcher_t&, LabelIterator, LabelIterator, size_t, size_t) {}

    /* Predict with labels */

    /*!
     * \brief Predict the output labels (only when pretrain with labels)
     */
    template <size_t I, typename Input, typename Output>
    std::enable_if_t<(I < layers)> predict_labels(const Input& input, Output& output, size_t labels) const {
        decltype(auto) layer = layer_get<I>();

        auto next_a = prepare_one_ready_output(layer, input);
        auto next_s = prepare_one_ready_output(layer, input);

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
                auto big_next_a = layer.template prepare_one_output<Input>(true, labels);

                for (size_t i = 0; i < next_a.size(); ++i) {
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
    template <size_t I, typename Input, typename Output>
    std::enable_if_t<(I == layers)> predict_labels(const Input&, Output&, size_t) const {}

    /* Activation Probabilities */

#ifdef DLL_SVM_SUPPORT

    template <typename Samples, typename Input, typename DBN = this_type, cpp_enable_iff(dbn_traits<DBN>::concatenate())>
    void add_activation_probabilities(Samples& result, const Input& sample) {
        result.emplace_back(full_output_size());
        full_activation_probabilities(sample, result.back());
    }

    template <typename Samples,typename Input, typename DBN = this_type, cpp_disable_if(dbn_traits<DBN>::concatenate())>
    void add_activation_probabilities(Samples& result, const Input& sample) {
        result.push_back(forward_one(sample));
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
