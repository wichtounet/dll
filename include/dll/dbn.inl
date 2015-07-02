//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_DBN_INL
#define DLL_DBN_INL

#include <tuple>

#include "cpp_utils/tuple_utils.hpp"
#include "cpp_utils/static_if.hpp"

#include "unit_type.hpp"
#include "dbn_trainer.hpp"
#include "conjugate_gradient.hpp"
#include "dbn_common.hpp"
#include "svm_common.hpp"
#include "input_converter.hpp"

namespace dll {

//TODO Could be good to ensure that either a) all layer have the same weight b) use the correct type for each layer

template<std::size_t I, typename DBN, typename Enable = void>
struct extract_weight_t;

template<std::size_t I, typename DBN>
struct extract_weight_t <I, DBN, std::enable_if_t<layer_traits<typename DBN::template layer_type<I>>::is_transform_layer()>> {
    using type = typename extract_weight_t<I+1, DBN>::type;
};

template<std::size_t I, typename DBN>
struct extract_weight_t <I, DBN, cpp::disable_if_t<layer_traits<typename DBN::template layer_type<I>>::is_transform_layer()>> {
    using type = typename DBN::template layer_type<I>::weight;
};

/*!
 * \brief A Deep Belief Network implementation
 */
template<typename Desc>
struct dbn final {
    using desc = Desc;
    using this_type = dbn<desc>;

    using tuple_type = typename desc::layers::tuple_type;
    tuple_type tuples;

    static constexpr const std::size_t layers = desc::layers::layers;

    template <std::size_t N>
    using layer_type = typename std::tuple_element<N, tuple_type>::type;

    using weight = typename extract_weight_t<0, this_type>::type;

    using watcher_t = typename desc::template watcher_t<this_type>;

    weight learning_rate = 0.77;

    weight initial_momentum = 0.5;      ///< The initial momentum
    weight final_momentum = 0.9;        ///< The final momentum applied after *final_momentum_epoch* epoch
    weight final_momentum_epoch = 6;    ///< The epoch at which momentum change

    weight weight_cost = 0.0002;        ///< The weight cost for weight decay

    weight momentum = 0;                ///< The current momentum

    thread_pool<dbn_traits<this_type>::is_parallel()> pool;

#ifdef DLL_SVM_SUPPORT
    svm::model svm_model;               ///< The learned model
    svm::problem problem;               ///< libsvm is stupid, therefore, you cannot destroy the problem if you want to use the model...
    bool svm_loaded = false;            ///< Indicates if a SVM model has been loaded (and therefore must be saved)
#endif //DLL_SVM_SUPPORT

    //No arguments by default
    template<cpp_disable_if_cst(dbn_traits<this_type>::is_dynamic())>
    dbn(){}

//Note: The tuple implementation of Clang and G++ seems highly
//different. Indeed, g++ only allows to forward arguments to the
//constructors if they are directly convertible.

#ifdef __clang__
    template<typename... T, cpp_enable_if_cst(dbn_traits<this_type>::is_dynamic())>
    explicit dbn(T&&... layers) : tuples(std::forward<T>(layers)...) {
        //Nothing else to init
    }
#else
    template<typename... T, cpp_enable_if_cst(dbn_traits<this_type>::is_dynamic())>
    explicit dbn(T&&... layers) : tuples({std::forward<T>(layers)}...) {
        //Nothing else to init
    }
#endif

    //No copying
    dbn(const dbn& dbn) = delete;
    dbn& operator=(const dbn& dbn) = delete;

    //No moving
    dbn(dbn&& dbn) = delete;
    dbn& operator=(dbn&& dbn) = delete;

    void display() const {
        std::size_t parameters = 0;

        std::cout << "DBN with " << layers << " layers" << std::endl;

        cpp::for_each(tuples, [&parameters](auto& layer){
            std::cout << "\t";
            cpp::static_if<decay_layer_traits<decltype(layer)>::is_rbm_layer()>([&](auto f){
                parameters += f(layer).parameters();
            });
            layer.display();
        });

        std::cout << "Total parameters: " << parameters << std::endl;
    }

    void store(const std::string& file) const {
        std::ofstream os(file, std::ofstream::binary);
        store(os);
    }

    void load(const std::string& file){
        std::ifstream is(file, std::ifstream::binary);
        load(is);
    }

    void store(std::ostream& os) const {
        cpp::for_each(tuples, [&os](auto& layer){
            cpp::static_if<decay_layer_traits<decltype(layer)>::is_rbm_layer()>([&](auto f){
                f(layer).store(os);
            });
        });

#ifdef DLL_SVM_SUPPORT
        svm_store(*this, os);
#endif //DLL_SVM_SUPPORT
    }

    void load(std::istream& is){
        cpp::for_each(tuples, [&is](auto& layer){
            cpp::static_if<decay_layer_traits<decltype(layer)>::is_rbm_layer()>([&](auto f){
                f(layer).load(is);
            });
        });

#ifdef DLL_SVM_SUPPORT
        svm_load(*this, is);
#endif //DLL_SVM_SUPPORT
    }

    template<std::size_t N>
    auto layer_get() -> typename std::add_lvalue_reference<layer_type<N>>::type {
        return std::get<N>(tuples);
    }

    template<std::size_t N>
    constexpr auto layer_get() const -> typename std::add_lvalue_reference<typename std::add_const<layer_type<N>>::type>::type {
        return std::get<N>(tuples);
    }

    template<std::size_t N>
    static constexpr std::size_t layer_input_size() noexcept {
        return layer_traits<layer_type<N>>::input_size();
    }

    template<std::size_t N>
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
        for_each_type<tuple_type>([&output](auto* layer){
            output += std::decay_t<std::remove_pointer_t<decltype(layer)>>::output_size();
        });
        return output;
    }

    /*{{{ Pretrain */

private:
    template<std::size_t I, class Enable = void>
    struct train_next;

    template<std::size_t I>
    struct train_next<I, std::enable_if_t<(I < layers - 1)>> : std::true_type {};

    template<std::size_t I>
    struct train_next<I, std::enable_if_t<(I == layers - 1)>> : cpp::bool_constant<layer_traits<layer_type<I>>::pretrain_last()> {};

    template<std::size_t I>
    struct train_next<I, std::enable_if_t<(I > layers - 1)>> : std::false_type {};

    template<typename Iterator>
    static std::size_t fast_distance(Iterator& first, Iterator& last){
        if(std::is_same<typename std::iterator_traits<Iterator>::iterator_category, std::random_access_iterator_tag>::value){
            return std::distance(first, last);
        } else {
            return 0;
        }
    }

    template<typename One>
    static void flatten_in(std::vector<std::vector<One>>& deep, std::vector<One>& flat){
        flat.reserve(deep.size());

        for(auto& d : deep){
            std::move(d.begin(), d.end(), std::back_inserter(flat));
        }
    }

    template<typename One>
    static void flatten_in_clr(std::vector<std::vector<One>>& deep, std::vector<One>& flat){
        flat.reserve(deep.size());

        for(auto& d : deep){
            std::move(d.begin(), d.end(), std::back_inserter(flat));
        }

        deep.clear();
    }

    template<typename One>
    static std::vector<One> flatten_clr(std::vector<std::vector<One>>& deep){
        std::vector<One> flat;

        flatten_in_clr(deep, flat);

        return flat;
    }

    template<typename One>
    static std::vector<One> flatten(std::vector<std::vector<One>>& deep){
        std::vector<One> flat;

        flatten_in(deep, flat);

        return flat;
    }

    template<std::size_t I, typename Iterator, typename Watcher>
    std::enable_if_t<(I<layers)> pretrain_layer(Iterator first, Iterator last, Watcher& watcher, std::size_t max_epochs){
        using layer_t = layer_type<I>;

        decltype(auto) layer = layer_get<I>();

        watcher.template pretrain_layer<layer_t>(*this, I, fast_distance(first, last));

        cpp::static_if<layer_traits<layer_t>::is_trained()>([&](auto f){
            f(layer).template train<
                !watcher_t::ignore_sub,                  //Enable the RBM Watcher or not
                dbn_detail::rbm_watcher_t<watcher_t>>    //Replace the RBM watcher if not void
                    (first, last, max_epochs);
        });

        if(train_next<I+1>::value){
            auto next_a = layer.template prepare_output<layer_input_t<I, Iterator>>(std::distance(first, last));

            maybe_parallel_foreach_i(pool, first, last, [&layer, &next_a](auto& v, std::size_t i){
                layer.activate_one(v, next_a[i]);
            });

            //In the standard case, pass the output to the next layer
            cpp::static_if<!layer_traits<layer_t>::is_multiplex_layer()>([&](auto f){
                f(this)->template pretrain_layer<I+1>(next_a.begin(), next_a.end(), watcher, max_epochs);
            });

            //In case of a multiplex layer, the output is flattened
            cpp::static_if<layer_traits<layer_t>::is_multiplex_layer()>([&](auto f){
                auto flattened_next_a = f(this)->flatten_clr(next_a);

                f(this)->template pretrain_layer<I+1>(flattened_next_a.begin(), flattened_next_a.end(), watcher, max_epochs);
            });
        }
    }

    //Stop template recursion
    template<std::size_t I, typename Iterator, typename Watcher>
    std::enable_if_t<(I==layers)> pretrain_layer(Iterator, Iterator, Watcher&, std::size_t){}

    template<std::size_t I, class Enable = void>
    struct batch_layer_ignore : std::false_type {};

    template<std::size_t I>
    struct batch_layer_ignore<I, std::enable_if_t<(I < layers)>> : cpp::or_u<layer_traits<layer_type<I>>::is_pooling_layer(), layer_traits<layer_type<I>>::is_transform_layer(), !layer_traits<layer_type<I>>::pretrain_last()> {};

    //Special handling for the layer 0
    //data is coming from iterators not from input
    template<std::size_t I, typename Iterator, typename Watcher, cpp_enable_if((I == 0 && !batch_layer_ignore<I>::value))>
    void pretrain_layer_batch(Iterator first, Iterator last, Watcher& watcher, std::size_t max_epochs){
        using rbm_t = layer_type<I>;

        decltype(auto) rbm = layer_get<I>();

        watcher.template pretrain_layer<rbm_t>(*this, I, 0);

        using rbm_trainer_t = dll::rbm_trainer<rbm_t, !watcher_t::ignore_sub, dbn_detail::rbm_watcher_t<watcher_t>>;

        //Initialize the RBM trainer
        rbm_trainer_t r_trainer;

        //Init the RBM and training parameters
        r_trainer.init_training(rbm, first, last); //TODO This may be highly slow...

        //Get the specific trainer (CD)
        auto trainer = rbm_trainer_t::template get_trainer<false>(rbm);

        auto big_batch_size = desc::BatchSize * get_batch_size(rbm);

        //Train for max_epochs epoch
        for(std::size_t epoch = 0; epoch < max_epochs; ++epoch){
            std::size_t big_batch = 0;

            //Create a new context for this epoch
            rbm_training_context context;

            r_trainer.init_epoch();

            auto it = first;
            auto end = last;

            while(it != end){
                auto batch_start = it;

                std::size_t i = 0;
                while(it != end && i < big_batch_size){
                    ++it;
                    ++i;
                }

                //Convert data to an useful form
                input_converter<this_type, 0, Iterator> converter(*this, batch_start, it);

                if(desc::BatchSize == 1){
                    //Train the RBM on this batch
                    r_trainer.train_batch(converter.begin(), converter.end(), converter.begin(), converter.end(), trainer, context, rbm);
                } else {
                    //Train the RBM on this big batch
                    r_trainer.train_sub(converter.begin(), converter.end(), converter.begin(), trainer, context, rbm);
                }

                if(dbn_traits<this_type>::is_verbose()){
                    watcher.pretraining_batch(*this, big_batch);
                }

                ++big_batch;
            }

            r_trainer.finalize_epoch(epoch, context, rbm);
        }

        r_trainer.finalize_training(rbm);

        pretrain_layer_batch<I+1>(first, last, watcher, max_epochs);
    }

    //Special handling for untrained layers
    template<std::size_t I, typename Iterator, typename Watcher, cpp_enable_if(batch_layer_ignore<I>::value)>
    void pretrain_layer_batch(Iterator first, Iterator last, Watcher& watcher, std::size_t max_epochs){
        //We simply go up one layer on pooling layers
        pretrain_layer_batch<I+1>(first, last, watcher, max_epochs);
    }

    template<std::size_t I, typename Input, typename Enable = void>
    struct layer_output;

    template<std::size_t I, typename Input>
    using layer_output_t = typename layer_output<I, Input>::type;

    template<std::size_t I, typename Input, typename Enable = void>
    struct layer_input;

    template<std::size_t I, typename Input>
    using layer_input_t = typename layer_input<I, Input>::type;

    template<std::size_t I, typename Input>
    struct layer_output<I, Input, std::enable_if_t<!layer_traits<layer_type<I>>::is_transform_layer()>> {
        using type = typename layer_type<I>::output_one_t;
    };

    template<std::size_t I, typename Input>
    struct layer_output<I, Input, std::enable_if_t<I == 0 && layer_traits<layer_type<I>>::is_transform_layer()>> {
        using type = typename Input::value_type;
    };

    template<std::size_t I, typename Input>
    struct layer_output<I, Input, std::enable_if_t<(I > 0) && layer_traits<layer_type<I>>::is_transform_layer()>> {
        using type = layer_input_t<I, Input>;
    };

    template<std::size_t I, typename Input>
    struct layer_input<I, Input, std::enable_if_t<!layer_traits<layer_type<I>>::is_transform_layer()>> {
        using type = typename layer_type<I>::input_one_t;
    };

    template<std::size_t I, typename Input>
    struct layer_input<I, Input, std::enable_if_t<I == 0 && layer_traits<layer_type<I>>::is_transform_layer()>> {
        using type = layer_input_t<I + 1, Input>;
    };

    template<std::size_t I, typename Input>
    struct layer_input<I, Input, std::enable_if_t<(I > 0) && layer_traits<layer_type<I>>::is_transform_layer()>> {
        using type = layer_output_t<I - 1, Input>;
    };

    //Normal version
    template<std::size_t I, typename Iterator, typename Watcher, cpp_enable_if((I>0 && I<layers && !dbn_traits<this_type>::is_multiplex() && !batch_layer_ignore<I>::value))>
    void pretrain_layer_batch(Iterator first, Iterator last, Watcher& watcher, std::size_t max_epochs){
        using rbm_t = layer_type<I>;

        decltype(auto) rbm = layer_get<I>();

        watcher.template pretrain_layer<rbm_t>(*this, I, 0);

        using rbm_trainer_t = dll::rbm_trainer<rbm_t, !watcher_t::ignore_sub, dbn_detail::rbm_watcher_t<watcher_t>>;

        //Initialize the RBM trainer
        rbm_trainer_t r_trainer;

        //Init the RBM and training parameters
        r_trainer.init_training(rbm, first, last);

        //Get the specific trainer (CD)
        auto trainer = rbm_trainer_t::template get_trainer<false>(rbm);

        auto big_batch_size = desc::BatchSize * get_batch_size(rbm);

        auto input = layer_get<I - 1>().template prepare_output<layer_input_t<I - 1, Iterator>>(big_batch_size);

        //Train for max_epochs epoch
        for(std::size_t epoch = 0; epoch < max_epochs; ++epoch){
            std::size_t big_batch = 0;

            //Create a new context for this epoch
            rbm_training_context context;

            r_trainer.init_epoch();

            auto it = first;
            auto end = last;

            while(it != end){
                auto batch_start = it;

                std::size_t i = 0;
                while(it != end && i < big_batch_size){
                    ++it;
                    ++i;
                }

                //Convert data to an useful form
                input_converter<this_type, 0, Iterator> converter(*this, batch_start, it);

                //Collect a big batch
                maybe_parallel_foreach_i(pool, converter.begin(), converter.end(), [this,&input](auto& v, std::size_t i){
                    this->activation_probabilities<0, I>(v, input[i]);
                });

                if(desc::BatchSize == 1){
                    //Train the RBM on this batch
                    r_trainer.train_batch(input.begin(), input.end(), input.begin(), input.end(), trainer, context, rbm);
                } else {
                    //Train the RBM on this big batch
                    r_trainer.train_sub(input.begin(), input.end(), input.begin(), trainer, context, rbm);
                }

                if(dbn_traits<this_type>::is_verbose()){
                    watcher.pretraining_batch(*this, big_batch);
                }

                ++big_batch;
            }

            r_trainer.finalize_epoch(epoch, context, rbm);
        }

        r_trainer.finalize_training(rbm);

        //train the next layer, if any
        pretrain_layer_batch<I+1>(first, last, watcher, max_epochs);
    }

    //Multiplex version
    template<std::size_t I, typename Iterator, typename Watcher, cpp_enable_if((I>0 && I<layers && dbn_traits<this_type>::is_multiplex() && !batch_layer_ignore<I>::value))>
    void pretrain_layer_batch(Iterator first, Iterator last, Watcher& watcher, std::size_t max_epochs){
        using rbm_t = layer_type<I>;

        decltype(auto) rbm = layer_get<I>();

        watcher.template pretrain_layer<rbm_t>(*this, I, 0);

        using rbm_trainer_t = dll::rbm_trainer<rbm_t, !watcher_t::ignore_sub, dbn_detail::rbm_watcher_t<watcher_t>>;

        //Initialize the RBM trainer
        rbm_trainer_t r_trainer;

        //Init the RBM and training parameters
        r_trainer.init_training(rbm, first, last);

        //Get the specific trainer (CD)
        auto trainer = rbm_trainer_t::template get_trainer<false>(rbm);

        auto rbm_batch_size = get_batch_size(rbm);
        auto big_batch_size = desc::BatchSize * rbm_batch_size;

        std::vector<std::vector<typename layer_type<I - 1>::output_deep_t>> input(big_batch_size);

        std::vector<typename rbm_t::input_one_t> input_flat;

        //Train for max_epochs epoch
        for(std::size_t epoch = 0; epoch < max_epochs; ++epoch){
            std::size_t big_batch = 0;

            //Create a new context for this epoch
            rbm_training_context context;

            r_trainer.init_epoch();

            auto it = first;
            auto end = last;

            while(it != end){
                auto batch_start = it;

                std::size_t i = 0;
                while(it != end && i < big_batch_size){
                    ++it;
                    ++i;
                }

                //Convert data to an useful form
                input_converter<this_type, 0, Iterator> converter(*this, batch_start, it);

                //Collect a big batch
                maybe_parallel_foreach_i(pool, converter.begin(), converter.end(), [this,&input](auto& v, std::size_t i){
                    this->activation_probabilities<0, I>(v, input[i]);
                });

                flatten_in(input, input_flat);

                for(auto& i : input){
                    i.clear();
                }

                auto batches = input_flat.size() / rbm_batch_size;
                auto offset = std::min(batches * rbm_batch_size, input_flat.size());

                if(batches <= 1){
                    //Train the RBM on one batch
                    r_trainer.train_batch(
                        input_flat.begin(), input_flat.begin() + offset,
                        input_flat.begin(), input_flat.begin() + offset, trainer, context, rbm);
                } else if(batches > 1){
                    //Train the RBM on this big batch
                    r_trainer.train_sub(input_flat.begin(), input_flat.begin() + offset, input_flat.begin(), trainer, context, rbm);
                }

                input_flat.erase(input_flat.begin(), input_flat.begin() + offset);

                if(dbn_traits<this_type>::is_verbose()){
                    watcher.pretraining_batch(*this, big_batch);
                }

                ++big_batch;
            }

            r_trainer.finalize_epoch(epoch, context, rbm);
        }

        r_trainer.finalize_training(rbm);

        //train the next layer, if any
        pretrain_layer_batch<I+1>(first, last, watcher, max_epochs);
    }

    //Stop template recursion
    template<std::size_t I, typename Iterator, typename Watcher, cpp_enable_if((I==layers && !batch_layer_ignore<I>::value))>
    void pretrain_layer_batch(Iterator, Iterator, Watcher&, std::size_t){}

public:
    /*!
     * \brief Pretrain the network by training all layers in an unsupervised
     * manner.
     */
    template<typename Samples>
    void pretrain(const Samples& training_data, std::size_t max_epochs){
        pretrain(training_data.begin(), training_data.end(), max_epochs);
    }

    /*!
     * \brief Pretrain the network by training all layers in an unsupervised
     * manner.
     */
    template<typename Iterator>
    void pretrain(Iterator first, Iterator last, std::size_t max_epochs){
        watcher_t watcher;

        watcher.pretraining_begin(*this, max_epochs);

        //Pretrain each layer one-by-one
        if(dbn_traits<this_type>::save_memory()){
            std::cout << "DBN: Pretraining done in batch mode to save memory" << std::endl;
            pretrain_layer_batch<0>(first, last, watcher, max_epochs);
        } else {
            //Convert data to an useful form
            input_converter<this_type, 0, Iterator> converter(*this, first, last);

            pretrain_layer<0>(converter.begin(), converter.end(), watcher, max_epochs);
        }

        watcher.pretraining_end(*this);
    }

    /*}}}*/

    /*{{{ Train with labels */

private:

    template<std::size_t I, typename Input, typename Watcher, typename LabelIterator>
    std::enable_if_t<(I<layers)> train_with_labels(const Input& input, Watcher& watcher, LabelIterator lit, LabelIterator lend, std::size_t labels, std::size_t max_epochs){
        using layer_t = layer_type<I>;

        decltype(auto) layer = layer_get<I>();

        watcher.template pretrain_layer<layer_t>(*this, I, input.size());

        cpp::static_if<layer_traits<layer_t>::is_trained()>([&](auto f){
            f(layer).template train<
                !watcher_t::ignore_sub, //Enable the RBM Watcher or not
                dbn_detail::rbm_watcher_t<watcher_t>> //Replace the RBM watcher if not void
                    (input, max_epochs);
        });

        if(I < layers - 1){
            bool is_last = I == layers - 2;

            auto next_a = layer.template prepare_output<layer_input_t<I, Input>>(input.size());
            auto next_s = layer.template prepare_output<layer_input_t<I, Input>>(input.size());

            layer.activate_many(input, next_a, next_s);

            if(is_last){
                auto big_next_a = layer.template prepare_output<layer_input_t<I, Input>>(input.size(), is_last, labels);

                //Cannot use std copy since the sub elements have different size
                for(std::size_t i = 0; i < next_a.size(); ++i){
                    for(std::size_t j = 0; j < next_a[i].size(); ++j){
                        big_next_a[i][j] = next_a[i][j];
                    }
                }

                std::size_t i = 0;
                while(lit != lend){
                    decltype(auto) label = *lit;

                    for(size_t l = 0; l < labels; ++l){
                        big_next_a[i][dll::output_size(layer) + l] = label == l ? 1.0 : 0.0;
                    }

                    ++i;
                    ++lit;
                }

                train_with_labels<I+1>(big_next_a, watcher, lit, lend, labels, max_epochs);
            } else {
                train_with_labels<I+1>(next_a, watcher, lit, lend, labels, max_epochs);
            }
        }
    }

    template<std::size_t I, typename Input, typename Watcher, typename LabelIterator>
    std::enable_if_t<(I==layers)> train_with_labels(Input&, Watcher&, LabelIterator, LabelIterator, std::size_t, std::size_t){}

public:

    template<typename Iterator, typename LabelIterator>
    void train_with_labels(Iterator&& first, Iterator&& last, LabelIterator&& lfirst, LabelIterator&& llast, std::size_t labels, std::size_t max_epochs){
        cpp_assert(std::distance(first, last) == std::distance(lfirst, llast), "There must be the same number of values than labels");
        cpp_assert(dll::input_size(layer_get<layers - 1>()) == dll::output_size(layer_get<layers - 2>()) + labels, "There is no room for the labels units");

        watcher_t watcher;

        watcher.pretraining_begin(*this, max_epochs);

        //Convert data to an useful form
        auto data = layer_get<0>().convert_input(std::forward<Iterator>(first), std::forward<Iterator>(last));

        train_with_labels<0>(data, watcher, std::forward<LabelIterator>(lfirst), std::forward<LabelIterator>(llast), labels, max_epochs);

        watcher.pretraining_end(*this);
    }

    //Note: dyn_vector cannot be replaced with fast_vector, because labels is runtime

    template<typename Samples, typename Labels>
    void train_with_labels(const Samples& training_data, const Labels& training_labels, std::size_t labels, std::size_t max_epochs){
        cpp_assert(training_data.size() == training_labels.size(), "There must be the same number of values than labels");
        cpp_assert(dll::input_size(layer_get<layers - 1>()) == dll::output_size(layer_get<layers - 2>()) + labels, "There is no room for the labels units");

        train_with_labels(training_data.begin(), training_data.end(), training_labels.begin(), training_labels.end(), labels, max_epochs);
    }

    /*}}}*/

    /*{{{ Predict with labels */

private:

    template<std::size_t I, typename Input, typename Output>
    std::enable_if_t<(I<layers)> predict_labels(const Input& input, Output& output, std::size_t labels) const {
        decltype(auto) layer = layer_get<I>();

        auto next_a = layer.template prepare_one_output<Input>();
        auto next_s = layer.template prepare_one_output<Input>();

        layer.activate_hidden(next_a, next_s, input, input);

        if(I == layers - 1){
            auto output_a = layer.prepare_one_input();
            auto output_s = layer.prepare_one_input();

            layer.activate_visible(next_a, next_s, output_a, output_s);

            output = std::move(output_a);
        } else {
            bool is_last = I == layers - 2;

            //If the next layers is the last layer
            if(is_last){
                auto big_next_a = layer.template prepare_one_output<layer_input_t<I, Input>>(is_last, labels);

                for(std::size_t i = 0; i < next_a.size(); ++i){
                    big_next_a[i] = next_a[i];
                }

                std::fill(big_next_a.begin() + dll::output_size(layer), big_next_a.end(), 0.1);

                predict_labels<I+1>(big_next_a, output, labels);
            } else {
                predict_labels<I+1>(next_a, output, labels);
            }
        }
    }

    //Stop recursion
    template<std::size_t I, typename Input, typename Output>
    std::enable_if_t<(I==layers)> predict_labels(const Input&, Output&, std::size_t) const {}

public:

    template<typename TrainingItem>
    size_t predict_labels(const TrainingItem& item_data, std::size_t labels) const {
        cpp_assert(dll::input_size(layer_get<layers - 1>()) == dll::output_size(layer_get<layers - 2>()) + labels, "There is no room for the labels units");

        typename layer_type<0>::input_one_t item(item_data);

        auto output_a = layer_get<layers - 1>().prepare_one_input();

        predict_labels<0>(item, output_a, labels);

        return std::distance(
            std::prev(output_a.end(), labels),
            std::max_element(std::prev(output_a.end(), labels), output_a.end()));
    }

    /*}}}*/

    /*{{{ Predict */

    //activation_probabilities

private:

    template<std::size_t I, std::size_t S = layers, typename Input, typename Result>
    std::enable_if_t<(I<S)> activation_probabilities(const Input& input, Result& result) const {
        static constexpr const bool multi_layer = layer_traits<layer_type<I>>::is_multiplex_layer();

        auto& layer = layer_get<I>();

        cpp::static_if<(I < S - 1 && !multi_layer)>([&](auto f){
            auto next_a = layer.template prepare_one_output<Input>();
            f(layer).activate_one(input, next_a);
            this->template activation_probabilities<I+1, S>(next_a, result);
        });

        cpp::static_if<(I < S - 1 && multi_layer)>([&](auto f){
            auto next_a = layer.template prepare_one_output<Input>();
            layer.activate_one(input, next_a);

            cpp_assert(f(result).empty(), "result must be empty on entry of activation_probabilities");

            f(result).reserve(next_a.size());

            for(std::size_t i = 0; i < next_a.size(); ++i){
                f(result).push_back(this->template layer_get<S-1>().template prepare_one_output<layer_input_t<I, Input>>());
                this->template activation_probabilities<I+1, S>(next_a[i], f(result)[i]);
            }
        });

        cpp::static_if<(I == S - 1)>([&](auto f){
            f(layer).activate_one(input, result);
        });
    }

    //Stop template recursion
    template<std::size_t I, std::size_t S = layers, typename Input, typename Result>
    std::enable_if_t<(I==S)> activation_probabilities(const Input&, Result&) const {}

public:

    template<typename Sample, typename Output>
    void activation_probabilities(const Sample& item_data, Output& result) const {
        sample_converter<this_type, 0, Sample> converter(*this, item_data);

        activation_probabilities<0>(converter.get(), result);
    }

    template<typename Sample, typename T = this_type, cpp_disable_if(dbn_traits<T>::is_multiplex())>
    auto activation_probabilities(const Sample& item_data) const {
        auto result = layer_get<layers - 1>().template prepare_one_output<Sample>();

        activation_probabilities(item_data, result);

        return result;
    }

    template<typename Sample, typename T = this_type, cpp_enable_if(dbn_traits<T>::is_multiplex())>
    auto activation_probabilities(const Sample& item_data) const {
        std::vector<typename layer_type<layers - 1>::output_one_t> result;

        activation_probabilities(item_data, result);

        return result;
    }

    template<std::size_t I, typename Sample, typename T = this_type, cpp_disable_if(dbn_traits<T>::is_multiplex())>
    auto activation_probabilities_sub(const Sample& item_data) const {
        auto result = layer_get<I>().template prepare_one_output<Sample>();

        activation_probabilities<0, I>(item_data, result);

        return result;
    }

    //full_activation_probabilities

private:

    template<std::size_t I, typename Input, typename Result>
    std::enable_if_t<(I<layers)> full_activation_probabilities(const Input& input, std::size_t& i, Result& result) const {
        auto& layer = layer_get<I>();

        auto next_s = layer.template prepare_one_output<Input>();
        auto next_a = layer.template prepare_one_output<Input>();

        layer.activate_one(input, next_a, next_s);

        for(auto& value : next_a){
            result[i++] = value;
        }

        full_activation_probabilities<I+1>(next_a, i, result);
    }

    //Stop template recursion
    template<std::size_t I, typename Input, typename Result>
    std::enable_if_t<(I==layers)> full_activation_probabilities(const Input&, std::size_t&, Result&) const {}

public:

    template<typename Sample, typename Output>
    void full_activation_probabilities(const Sample& item_data, Output& result) const {
        sample_converter<this_type, 0, Sample> converter(*this, item_data);

        std::size_t i = 0;

        full_activation_probabilities<0>(converter.get(), i, result);
    }

    template<typename Sample>
    etl::dyn_vector<weight> full_activation_probabilities(const Sample& item_data) const {
        etl::dyn_vector<weight> result(full_output_size());

        full_activation_probabilities(item_data, result);

        return result;
    }

    template<typename Sample, typename DBN = this_type, cpp::enable_if_u<dbn_traits<DBN>::concatenate()> = cpp::detail::dummy>
    auto get_final_activation_probabilities(const Sample& sample) const {
        return full_activation_probabilities(sample);
    }

    template<typename Sample, typename DBN = this_type, cpp::disable_if_u<dbn_traits<DBN>::concatenate()> = cpp::detail::dummy>
    auto get_final_activation_probabilities(const Sample& sample) const {
        return activation_probabilities(sample);
    }

    template<typename Weights>
    size_t predict_label(const Weights& result) const {
        return std::distance(result.begin(), std::max_element(result.begin(), result.end()));
    }

    template<typename Sample>
    size_t predict(const Sample& item) const {
        auto result = activation_probabilities(item);
        return predict_label(result);
    }

    /*}}}*/

    /*{{{ Fine-tuning */

    template<typename Samples, typename Labels>
    weight fine_tune(const Samples& training_data, Labels& labels, size_t max_epochs, size_t batch_size){
        return fine_tune(training_data.begin(), training_data.end(), labels.begin(), labels.end(), max_epochs, batch_size);
    }

    template<typename Iterator, typename LIterator>
    weight fine_tune(Iterator&& first, Iterator&& last, LIterator&& lfirst, LIterator&& llast, size_t max_epochs, size_t batch_size){
        dll::dbn_trainer<this_type> trainer;
        return trainer.train(*this,
            std::forward<Iterator>(first), std::forward<Iterator>(last),
            std::forward<LIterator>(lfirst), std::forward<LIterator>(llast),
            max_epochs, batch_size);
    }

    /*}}}*/

    using output_one_t = typename layer_type<layers - 1>::output_one_t;
    using output_t = typename layer_type<layers - 1>::output_one_t;

    //TODO This is broken if the last layer is a transform layer

    output_one_t prepare_one_output() const {
        return layer_get<layers - 1>().template prepare_one_output<typename layer_type<layers - 1>::input_one_t>();
    }

#ifdef DLL_SVM_SUPPORT

    /*{{{ SVM Training and prediction */

    using svm_samples_t = std::conditional_t<
        dbn_traits<this_type>::concatenate(),
        std::vector<etl::dyn_vector<weight>>,     //In full mode, use a simple 1D vector
        typename layer_type<layers - 1>::output_t>; //In normal mode, use the output of the last layer

private:

    template<typename DBN = this_type, typename Result, typename Sample, cpp::enable_if_u<dbn_traits<DBN>::concatenate()> = cpp::detail::dummy>
    void add_activation_probabilities(Result& result, const Sample& sample){
        result.emplace_back(full_output_size());
        full_activation_probabilities(sample, result.back());
    }

    template<typename DBN = this_type, typename Result, typename Sample, cpp::disable_if_u<dbn_traits<DBN>::concatenate()> = cpp::detail::dummy>
    void add_activation_probabilities(Result& result, const Sample& sample){
        result.push_back(layer_get<layers - 1>().template prepare_one_output<Sample>());
        activation_probabilities(sample, result.back());
    }

    template<typename Samples, typename Labels>
    void make_problem(const Samples& training_data, const Labels& labels, bool scale = false){
        svm_samples_t svm_samples;

        //Get all the activation probabilities
        for(auto& sample : training_data){
            add_activation_probabilities(svm_samples, sample);
        }

        //static_cast ensure using the correct overload
        problem = svm::make_problem(labels, static_cast<const svm_samples_t&>(svm_samples), scale);
    }

    template<typename Iterator, typename LIterator>
    void make_problem(Iterator first, Iterator last, LIterator&& lfirst, LIterator&& llast, bool scale = false){
        svm_samples_t svm_samples;

        //Get all the activation probabilities
        std::for_each(first, last, [this, &svm_samples](auto& sample){
            this->add_activation_probabilities(svm_samples, sample);
        });

        //static_cast ensure using the correct overload
        problem = svm::make_problem(
            std::forward<LIterator>(lfirst), std::forward<LIterator>(llast),
            svm_samples.begin(), svm_samples.end(),
            scale);
    }

public:

    template<typename Samples, typename Labels>
    bool svm_train(const Samples& training_data, const Labels& labels, const svm_parameter& parameters = default_svm_parameters()){
        cpp::stop_watch<std::chrono::seconds> watch;

        make_problem(training_data, labels, dbn_traits<this_type>::scale());

        //Make libsvm quiet
        svm::make_quiet();

        //Make sure parameters are not messed up
        if(!svm::check(problem, parameters)){
            return false;
        }

        //Train the SVM
        svm_model = svm::train(problem, parameters);

        svm_loaded = true;

        std::cout << "SVM training took " << watch.elapsed() << "s" << std::endl;

        return true;
    }

    template<typename Iterator, typename LIterator>
    bool svm_train(Iterator&& first, Iterator&& last, LIterator&& lfirst, LIterator&& llast, const svm_parameter& parameters = default_svm_parameters()){
        cpp::stop_watch<std::chrono::seconds> watch;

        make_problem(
            std::forward<Iterator>(first), std::forward<Iterator>(last),
            std::forward<LIterator>(lfirst), std::forward<LIterator>(llast),
            dbn_traits<this_type>::scale());

        //Make libsvm quiet
        svm::make_quiet();

        //Make sure parameters are not messed up
        if(!svm::check(problem, parameters)){
            return false;
        }

        //Train the SVM
        svm_model = svm::train(problem, parameters);

        svm_loaded = true;

        std::cout << "SVM training took " << watch.elapsed() << "s" << std::endl;

        return true;
    }

    template<typename Samples, typename Labels>
    bool svm_grid_search(const Samples& training_data, const Labels& labels, std::size_t n_fold = 5, const svm::rbf_grid& g = svm::rbf_grid()){
        make_problem(training_data, labels, dbn_traits<this_type>::scale());

        //Make libsvm quiet
        svm::make_quiet();

        auto parameters = default_svm_parameters();

        //Make sure parameters are not messed up
        if(!svm::check(problem, parameters)){
            return false;
        }

        //Perform a grid-search
        svm::rbf_grid_search(problem, parameters, n_fold, g);

        return true;
    }

    template<typename It, typename LIt>
    bool svm_grid_search(It&& first, It&& last, LIt&& lfirst, LIt&& llast, std::size_t n_fold = 5, const svm::rbf_grid& g = svm::rbf_grid()){
        make_problem(
            std::forward<It>(first), std::forward<It>(last),
            std::forward<LIt>(lfirst), std::forward<LIt>(llast),
            dbn_traits<this_type>::scale());

        //Make libsvm quiet
        svm::make_quiet();

        auto parameters = default_svm_parameters();

        //Make sure parameters are not messed up
        if(!svm::check(problem, parameters)){
            return false;
        }

        //Perform a grid-search
        svm::rbf_grid_search(problem, parameters, n_fold, g);

        return true;
    }

    template<typename Sample>
    double svm_predict(const Sample& sample){
        auto features = get_final_activation_probabilities(sample);
        return svm::predict(svm_model, features);
    }

    /*}}}*/

#endif //DLL_SVM_SUPPORT

};

} //end of namespace dll

#endif
