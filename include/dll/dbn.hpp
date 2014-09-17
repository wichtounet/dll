//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_DBN_HPP
#define DLL_DBN_HPP

#include <tuple>

#include "rbm.hpp"
#include "tuple_utils.hpp"
#include "dbn_trainer.hpp"
#include "conjugate_gradient.hpp"

namespace dll {

namespace dbn_detail {

template<typename W, typename Enable = void>
struct rbm_watcher_t {
    using type = void;
};

template<typename W>
struct rbm_watcher_t<W, enable_if_t<W::replace_sub> > {
    using type = W;
};

} //end of namespace dbn_detail

/*!
 * \brief A Deep Belief Network implementation
 */
template<typename Desc>
struct dbn {
    using desc = Desc;
    using this_type = dbn<desc>;

    using tuple_type = typename desc::layers::tuple_type;
    tuple_type tuples;

    static constexpr const std::size_t layers = desc::layers::layers;

    template <std::size_t N>
    using rbm_type = typename std::tuple_element<N, tuple_type>::type;

    using weight = typename rbm_type<0>::weight;

    weight learning_rate = 0.77;

    weight initial_momentum = 0.5;      ///< The initial momentum
    weight final_momentum = 0.9;        ///< The final momentum applied after *final_momentum_epoch* epoch
    weight final_momentum_epoch = 6;    ///< The epoch at which momentum change

    weight weight_cost = 0.0002;        ///< The weight cost for weight decay

    weight momentum = 0;                ///< The current momentum

    //No arguments by default
    dbn(){};

    //No copying
    dbn(const dbn& dbn) = delete;
    dbn& operator=(const dbn& dbn) = delete;

    //No moving
    dbn(dbn&& dbn) = delete;
    dbn& operator=(dbn&& dbn) = delete;

    void display() const {
        std::size_t parameters = 0;

        std::cout << "DBN with " << layers << " layers" << std::endl;

        detail::for_each(tuples, [&parameters](auto& rbm){
            typedef typename std::remove_reference<decltype(rbm)>::type rbm_t;
            constexpr const auto num_visible = rbm_t::num_visible;
            constexpr const auto num_hidden = rbm_t::num_hidden;

            parameters += num_visible * num_hidden;

            std::cout << "\tRBM: " << num_visible << "->" << num_hidden
                << ": " << (num_visible * num_hidden) << " parameters" << std::endl;
        });

        std::cout << "Total parameters: " << parameters << std::endl;
    }

    void store(std::ostream& os) const {
        detail::for_each(tuples, [&os](auto& rbm){
            rbm.store(os);
        });
    }

    void load(std::istream& is){
        detail::for_each(tuples, [&is](auto& rbm){
            rbm.load(is);
        });
    }

    template<std::size_t N>
    auto layer() -> typename std::add_lvalue_reference<rbm_type<N>>::type {
        return std::get<N>(tuples);
    }

    template<std::size_t N>
    constexpr auto layer() const -> typename std::add_lvalue_reference<typename std::add_const<rbm_type<N>>::type>::type {
        return std::get<N>(tuples);
    }

    template<std::size_t N>
    static constexpr std::size_t num_visible(){
        return rbm_type<N>::num_visible;
    }

    template<std::size_t N>
    static constexpr std::size_t num_hidden(){
        return rbm_type<N>::num_hidden;
    }

    /*{{{ Pretrain */

    /*!
     * \brief Pretrain the network by training all layers in an unsupervised
     * manner.
     */
    template<typename Samples>
    void pretrain(const Samples& training_data, std::size_t max_epochs){
        using training_t = std::vector<etl::dyn_vector<typename Samples::value_type::value_type>>;

        using watcher_t = typename desc::template watcher_t<this_type>;

        watcher_t watcher;

        watcher.pretraining_begin(*this);

        //Convert data to an useful form
        training_t data;
        data.reserve(training_data.size());

        for(auto& sample : training_data){
            data.emplace_back(sample);
        }

        training_t next_a;
        training_t next_s;

        auto input = std::ref(data);

        detail::for_each_i(tuples, [&watcher, this, &input, &next_a, &next_s, max_epochs](std::size_t I, auto& rbm){
            typedef typename std::remove_reference<decltype(rbm)>::type rbm_t;
            constexpr const auto num_hidden = rbm_t::num_hidden;

            auto input_size = static_cast<const training_t&>(input).size();

            //Train each layer but the last one
            if(I <= layers - 1 && rbm_t::hidden_unit != unit_type::EXP){
                watcher.template pretrain_layer<rbm_t>(*this, I, input_size);

                rbm.template train<
                        training_t,
                        !watcher_t::ignore_sub,                                 //Enable the RBM Watcher or not
                        typename dbn_detail::rbm_watcher_t<watcher_t>::type>    //Replace the RBM watcher if not void
                    (static_cast<const training_t&>(input), max_epochs);

                //Get the activation probabilities for the next level
                if(I < layers - 1){
                    next_a.clear();
                    next_a.reserve(input_size);
                    next_s.clear();
                    next_s.reserve(input_size);

                    for(std::size_t i = 0; i < input_size; ++i){
                        next_a.emplace_back(num_hidden);
                        next_s.emplace_back(num_hidden);
                    }

                    for(size_t i = 0; i < input_size; ++i){
                        rbm.activate_hidden(next_a[i], next_s[i], static_cast<const training_t&>(input)[i], static_cast<const training_t&>(input)[i]);
                    }

                    input = std::ref(next_a);
                }
            }
        });

        watcher.pretraining_end(*this);
    }

    /*}}}*/

    /*{{{ With labels */

    template<typename Samples, typename Labels>
    void train_with_labels(const Samples& training_data, const Labels& training_labels, std::size_t labels, std::size_t max_epochs){
        dll_assert(training_data.size() == training_labels.size(), "There must be the same number of values than labels");
        dll_assert(num_visible<layers - 1>() == num_hidden<layers - 2>() + labels, "There is no room for the labels units");

        using training_t = std::vector<etl::dyn_vector<typename Samples::value_type::value_type>>;

        //Convert data to an useful form
        training_t data;
        data.reserve(training_data.size());

        for(auto& sample : training_data){
            data.emplace_back(sample);
        }

        auto input = std::cref(data);

        detail::for_each_i(tuples, [&input, &training_labels, labels, max_epochs](size_t I, auto& rbm){
            typedef typename std::remove_reference<decltype(rbm)>::type rbm_t;
            constexpr const auto num_hidden = rbm_t::num_hidden;

            static training_t next;

            next.reserve(static_cast<const training_t&>(input).size());

            rbm.train(static_cast<const training_t&>(input), max_epochs);

            if(I < layers - 1){
                auto append_labels = (I + 1 == layers - 1);

                for(auto& training_item : static_cast<const training_t&>(input)){
                    etl::dyn_vector<weight> next_item_a(num_hidden + (append_labels ? labels : 0));
                    etl::dyn_vector<weight> next_item_s(num_hidden + (append_labels ? labels : 0));
                    rbm.activate_hidden(next_item_a, next_item_s, training_item, training_item);
                    next.emplace_back(std::move(next_item_a));
                }

                //If the next layers is the last layer
                if(append_labels){
                    for(size_t i = 0; i < training_labels.size(); ++i){
                        auto label = training_labels[i];

                        for(size_t l = 0; l < labels; ++l){
                            next[i][num_hidden + l] = label == l ? 1.0 : 0.0;
                        }
                    }
                }
            }

            input = std::cref(next);
        });
    }

    template<typename TrainingItem>
    size_t predict_labels(const TrainingItem& item_data, std::size_t labels){
        dll_assert(num_visible<layers - 1>() == num_hidden<layers - 2>() + labels, "There is no room for the labels units");

        etl::dyn_vector<typename TrainingItem::value_type> item(item_data);

        etl::dyn_vector<weight> output_a(num_visible<layers - 1>());
        etl::dyn_vector<weight> output_s(num_visible<layers - 1>());

        auto input_ref = std::cref(item);

        detail::for_each_i(tuples, [labels,&input_ref,&output_a,&output_s](size_t I, auto& rbm){
            typedef typename std::remove_reference<decltype(rbm)>::type rbm_t;
            constexpr const auto num_hidden = rbm_t::num_hidden;

            auto& input = static_cast<const etl::dyn_vector<weight>&>(input_ref);

            if(I == layers - 1){
                static etl::dyn_vector<weight> h1_a(num_hidden);
                static etl::dyn_vector<weight> h1_s(num_hidden);

                rbm.activate_hidden(h1_a, h1_s, input, input);
                rbm.activate_visible(h1_a, h1_s, output_a, output_s);
            } else {
                static etl::dyn_vector<weight> next_a(num_hidden);
                static etl::dyn_vector<weight> next_s(num_hidden);
                static etl::dyn_vector<weight> big_next_a(num_hidden + labels);

                rbm.activate_hidden(next_a, next_s, input, input);

                //If the next layers is the last layer
                if(I + 1 == layers - 1){
                    for(std::size_t i = 0; i < next_a.size(); ++i){
                        big_next_a[i] = next_a[i];
                    }

                    for(size_t l = 0; l < labels; ++l){
                        big_next_a[num_hidden + l] = 0.1;
                    }

                    input_ref = std::cref(big_next_a);
                } else {
                    input_ref = std::cref(next_a);
                }

            }
        });

        size_t label = 0;
        weight max = 0;
        for(size_t l = 0; l < labels; ++l){
            auto value = output_a[num_visible<layers - 1>() - labels + l];

            if(value > max){
                max = value;
                label = l;
            }
        }

        return label;
    }

    /*}}}*/

    /*{{{ Predict */

    template<typename Sample, typename Output>
    void predict_weights(const Sample& item_data, Output& result){
        etl::dyn_vector<typename Sample::value_type> item(item_data);

        auto input = std::cref(item);

        detail::for_each_i(tuples, [&item, &input, &result](std::size_t I, auto& rbm){
            if(I != layers - 1){
                typedef typename std::remove_reference<decltype(rbm)>::type rbm_t;
                constexpr const auto num_hidden = rbm_t::num_hidden;

                static etl::dyn_vector<weight> next(num_hidden);
                static etl::dyn_vector<weight> next_s(num_hidden);

                rbm.activate_hidden(next, next_s, static_cast<const Sample&>(input), static_cast<const Sample&>(input));

                input = std::cref(next);
            }
        });

        constexpr const auto num_hidden = rbm_type<layers - 1>::num_hidden;

        static etl::dyn_vector<weight> next_s(num_hidden);

        layer<layers - 1>().activate_hidden(result, next_s, static_cast<const Sample&>(input), static_cast<const Sample&>(input));
    }

    template<typename Sample>
    etl::dyn_vector<weight> predict_weights(const Sample& item_data){
        etl::dyn_vector<weight> result(num_hidden<layers - 1>());

        etl::dyn_vector<typename Sample::value_type> item(item_data);

        auto input = std::cref(item);

        detail::for_each_i(tuples, [&item, &input, &result](std::size_t I, auto& rbm){
            typedef typename std::remove_reference<decltype(rbm)>::type rbm_t;
            constexpr const auto num_hidden = rbm_t::num_hidden;

            static etl::dyn_vector<weight> next(num_hidden);
            static etl::dyn_vector<weight> next_s(num_hidden);
            auto& output = (I == layers - 1) ? result : next;

            rbm.activate_hidden(output, next_s, static_cast<const Sample&>(input), static_cast<const Sample&>(input));

            input = std::cref(next);
        });

        return result;
    }

    template<typename Weights>
    size_t predict_final(const Weights& result){
        size_t label = 0;
        weight max = 0;
        for(size_t l = 0; l < result.size(); ++l){
            auto value = result[l];

            if(value > max){
                max = value;
                label = l;
            }
        }

        return label;
    }

    template<typename Sample>
    size_t predict(const Sample& item){
        auto result = predict_weights(item);
        return predict_final(result);;
    }

    /*}}}*/

    template<typename Samples, typename Labels>
    weight fine_tune(const Samples& training_data, Labels& labels, size_t max_epochs, size_t batch_size){
        dll::dbn_trainer<this_type> trainer;
        return trainer.train(*this, training_data, labels, max_epochs, batch_size);
    }
};

} //end of namespace dll

#endif
