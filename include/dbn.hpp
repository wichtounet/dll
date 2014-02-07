//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DBN_DBN_HPP
#define DBN_DBN_HPP

#include <tuple>

#include "rbm.hpp"
#include "vector.hpp"

template< bool B, class T = void >
using enable_if_t = typename std::enable_if<B, T>::type;

namespace dbn {

template<typename... Layers>
struct dbn {
private:
    typedef std::tuple<rbm<Layers>...> tuple_type;
    tuple_type tuples;

    template <std::size_t N>
    using rbm_type = typename std::tuple_element<N, tuple_type>::type;

    static constexpr const std::size_t layers = sizeof...(Layers);

public:
    template<std::size_t N>
    auto layer() -> typename std::add_lvalue_reference<rbm_type<N>>::type {
        return std::get<N>(tuples);
    }

    template<std::size_t N>
    constexpr auto layer() const -> typename std::add_lvalue_reference<typename std::add_const<rbm_type<N>>::type>::type {
        return std::get<N>(tuples);
    }

    template<std::size_t N>
    constexpr std::size_t num_visible() const {
        return rbm_type<N>::num_visible;
    }

    template<std::size_t N>
    constexpr std::size_t num_hidden() const {
        return rbm_type<N>::num_hidden;
    }

    template<std::size_t I, typename TrainingItems, typename LabelItems>
    inline enable_if_t<(I == layers - 1), void>
    train_rbm_layers(const TrainingItems& training_data, std::size_t max_epochs, const LabelItems&, std::size_t){
        std::cout << "Train layer " << I << std::endl;

        std::get<I>(tuples).train(training_data, max_epochs);
    }

    template<std::size_t I, typename TrainingItems, typename LabelItems>
    inline enable_if_t<(I < layers - 1), void>
    train_rbm_layers(const TrainingItems& training_data, std::size_t max_epochs, const LabelItems& training_labels = {}, std::size_t labels = 0){
        std::cout << "Train layer " << I << std::endl;

        auto& rbm = layer<I>();

        rbm.train(training_data, max_epochs);

        auto append_labels = I + 1 == layers - 1 && !training_labels.empty();

        std::vector<vector<double>> next;
        next.reserve(training_data.size());

        for(auto& training_item : training_data){
            vector<double> next_item(num_hidden<I>() + (append_labels ? labels : 0));
            rbm.activate_hidden(next_item, training_item);
            next.emplace_back(std::move(next_item));
        }

        //If the next layers is the last layer
        if(append_labels){
            for(size_t i = 0; i < training_labels.size(); ++i){
                auto label = training_labels[i];

                for(size_t l = 0; l < labels; ++l){
                    if(label == l){
                        next[i][num_hidden<I>() + l] = 1;
                    } else {
                        next[i][num_hidden<I>() + l] = 0;
                    }
                }
            }
        }

        train_rbm_layers<I + 1>(next, max_epochs, training_labels, labels);
    }

    template<typename TrainingItem>
    void train(const std::vector<std::vector<TrainingItem>>& training_data, std::size_t max_epochs){
        train_rbm_layers<0>(training_data, max_epochs);
    }

    template<typename TrainingItem, typename Label>
    void train_with_labels(const std::vector<TrainingItem>& training_data, const std::vector<Label>& training_labels, std::size_t labels, std::size_t max_epochs){
        dbn_assert(training_data.size() == training_labels.size(), "There must be the same number of values than labels");
        dbn_assert(num_visible<layers - 1>() == num_hidden<layers - 2>() + labels, "There is no room for the labels units");

        train_rbm_layers<0>(training_data, max_epochs, training_labels, labels);
    }

    template<std::size_t I, typename TrainingItem, typename Output>
    inline enable_if_t<(I == layers - 1), void>
    activate_layers(const TrainingItem& input, size_t, Output& output){
        auto& rbm = layer<I>();

        static vector<double> h1(num_hidden<I>());
        static vector<double> hs(num_hidden<I>());

        rbm.activate_hidden(h1, input);
        rbm.activate_visible(rbm_type<I>::bernoulli(h1, hs), output);
    }

    template<std::size_t I, typename TrainingItem, typename Output>
    inline enable_if_t<(I < layers - 1), void>
    activate_layers(const TrainingItem& input, std::size_t labels, Output& output){
        auto& rbm = layer<I>();

        static vector<double> next(num_visible<I+1>());

        rbm.activate_hidden(next, input);

        //If the next layers is the last layer
        if(I + 1 == layers - 1){
            for(size_t l = 0; l < labels; ++l){
                next[num_hidden<I>() + l] = 0.1;
            }
        }

        activate_layers<I + 1>(next, labels, output);
    }

    template<typename TrainingItem>
    size_t predict(TrainingItem& item, std::size_t labels){
        dbn_assert(num_visible<layers - 1>() == num_hidden<layers - 2>() + labels, "There is no room for the labels units");

        static vector<double> output(num_visible<layers - 1>());

        activate_layers<0>(item, labels, output);

        size_t label = 0;
        double max = 0;
        for(size_t l = 0; l < labels; ++l){
            auto value = output[num_visible<layers - 1>() - labels + l];

            if(value > max){
                max = value;
                label = l;
            }
        }

        return label;
    }

    template<std::size_t I, typename TrainingItem, typename Output>
    inline enable_if_t<(I == layers - 1), void>
    deep_activate_layers(const TrainingItem& input, size_t, Output& output, std::size_t sampling){
        auto& rbm = layer<I>();

        static vector<double> v1(num_visible<I>());
        static vector<double> v2(num_visible<I>());

        for(size_t i = 0; i < input.size(); ++i){
            v1(i) = input[i];
        }

        static vector<double> h1(num_hidden<I>());
        static vector<double> h2(num_hidden<I>());
        static vector<double> hs(num_hidden<I>());

        for(size_t i = 0; i< sampling; ++i){
            rbm.activate_hidden(h1, v1);
            rbm.activate_visible(rbm_type<I>::bernoulli(h1, hs), v1);

            //TODO Perhaps we should apply a new bernoulli on v1 ?
        }

        rbm.activate_hidden(h1, input);
        rbm.activate_visible(rbm_type<I>::bernoulli(h1, hs), output);
    }

    template<std::size_t I, typename TrainingItem, typename Output>
    inline enable_if_t<(I < layers - 1), void>
    deep_activate_layers(const TrainingItem& input, std::size_t labels, Output& output, std::size_t sampling){
        auto& rbm = layer<I>();

        static vector<double> next(num_visible<I+1>());

        rbm.activate_hidden(next, input);

        //If the next layers is the last layer
        if(I + 1 == layers - 1){
            for(size_t l = 0; l < labels; ++l){
                next[num_hidden<I>() + l] = 0.1;
            }
        }

        deep_activate_layers<I + 1>(next, labels, output, sampling);
    }

    template<typename TrainingItem>
    size_t deep_predict(TrainingItem& item, std::size_t labels, std::size_t sampling){
        dbn_assert(num_visible<layers - 1>() == num_hidden<layers - 2>() + labels, "There is no room for the labels units");

        vector<double> output(num_visible<layers - 1>());
        deep_activate_layers<0>(item, labels, output, sampling);

        size_t label = 0;
        double max = 0;
        for(size_t l = 0; l < labels; ++l){
            auto value = output[num_visible<layers - 1>() - labels + l];

            if(value > max){
                max = value;
                label = l;
            }
        }

        return label;
    }
};

} //end of namespace dbn

#endif
