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

template< bool B, class T = void >
using enable_if_t = typename std::enable_if<B, T>::type;

namespace dbn {

template<typename Conf, typename... Layers>
struct dbn {
private:
    typedef std::tuple<rbm<Layers, Conf>...> tuple_type;
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

    template<typename TrainingItem, std::size_t I>
    inline enable_if_t<(I == layers - 1), void> train_rbm_layers(const std::vector<std::vector<TrainingItem>>& training_data, std::size_t max_epochs){
        std::get<I>(tuples).train(training_data, max_epochs);
    }

    template<typename TrainingItem, std::size_t I>
    inline enable_if_t<(I < layers - 1), void> train_rbm_layers(const std::vector<std::vector<TrainingItem>>& training_data, std::size_t max_epochs){
        auto& rbm = layer<I>();
        if(I == 0){
            rbm.train(training_data, max_epochs);
        } else {
            //TODO Train with data of previous layer
            //rbm.train(training_data, max_epochs);
        }

        train_rbm_layers<TrainingItem, I + 1>(training_data, max_epochs);
    }

    template<typename TrainingItem>
    void train(const std::vector<std::vector<TrainingItem>>& training_data, std::size_t max_epochs){
        train_rbm_layers<TrainingItem, 0>(training_data, max_epochs);
    }
};

} //end of namespace dbn

#endif
