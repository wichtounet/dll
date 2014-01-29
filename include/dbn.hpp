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
    constexpr auto layer() -> typename std::add_lvalue_reference<rbm_type<N>>::type {
        return std::get<N>(tuples);
    }

    template<std::size_t N>
    constexpr auto layer() const -> typename std::add_const<typename std::add_lvalue_reference<rbm_type<N>>::type>::type {
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
};

} //end of namespace dbn

#endif
