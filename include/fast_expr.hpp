//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DBN_FAST_EXPR_HPP
#define DBN_FAST_EXPR_HPP

#include "fast_op.hpp"
#include "utils.hpp"

template <typename T, typename LeftExpr, typename BinaryOp, typename RightExpr>
class fast_expr {
private:
    LeftExpr _lhs;
    RightExpr _rhs;

    typedef fast_expr<T, LeftExpr, BinaryOp, RightExpr> this_type;

public:
    //Cannot be constructed with no args
    fast_expr() = delete;

    //Construct a new expression
    fast_expr(LeftExpr l, RightExpr r) :
            _lhs(std::forward<LeftExpr>(l)), _rhs(std::forward<RightExpr>(r)){
        //Nothing else to init
    }

    //No copying
    fast_expr(const fast_expr&) = delete;
    fast_expr& operator=(const fast_expr&) = delete;

    //Make sure move is supported
    fast_expr(fast_expr&&) = default;
    fast_expr& operator=(fast_expr&&) = default;

    //Accessors

    typename std::add_lvalue_reference<LeftExpr>::type lhs(){
        return _lhs;
    }

    typename std::add_lvalue_reference<typename std::add_const<LeftExpr>::type>::type lhs() const {
        return _lhs;
    }

    typename std::add_lvalue_reference<RightExpr>::type rhs(){
        return _rhs;
    }

    typename std::add_lvalue_reference<typename std::add_const<RightExpr>::type>::type rhs() const {
        return _rhs;
    }

    //Create more complex expressions

    template<typename RE, typename = std::enable_if_t<std::is_convertible<RE, T>::value>>
    auto operator+(RE re) const -> fast_expr<T, this_type const&, plus_binary_op<T>, scalar<T>> {
        return {*this, re};
    }

    template<typename RE, typename = disable_if_t<std::is_convertible<RE, T>::value>>
    auto operator+(RE&& re) const -> fast_expr<T, this_type const&, plus_binary_op<T>, decltype(std::forward<RE>(re))>{
        return {*this, std::forward<RE>(re)};
    }

    template<typename RE, typename = std::enable_if_t<std::is_convertible<RE, T>::value>>
    auto operator-(RE re) const -> fast_expr<T, this_type const&, minus_binary_op<T>, scalar<T>> {
        return {*this, re};
    }

    template<typename RE, typename = disable_if_t<std::is_convertible<RE, T>::value>>
    auto operator-(RE&& re) const -> fast_expr<T, this_type const&, minus_binary_op<T>, decltype(std::forward<RE>(re))>{
        return {*this, std::forward<RE>(re)};
    }

    template<typename RE, typename = std::enable_if_t<std::is_convertible<RE, T>::value>>
    auto operator*(RE re) const -> fast_expr<T, this_type const&, mul_binary_op<T>, scalar<T>> {
        return {*this, re};
    }

    template<typename RE, typename = disable_if_t<std::is_convertible<RE, T>::value>>
    auto operator*(RE&& re) const -> fast_expr<T, this_type const&, mul_binary_op<T>, decltype(std::forward<RE>(re))>{
        return {*this, std::forward<RE>(re)};
    }

    template<typename RE, typename = std::enable_if_t<std::is_convertible<RE, T>::value>>
    auto operator/(RE re) const -> fast_expr<T, this_type const&, div_binary_op<T>, scalar<T>> {
        return {*this, re};
    }

    template<typename RE, typename = disable_if_t<std::is_convertible<RE, T>::value>>
    auto operator/(RE&& re) const -> fast_expr<T, this_type const&, div_binary_op<T>, decltype(std::forward<RE>(re))>{
        return {*this, std::forward<RE>(re)};
    }

    //Apply the expression

    decltype(auto) operator[](std::size_t i) const {
        return BinaryOp::apply(lhs()[i], rhs()[i]);
    }
};

#endif
