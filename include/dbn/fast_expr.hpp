//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DBN_binary_expr_HPP
#define DBN_binary_expr_HPP

#include "fast_op.hpp"
#include "utils.hpp"

template <typename T, typename LeftExpr, typename BinaryOp, typename RightExpr>
class binary_expr {
private:
    LeftExpr _lhs;
    RightExpr _rhs;

    typedef binary_expr<T, LeftExpr, BinaryOp, RightExpr> this_type;

public:
    //Cannot be constructed with no args
    binary_expr() = delete;

    //Construct a new expression
    binary_expr(LeftExpr l, RightExpr r) :
            _lhs(std::forward<LeftExpr>(l)), _rhs(std::forward<RightExpr>(r)){
        //Nothing else to init
    }

    //No copying
    binary_expr(const binary_expr&) = delete;
    binary_expr& operator=(const binary_expr&) = delete;

    //Make sure move is supported
    binary_expr(binary_expr&&) = default;
    binary_expr& operator=(binary_expr&&) = default;

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

    template<typename RE, typename = enable_if_t<std::is_convertible<RE, T>::value>>
    auto operator+(RE re) const -> binary_expr<T, this_type const&, plus_binary_op<T>, scalar<T>> {
        return {*this, re};
    }

    template<typename RE, typename = disable_if_t<std::is_convertible<RE, T>::value>>
    auto operator+(RE&& re) const -> binary_expr<T, this_type const&, plus_binary_op<T>, decltype(std::forward<RE>(re))>{
        return {*this, std::forward<RE>(re)};
    }

    template<typename RE, typename = enable_if_t<std::is_convertible<RE, T>::value>>
    auto operator-(RE re) const -> binary_expr<T, this_type const&, minus_binary_op<T>, scalar<T>> {
        return {*this, re};
    }

    template<typename RE, typename = disable_if_t<std::is_convertible<RE, T>::value>>
    auto operator-(RE&& re) const -> binary_expr<T, this_type const&, minus_binary_op<T>, decltype(std::forward<RE>(re))>{
        return {*this, std::forward<RE>(re)};
    }

    template<typename RE, typename = disable_if_t<std::is_convertible<RE, T>::value>>
    auto operator*(RE&& re) const -> binary_expr<T, this_type const&, mul_binary_op<T>, decltype(std::forward<RE>(re))>{
        return {*this, std::forward<RE>(re)};
    }

    template<typename RE, typename = enable_if_t<std::is_convertible<RE, T>::value>>
    auto operator/(RE re) const -> binary_expr<T, this_type const&, div_binary_op<T>, scalar<T>> {
        return {*this, re};
    }

    template<typename RE, typename = disable_if_t<std::is_convertible<RE, T>::value>>
    auto operator/(RE&& re) const -> binary_expr<T, this_type const&, div_binary_op<T>, decltype(std::forward<RE>(re))>{
        return {*this, std::forward<RE>(re)};
    }

    //Apply the expression

    decltype(auto) operator[](std::size_t i) const {
        return BinaryOp::apply(lhs()[i], rhs()[i]);
    }
};

template <typename T, typename LE, typename Op, typename RE, typename Scalar, typename = enable_if_t<std::is_convertible<Scalar, T>::value>>
auto operator*(const binary_expr<T, LE, Op, RE>& lhs, Scalar rhs) -> binary_expr<T, const binary_expr<T, LE, Op, RE>&, mul_binary_op<T>, scalar<T>> {
    return {lhs, rhs};
}

template <typename T, typename LE, typename Op, typename RE, typename Scalar, typename = enable_if_t<std::is_convertible<Scalar, T>::value>>
auto operator*(Scalar lhs, const binary_expr<T, LE, Op, RE>& rhs) -> binary_expr<T, scalar<T>, mul_binary_op<T>, const binary_expr<T, LE, Op, RE>&> {
    return {lhs, rhs};
}

template <typename T, typename Expr, typename UnaryOp>
class unary_expr {
private:
    Expr _value;

    typedef unary_expr<T, Expr, UnaryOp> this_type;

public:
    //Cannot be constructed with no args
    unary_expr() = delete;

    //Construct a new expression
    unary_expr(Expr l) : _value(std::forward<Expr>(l)){
        //Nothing else to init
    }

    //No copying
    unary_expr(const unary_expr&) = delete;
    unary_expr& operator=(const unary_expr&) = delete;

    //Make sure move is supported
    unary_expr(unary_expr&&) = default;
    unary_expr& operator=(unary_expr&&) = default;

    //Accessors

    typename std::add_lvalue_reference<Expr>::type value(){
        return _value;
    }

    typename std::add_lvalue_reference<typename std::add_const<Expr>::type>::type value() const {
        return _value;
    }

    //Apply the expression

    decltype(auto) operator[](std::size_t i) const {
        return UnaryOp::apply(value()[i]);
    }
};

//Convert x * unary_expr and unary_expr * x into binary_expr

template <typename T, typename E, typename Op, typename Scalar, typename = enable_if_t<std::is_convertible<Scalar, T>::value>>
auto operator*(const unary_expr<T, E, Op>& lhs, Scalar rhs) -> binary_expr<T, const unary_expr<T, E, Op>&, mul_binary_op<T>, scalar<T>> {
    return {lhs, rhs};
}

template <typename T, typename E, typename Op, typename Scalar, typename = enable_if_t<std::is_convertible<Scalar, T>::value>>
auto operator*(Scalar lhs, const unary_expr<T, E, Op>& rhs) -> binary_expr<T, scalar<T>, mul_binary_op<T>, const unary_expr<T, E, Op>&> {
    return {lhs, rhs};
}


#endif
