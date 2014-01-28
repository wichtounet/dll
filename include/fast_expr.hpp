//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DBN_FAST_EXPR_HPP
#define DBN_FAST_EXPR_HPP

#include "fast_op.hpp"

template <typename T, typename LeftExpr, typename BinaryOp, typename RightExpr>
class fast_vector_expr {
private:
    LeftExpr _lhs;
    RightExpr _rhs;

    typedef fast_vector_expr<T, LeftExpr, BinaryOp, RightExpr> this_type;

public:
    fast_vector_expr(LeftExpr l, RightExpr r) :
            _lhs(std::forward<LeftExpr>(l)), _rhs(std::forward<RightExpr>(r)){
        //Nothing else to init
    }

    //No copying
    fast_vector_expr(const fast_vector_expr&) = delete;
    fast_vector_expr& operator=(const fast_vector_expr&) = delete;

    //Make sure move is supported
    fast_vector_expr(fast_vector_expr&&) = default;
    fast_vector_expr& operator=(fast_vector_expr&&) = default;

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

    auto operator+(T re) const -> fast_vector_expr<T, this_type const&, plus_binary_op<T>, scalar<T>> {
        return {*this, re};
    }

    template<typename RE>
    auto operator+(RE&& re) const -> fast_vector_expr<T, this_type const&, plus_binary_op<T>, decltype(std::forward<RE>(re))>{
        return {*this, std::forward<RE>(re)};
    }

    auto operator-(T re) const -> fast_vector_expr<T, this_type const&, minus_binary_op<T>, scalar<T>> {
        return {*this, re};
    }

    template<typename RE>
    auto operator-(RE&& re) const -> fast_vector_expr<T, this_type const&, minus_binary_op<T>, decltype(std::forward<RE>(re))>{
        return {*this, std::forward<RE>(re)};
    }

    auto operator*(T re) const -> fast_vector_expr<T, this_type const&, mul_binary_op<T>, scalar<T>> {
        return {*this, re};
    }

    template<typename RE>
    auto operator*(RE&& re) const -> fast_vector_expr<T, this_type const&, mul_binary_op<T>, decltype(std::forward<RE>(re))>{
        return {*this, std::forward<RE>(re)};
    }

    auto operator/(T re) const -> fast_vector_expr<T, this_type const&, div_binary_op<T>, scalar<T>> {
        return {*this, re};
    }

    template<typename RE>
    auto operator/(RE&& re) const -> fast_vector_expr<T, this_type const&, div_binary_op<T>, decltype(std::forward<RE>(re))>{
        return {*this, std::forward<RE>(re)};
    }

    //Apply the expression

    auto operator[](std::size_t i) const -> decltype(BinaryOp::apply(this->lhs()[i], this->rhs()[i])) {
        return BinaryOp::apply(lhs()[i], rhs()[i]);
    }
};

#endif
