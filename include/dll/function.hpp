//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file function.hpp
 * \brief Activation functions for neural networks
 */

#pragma once

namespace dll {

/*!
 * \brief An activation function
 */
enum class function {
    IDENTITY, ///< Identity activation function
    SIGMOID,  ///< Sigmoid activation function
    TANH,     ///< Hyperbolic tangent
    RELU,     ///< Rectified Linear Unit
    SOFTMAX   ///< Softmax
};

/*!
 * \brief Returns a string representation of an activation function
 * \param f The function to transform to string
 * \return a string representation of an activation function
 */
inline std::string to_string(function f) {
    switch (f) {
        case function::IDENTITY:
            return "IDENTITY";
        case function::SIGMOID:
            return "SIGMOID";
        case function::TANH:
            return "TANH";
        case function::RELU:
            return "RELU";
        case function::SOFTMAX:
            return "SOFTMAX";
    }

    cpp_unreachable("Unreachable code");

    return "UNDEFINED";
}

/*!
 * \brief Computes the activations from the given input using the specified activation function
 * \param expr The input expression
 * \tparam F The activation function to use
 * \return The result of the activation function
 */
template <function F, typename E, cpp_enable_iff(F == function::IDENTITY)>
decltype(auto) f_activate(E&& expr) {
    return etl::identity(std::forward<E>(expr));
}

/*!
 * \copydoc f_activate
 */
template <function F, typename E, cpp_enable_iff(F == function::SIGMOID)>
decltype(auto) f_activate(E&& expr) {
    return etl::sigmoid(std::forward<E>(expr));
}

/*!
 * \copydoc f_activate
 */
template <function F, typename E, cpp_enable_iff(F == function::TANH)>
decltype(auto) f_activate(E&& expr) {
    return etl::tanh(std::forward<E>(expr));
}

/*!
 * \copydoc f_activate
 */
template <function F, typename E, cpp_enable_iff(F == function::RELU)>
decltype(auto) f_activate(E&& expr) {
    return etl::relu(std::forward<E>(expr));
}

/*!
 * \copydoc f_activate
 */
template <function F, typename E, cpp_enable_iff(F == function::SOFTMAX)>
decltype(auto) f_activate(E&& expr) {
    return etl::stable_softmax(std::forward<E>(expr));
}

/*!
 * \brief Computes the derivatives from the given output using the specified activation function
 * \param expr The input expression
 * \tparam F The activation function to use
 * \return The derivative of the activation function
 */
template <function F, typename E, cpp_enable_iff(F == function::IDENTITY)>
decltype(auto) f_derivative(E&& expr) {
    return etl::ml::identity_derivative_out(std::forward<E>(expr));
}

/*!
 * \copydoc f_derivative
 */
template <function F, typename E, cpp_enable_iff(F == function::SIGMOID)>
decltype(auto) f_derivative(E&& expr) {
    return etl::ml::sigmoid_derivative_out(std::forward<E>(expr));
}

/*!
 * \copydoc f_derivative
 */
template <function F, typename E, cpp_enable_iff(F == function::TANH)>
decltype(auto) f_derivative(E&& expr) {
    return etl::ml::tanh_derivative_out(std::forward<E>(expr));
}

/*!
 * \copydoc f_derivative
 */
template <function F, typename E, cpp_enable_iff(F == function::RELU)>
decltype(auto) f_derivative(E&& expr) {
    return etl::ml::relu_derivative_out(std::forward<E>(expr));
}

/*!
 * \copydoc f_derivative
 */
template <function F, typename E, cpp_enable_iff(F == function::SOFTMAX)>
decltype(auto) f_derivative(E&& expr) {
    return etl::ml::softmax_derivative_out(std::forward<E>(expr));
}

} //end of dll namespace
