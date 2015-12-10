//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file function.hpp
 * \brief Activation functions for neural networks
 */

#ifndef DLL_FUNCTION_HPP
#define DLL_FUNCTION_HPP

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

template <function F, typename E, cpp_enable_if(F == function::IDENTITY)>
decltype(auto) f_activate(E&& expr) {
    return etl::identity(std::forward<E>(expr));
}

template <function F, typename E, cpp_enable_if(F == function::SIGMOID)>
decltype(auto) f_activate(E&& expr) {
    return etl::sigmoid(std::forward<E>(expr));
}

template <function F, typename E, cpp_enable_if(F == function::TANH)>
decltype(auto) f_activate(E&& expr) {
    return etl::tanh(std::forward<E>(expr));
}

template <function F, typename E, cpp_enable_if(F == function::RELU)>
decltype(auto) f_activate(E&& expr) {
    return etl::relu(std::forward<E>(expr));
}

template <function F, typename E, cpp_enable_if(F == function::SOFTMAX)>
decltype(auto) f_activate(E&& expr) {
    return etl::softmax(std::forward<E>(expr));
}

template <function F, typename E, cpp_enable_if(F == function::IDENTITY)>
decltype(auto) f_derivative(E&& expr) {
    return etl::identity_derivative(std::forward<E>(expr));
}

template <function F, typename E, cpp_enable_if(F == function::SIGMOID)>
decltype(auto) f_derivative(E&& expr) {
    return etl::sigmoid_derivative(std::forward<E>(expr));
}

template <function F, typename E, cpp_enable_if(F == function::TANH)>
decltype(auto) f_derivative(E&& expr) {
    return etl::tanh_derivative(std::forward<E>(expr));
}

template <function F, typename E, cpp_enable_if(F == function::RELU)>
decltype(auto) f_derivative(E&& expr) {
    return etl::relu_derivative(std::forward<E>(expr));
}

template <function F, typename E, cpp_enable_if(F == function::SOFTMAX)>
decltype(auto) f_derivative(E&& expr) {
    return etl::softmax_derivative(std::forward<E>(expr));
}

} //end of dll namespace

#endif
