//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

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
auto f_activate(E&& expr) {
    return etl::identity(expr);
}

template <function F, typename E, cpp_enable_if(F == function::SIGMOID)>
auto f_activate(E&& expr) {
    return etl::sigmoid(expr);
}

template <function F, typename E, cpp_enable_if(F == function::TANH)>
auto f_activate(E&& expr) {
    return etl::tanh(expr);
}

template <function F, typename E, cpp_enable_if(F == function::RELU)>
auto f_activate(E&& expr) {
    return etl::relu(expr);
}

template <function F, typename E, cpp_enable_if(F == function::SOFTMAX)>
auto f_activate(E&& expr) {
    return etl::softmax(expr);
}

template <function F, typename E, cpp_enable_if(F == function::IDENTITY)>
auto f_derivative(E&& expr) {
    return etl::identity_derivative(expr);
}

template <function F, typename E, cpp_enable_if(F == function::SIGMOID)>
auto f_derivative(E&& expr) {
    return etl::sigmoid_derivative(expr);
}

template <function F, typename E, cpp_enable_if(F == function::TANH)>
auto f_derivative(E&& expr) {
    return etl::tanh_derivative(expr);
}

template <function F, typename E, cpp_enable_if(F == function::RELU)>
auto f_derivative(E&& expr) {
    return etl::relu_derivative(expr);
}

template <function F, typename E, cpp_enable_if(F == function::SOFTMAX)>
auto f_derivative(E&&) {
    return 1.0;
}

} //end of dll namespace

#endif
