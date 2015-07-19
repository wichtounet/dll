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
    IDENTITY,    ///< Identity activation function
    SIGMOID,     ///< Sigmoid activation function
    TANH,        ///< Hyperbolic tangent
    RELU         ///< Rectified Linear Unit
};

inline std::string to_string(function f){
    switch(f){
        case function::IDENTITY:
            return "IDENTITY";
        case function::SIGMOID:
            return "SIGMOID";
        case function::TANH:
            return "TANH";
        case function::RELU:
            return "RELU";
    }

    cpp_unreachable("Unreachable code");

    return "UNDEFINED";
}

} //end of dll namespace

#endif
