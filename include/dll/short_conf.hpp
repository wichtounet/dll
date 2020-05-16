//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Shortcut to configuration
 */

#pragma once

namespace dll {

/*!
 * \brief Specify that a layer does not use an activation function
 */
using no_activation = activation<function::IDENTITY>;

/*!
 * \brief Specify that a layer uses a sigmoid activation function
 */
using sigmoid = activation<function::SIGMOID>;

/*!
 * \brief Specify that a layer uses a relu activation function
 */
using relu = activation<function::RELU>;

/*!
 * \brief Specify that a layer uses a softmax activation function
 */
using softmax = activation<function::SOFTMAX>;

/*!
 * \brief Specify that a layer uses a identity activation function
 */
using identity = activation<function::IDENTITY>;

/*!
 * \brief Specify that a layer uses a tanh activation function
 */
using tanh = activation<function::TANH>;

/*!
 * \brief Specify that a network uses the categorical cross entropy loss
 * function.
 */
using categorical_cross_entropy = loss<loss_function::CATEGORICAL_CROSS_ENTROPY>;

/*!
 * \brief Specify that a network uses the binary cross entropy loss
 * function.
 */
using binary_cross_entropy = loss<loss_function::BINARY_CROSS_ENTROPY>;

/*!
 * \brief Specify that a network uses the mean squared error loss
 * function.
 */
using mean_squared_error = loss<loss_function::MEAN_SQUARED_ERROR>;

/*!
 * \brief Specify that a network uses the ADADELTA updater for
 * gradient descent.
 */
using adadelta = updater<updater_type::ADADELTA>;

/*!
 * \brief Specify that a network uses the ADAM updater for
 * gradient descent.
 */
using adam = updater<updater_type::ADAM>;

/*!
 * \brief Specify that a network uses the NADAM updater for
 * gradient descent.
 */
using nadam = updater<updater_type::NADAM>;

/*!
 * \brief Specify that the network should not output anything
 */
using silent = dll::output_policy<dll::null_output_policy>;

} //end of dll namespace
