//=======================================================================
// Copyright Baptiste Wicht 2014.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//=======================================================================

#ifndef DBN_RBM_HPP
#define DBN_RBM_HPP

namespace dbn {

/*!
 * \brief Restricted Boltzmann Machine
 */
template<typename Visible, typename Hidden, typename Bias, typename Weight = double>
struct rbm {
    Visible[] visibles;
    Hidden[] hiddens;
    Bias bias;
    Weight[][] weights;

    std::size_t num_hidden;
    std::size_t num_visible;

    double learning_rate;
};

} //end of dbn namespace

#endif