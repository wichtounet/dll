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
template<typename Visible, typename Hidden, typename Weight = double>
struct rbm {
    std::vector<Visible> visibles;
    std::vector<Hidden> hiddens;

    Weight[][] weights;
    Weight[] bias_visibles;
    Weight[] bias_hidden;

    std::size_t num_hidden;
    std::size_t num_visible;

    double learning_rate;

    void epoch(TrainingItem[] item){
        //Size should match

        //Set the states of the visible units
        for(size_t v = 0; v < num_visible; ++v){
            visibles[v] = item[v];
        }

        //Update the states of the hidden units
        auto energy_vh = - mul_sum(bias_hidden, visibles);





    }
};

template<typename A, typename B>
double mul_sum(const std::vector<A>& a, const std::vector<B>& b){
    double acc = 0;

    for(size_t i = 0; i < a.size(); ++i){
        acc += a[i] * b[i];
    }

    return acc;
}

} //end of dbn namespace

#endif