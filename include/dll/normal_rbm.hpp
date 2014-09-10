//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_NORMAL_RBM_INL
#define DLL_NORMAL_RBM_INL

#include <cmath>
#include <vector>
#include <random>
#include <functional>
#include <ctime>

#include "etl/multiplication.hpp"

#include "rbm_base.hpp"      //The base class
#include "stop_watch.hpp"    //Performance counter
#include "assert.hpp"
#include "base_conf.hpp"
#include "math.hpp"
#include "io.hpp"

#include "rbm_common.hpp"

namespace dll {

/*!
 * \brief Standard version of Restricted Boltzmann Machine
 *
 * This follows the definition of a RBM by Geoffrey Hinton.
 */
template<typename Parent, typename Desc>
class normal_rbm : public rbm_base<Desc> {
public:
    typedef double weight;
    typedef double value_t;

    using desc = Desc;
    using parent_t = Parent;

    static constexpr const unit_type visible_unit = desc::visible_unit;
    static constexpr const unit_type hidden_unit = desc::hidden_unit;

    static_assert(visible_unit != unit_type::SOFTMAX && visible_unit != unit_type::EXP,
        "Exponential and softmax Visible units are not support");
    static_assert(hidden_unit != unit_type::GAUSSIAN,
        "Gaussian hidden units are not supported");

public:
    normal_rbm(){
        //Better initialization of learning rate
        rbm_base<desc>::learning_rate =
                visible_unit == unit_type::GAUSSIAN && is_relu(hidden_unit) ? 1e-5
            :   visible_unit == unit_type::GAUSSIAN || is_relu(hidden_unit) ? 1e-3
            :   /* Only ReLU and Gaussian Units needs lower rate */           1e-1;
    }

    void display() const {
        rbm_detail::display_visible_units(*static_cast<parent_t*>(this));
        rbm_detail::display_hidden_units(*static_cast<parent_t*>(this));
    }

    template<typename Samples>
    double train(const Samples& training_data, std::size_t max_epochs){
        dll::rbm_trainer<parent_t> trainer;
        return trainer.train(*static_cast<parent_t*>(this), training_data, max_epochs);
    }

    void store(std::ostream& os) const {
        rbm_detail::store(os, *static_cast<const parent_t*>(this));
    }

    void load(std::istream& is){
        rbm_detail::load(is, *static_cast<parent_t*>(this));
    }

    template<typename Samples>
    void init_weights(const Samples& training_data){
        rbm_detail::init_weights(training_data, *static_cast<parent_t*>(this));
    }

    template<typename V, typename H>
    weight energy(const V& v, const H& h) const {
        return rbm_detail::free_energy(*static_cast<const parent_t*>(this), v, h);
    }

    template<typename V>
    weight free_energy(const V& v) const {
        return rbm_detail::free_energy(*static_cast<const parent_t*>(this), v);
    }

    weight free_energy() const {
        auto& p = *static_cast<const parent_t*>(this);
        return rbm_detail::free_energy(p, p.v1);
    }

    template<typename Sample>
    void reconstruct(const Sample& items){
        rbm_detail::reconstruct(items, *static_cast<parent_t*>(this));
    }

    void display_weights() const {
        rbm_detail::display_weights(*static_cast<const parent_t*>(this));
    }

    void display_weights(size_t matrix) const {
        rbm_detail::display_weights(matrix, *static_cast<const parent_t*>(this));
    }
};

} //end of dbn namespace

#endif