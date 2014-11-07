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

#include "cpp_utils/stop_watch.hpp"    //Performance counter
#include "cpp_utils/assert.hpp"

#include "etl/multiplication.hpp"

#include "rbm_base.hpp"      //The base class
#include "base_conf.hpp"
#include "math.hpp"
#include "io.hpp"
#include "rbm_trainer_fwd.hpp"
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
    typedef float weight;

    using desc = Desc;
    using parent_t = Parent;

    static constexpr const unit_type visible_unit = desc::visible_unit;
    static constexpr const unit_type hidden_unit = desc::hidden_unit;

    static_assert(visible_unit != unit_type::SOFTMAX, "Softmax Visible units are not support");
    static_assert(hidden_unit != unit_type::GAUSSIAN, "Gaussian hidden units are not supported");

public:
    normal_rbm(){
        //Better initialization of learning rate
        rbm_base<desc>::learning_rate =
                visible_unit == unit_type::GAUSSIAN && is_relu(hidden_unit) ? 1e-5
            :   visible_unit == unit_type::GAUSSIAN || is_relu(hidden_unit) ? 1e-3
            :   /* Only ReLU and Gaussian Units needs lower rate */           1e-1;
    }

    template<typename Samples, bool EnableWatcher = true, typename RW = void, typename... Args>
    double train(Samples& training_data, std::size_t max_epochs, Args... args){
        dll::rbm_trainer<parent_t, EnableWatcher, RW> trainer(args...);
        return trainer.train(*static_cast<parent_t*>(this), training_data.begin(), training_data.end(), max_epochs);
    }

    template<typename Iterator, bool EnableWatcher = true, typename RW = void, typename... Args>
    double train(Iterator&& first, Iterator&& last, std::size_t max_epochs, Args... args){
        dll::rbm_trainer<parent_t, EnableWatcher, RW> trainer(args...);
        return trainer.train(*static_cast<parent_t*>(this), std::forward<Iterator>(first), std::forward<Iterator>(last), max_epochs);
    }

    void store(const std::string& file) const {
        rbm_detail::store(file, *static_cast<const parent_t*>(this));
    }

    void store(std::ostream& os) const {
        rbm_detail::store(os, *static_cast<const parent_t*>(this));
    }

    void load(const std::string& file){
        rbm_detail::load(file, *static_cast<parent_t*>(this));
    }

    void load(std::istream& is){
        rbm_detail::load(is, *static_cast<parent_t*>(this));
    }

    template<typename Iterator>
    void init_weights(Iterator&& first, Iterator&& last){
        rbm_detail::init_weights(std::forward<Iterator>(first), std::forward<Iterator>(last), *static_cast<parent_t*>(this));
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

    void display_units() const {
        rbm_detail::display_visible_units(*static_cast<const parent_t*>(this));
        rbm_detail::display_hidden_units(*static_cast<const parent_t*>(this));
    }

    void display_visible_units() const {
        rbm_detail::display_visible_units(*static_cast<const parent_t*>(this));
    }

    void display_visible_units(std::size_t matrix) const {
        rbm_detail::display_visible_units(*static_cast<const parent_t*>(this), matrix);
    }

    void display_hidden_units() const {
        rbm_detail::display_hidden_units(*static_cast<const parent_t*>(this));
    }

    void display_weights() const {
        rbm_detail::display_weights(*static_cast<const parent_t*>(this));
    }

    void display_weights(std::size_t matrix) const {
        rbm_detail::display_weights(matrix, *static_cast<const parent_t*>(this));
    }
};

} //end of dll namespace

#endif
