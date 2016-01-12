//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Deep Belief Network implementation with type erasure
 *
 * In this library, a DBN can also be used with standard neural network layers,
 * in which case, it acts as a standard neural network and cannot be
 * pretrained.
 */

#pragma once

#include "cpp_utils/static_if.hpp"

#include "unit_type.hpp"
#include "dbn_trainer.hpp"
#include "conjugate_gradient.hpp"
#include "dbn_common.hpp"
#include "svm_common.hpp"
#include "flatten.hpp"
#include "compat.hpp"
#include "format.hpp"
#include "export.hpp"
#include "dbn_detail.hpp" //dbn_detail namespace

namespace dll {

//TODO It is necessary to find a way to reduce duplication between dbn and dbn_fast
// A base class does not work because its typedef are not used in resolution in non-dependend names
// moreover, it needs the weight type which needs the class itself and therefore the type is not complete at this point...
// macro will be ugly as hell, but at least will be reducing duplication

/*!
 * \brief A Deep Belief Network implementation
 */
template<typename Desc>
struct dbn_fast final {
    using desc = Desc;              ///< The network descriptor
    using this_type = dbn_fast<desc>;    ///< The network type

    using layers_t = typename desc::layers;  ///< The layers container type

    template<std::size_t N>
    using layer_type = detail::layer_type_t<N, layers_t>;             ///< The type of the layer at index Nth

    using weight = typename dbn_detail::extract_weight_t<0, this_type>::type;     ///< The tpyeof the weights

    using watcher_t = typename desc::template watcher_t<this_type>;   ///< The watcher type

    using input_t = typename dbn_detail::layer_input_simple<this_type, 0>::type;    ///< The input type of the network

    template<std::size_t B>
    using input_batch_t = typename dbn_detail::layer_input_batch<this_type, 0>::template type<B>;      ///< The input batch type of the network for a batch size of B

    template<std::size_t N>
    using layer_input_t = dbn_detail::layer_input_t<this_type, N>;

    template<std::size_t N>
    using layer_output_t = dbn_detail::layer_output_t<this_type, N>;

    using label_output_t = layer_input_t<layers_t::size - 1>;
    using output_one_t = layer_output_t<layers_t::size - 1>; ///< The type of a single output of the network

    using output_t = std::conditional_t<
            dbn_traits<this_type>::is_multiplex(),
            std::vector<output_one_t>,
            output_one_t>;                           ///< The output type of the network

    using full_output_t = etl::dyn_vector<weight>;

    using svm_samples_t = std::conditional_t<
        dbn_traits<this_type>::concatenate(),
        std::vector<etl::dyn_vector<weight>>,       //In full mode, use a simple 1D vector
        typename layer_type<layers_t::size - 1>::output_t>; //In normal mode, use the output of the last layer

    static constexpr const std::size_t layers = layers_t::size;                 ///< The number of layers
    static constexpr const std::size_t batch_size = desc::BatchSize;            ///< The batch size (for finetuning)
    static constexpr const std::size_t big_batch_size = desc::BigBatchSize;     ///< The number of pretraining batch to do at once

    weight learning_rate = 0.1;         ///< The learning rate for finetuning
    weight lr_bold_inc = 1.05;          ///< The multiplicative increase of learning rate for the bold driver
    weight lr_bold_dec = 0.5;           ///< The multiplicative decrease of learning rate for the bold driver
    weight lr_step_gamma = 0.5;         ///< The multiplicative decrease of learning rate for the step driver
    std::size_t lr_step_size = 10;      ///< The number of steps after which the step driver decreases the learning rate

    weight initial_momentum = 0.5;      ///< The initial momentum
    weight final_momentum = 0.9;        ///< The final momentum applied after *final_momentum_epoch* epoch
    weight final_momentum_epoch = 6;    ///< The epoch at which momentum change

    weight l1_weight_cost = 0.0002;     ///< The weight cost for L1 weight decay
    weight l2_weight_cost = 0.0002;     ///< The weight cost for L2 weight decay

    weight momentum = 0;                ///< The current momentum

    bool memory_mode = false;

#ifdef DLL_SVM_SUPPORT
    //TODO Ideally these fields should be private
    svm::model svm_model;               ///< The learned model
    svm::problem problem;               ///< libsvm is stupid, therefore, you cannot destroy the problem if you want to use the model...
    bool svm_loaded = false;            ///< Indicates if a SVM model has been loaded (and therefore must be saved)
#endif //DLL_SVM_SUPPORT

private:
    cpp::thread_pool<!dbn_traits<this_type>::is_serial()> pool;

    mutable int fake_resource;          ///< Simple field to get a reference from for resource management

public:
    /*!
     * Constructs a DBN and initializes all its members.
     *
     * This is the only way to create a DBN.
     */
    dbn_fast(){
        //Nothing else to init
    }

    //No copying
    dbn_fast(const dbn_fast& rhs) = delete;
    dbn_fast& operator=(const dbn_fast& rhs) = delete;

    //No moving
    dbn_fast(dbn_fast&& rhs) = delete;
    dbn_fast& operator=(dbn_fast&& rhs) = delete;
};

} //end of namespace dll
