//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <memory>

#include "etl/etl.hpp" // Every layer needs ETL

#include "dll/base_conf.hpp"           // Every layer description used the conf utility
#include "dll/trainer/context_fwd.hpp" // Forward declaration of the context classes
#include "dll/util/batch_reshape.hpp"  // For easy batch reshaping
#include "dll/util/ready.hpp"          // To create ready output
#include "dll/util/tmp.hpp"            // Every layer description needs TMP

namespace dll {

template <typename T>
T& unique_safe_get(std::unique_ptr<T>& ptr) {
    if (!ptr) {
        ptr = std::make_unique<T>();
    }

    return *ptr;
}

/*!
 * \brief A layer in a neural network
 */
template <typename Parent>
struct layer {
    using parent_t = Parent; ///< The CRTP parent layer

    //No copying
    layer(const layer& rbm) = delete;
    layer& operator=(const layer& rbm) = delete;

    //No moving
    layer(layer&& rbm) = delete;
    layer& operator=(layer&& rbm) = delete;

    /*!
     * \brief Default initialize the layer
     */
    layer() {
#ifndef DLL_DENORMALS
        // Disable denormals for performance reason
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
#endif
    }

    /*!
     * \brief Display a layer on the console
     */
    void display() const {
        std::cout << as_derived().to_short_string() << std::endl;
    }

    // Functions to forward propagate one sample at a time

    /*!
     * \brief Compute the test representation for a given input
     *
     * \param input The input to compute the representation from
     *
     * \return The test representation for the given input
     */
    template <typename Input>
    auto activate_hidden(const Input& input) const {
        // Prepare one fully-ready output
        auto output = prepare_one_ready_output(as_derived(), input);

        // Forward propagation
        test_activate_hidden(output, input);

        // Return the forward-propagated result
        return output;
    }

    /*!
     * \brief Compute the test presentation for a given input
     * \param output The output to fill
     * \param input The input to compute the representation from
     */
    template <typename Input, typename Output>
    void activate_hidden(Output&& output, const Input& input) const {
        test_activate_hidden(output, input);
    }

    /*!
     * \brief Compute the test presentation for a given input
     * \param output The output to fill
     * \param input The input to compute the representation from
     */
    template <typename Input, typename Output>
    void test_activate_hidden(Output&& output, const Input& input) const {
        as_derived().test_batch_activate_hidden(batch_reshape(output), batch_reshape(input));
    }

    /*!
     * \brief Compute the train presentation for a given input
     * \param output The output to fill
     * \param input The input to compute the representation from
     */
    template <typename Input, typename Output>
    void train_activate_hidden(Output&& output, const Input& input) const {
        as_derived().train_batch_activate_hidden(batch_reshape(output), batch_reshape(input));
    }

    /*!
     * \brief Compute the presentation for a given input, selecting train or
     * test at compile-time with Train template parameter
     *
     * \tparam Train if true compute the train representation, otherwise the test representation
     *
     * \param output The output to fill
     * \param input The input to compute the representation from
     */
    template <bool Train, typename Input, typename Output, cpp_enable_if(Train)>
    void select_activate_hidden(Output&& output, const Input& input) const {
        as_derived().train_batch_activate_hidden(batch_reshape(output), batch_reshape(input));
    }

    /*!
     * \brief Compute the presentation for a given input, selecting train or
     * test at compile-time with Train template parameter
     *
     * \tparam Train if true compute the train representation, otherwise the test representation
     *
     * \param output The output to fill
     * \param input The input to compute the representation from
     */
    template <bool Train, typename Input, typename Output, cpp_enable_if(!Train)>
    void select_activate_hidden(Output&& output, const Input& input) const {
        as_derived().test_batch_activate_hidden(batch_reshape(output), batch_reshape(input));
    }
    // Functions to forward propagate several samples (collection) at a time

    /*!
     * \brief Compute the test presentation for a collection of inputs
     *
     * \param output The output collection to fill
     * \param input The input collection to compute the representation from
     */
    template <typename Input, typename Output>
    void activate_many(Output&& output, const Input& input) const {
        test_activate_many(output, input);
    }

    /*!
     * \brief Compute the test presentation for a collection of inputs
     *
     * \param output The output collection to fill
     * \param input The input collection to compute the representation from
     */
    template <typename Input, typename Output>
    void test_activate_many(Output&& output, const Input& input) const {
        for(size_t i = 0; i < output.size(); ++i){
            test_activate_hidden(output[i], input[i]);
        }
    }

    /*!
     * \brief Compute the test presentation for a collection of inputs
     *
     * \param output The output collection to fill
     * \param input The input collection to compute the representation from
     */
    template <typename Input, typename Output>
    void train_activate_many(Output&& output, const Input& input) const {
        for(size_t i = 0; i < output.size(); ++i){
            train_activate_hidden(output[i], input[i]);
        }
    }

    /*!
     * \brief Compute the presentation for a collection of inputs, selecting train or
     * test at compile-time with Train template parameter
     *
     * \tparam Train if true compute the train representation,
     * otherwise the test representation
     *
     * \param output The collection of output to fill
     * \param input The collection of input to compute the representation from
     */
    template <bool Train, typename Input, typename Output, cpp_enable_if(Train)>
    void select_activate_many(Output&& output, const Input& input) const {
        train_activate_many(output, input);
    }

    /*!
     * \brief Compute the presentation for a collection of inputs, selecting train or
     * test at compile-time with Train template parameter
     *
     * \tparam Train if true compute the train representation,
     * otherwise the test representation
     *
     * \param output The collection of output to fill
     * \param input The collection of input to compute the representation from
     */
    template <bool Train, typename Input, typename Output, cpp_enable_if(!Train)>
    void select_activate_many(Output&& output, const Input& input) const {
        test_activate_many(output, input);
    }

    // Functions to propagate one batch at time

    template <typename Input>
    auto test_batch_activate_hidden(const Input& input) const {
        return as_derived().batch_activate_hidden(input);
    }

    template <typename Input>
    auto train_batch_activate_hidden(const Input& input) const {
        return as_derived().batch_activate_hidden(input);
    }

    template <typename Input, typename Output>
    void test_batch_activate_hidden(Output&& output, const Input& input) const {
        as_derived().batch_activate_hidden(output, input);
    }

    template <typename Input, typename Output>
    void train_batch_activate_hidden(Output&& output, const Input& input) const {
        as_derived().batch_activate_hidden(output, input);
    }

    // Prepare function

    template <typename Input>
    auto prepare_test_output(size_t samples) const {
        return as_derived().template prepare_output<Input>(samples);
    }

    template <typename Input>
    auto prepare_one_test_output() const {
        return as_derived().template prepare_one_output<Input>();
    }

    template <typename Input>
    auto prepare_train_output(size_t samples) const {
        return as_derived().template prepare_output<Input>(samples);
    }

    template <typename Input>
    auto prepare_one_train_output() const {
        return as_derived().template prepare_one_output<Input>();
    }

    template <bool Train, typename Input, cpp_enable_if(Train)>
    auto select_prepare_output(size_t samples) const {
        return as_derived().template prepare_train_output<Input>(samples);
    }

    template <bool Train, typename Input, cpp_enable_if(!Train)>
    auto select_prepare_output(size_t samples) const {
        return as_derived().template prepare_test_output<Input>(samples);
    }

    template <bool Train, typename Input, cpp_enable_if(Train)>
    auto select_prepare_one_output() const {
        return as_derived().template prepare_one_train_output<Input>();
    }

    template <bool Train, typename Input, cpp_enable_if(!Train)>
    auto select_prepare_one_output() const {
        return as_derived().template prepare_one_test_output<Input>();
    }

    //CG context

    /*
     * \brief Initialize the CG context
     */
    void init_cg_context() {
        if (!cg_context_ptr) {
            cg_context_ptr = std::make_shared<cg_context<parent_t>>();
        }
    }

    /*!
     * \brief Returns the context for CG training.
     * \return A reference to the CG context training.
     */
    cg_context<parent_t>& get_cg_context() {
        cpp_assert(cg_context_ptr, "Use of empty cg_context");

        return *cg_context_ptr;
    }

    /*!
     * \brief Returns the context for CG training.
     * \return A reference to the CG context training.
     */
    const cg_context<parent_t>& get_cg_context() const {
        cpp_assert(sgd_context_ptr, "Use of empty cg_context");

        return *cg_context_ptr;
    }

private:
    //CRTP Deduction

    /*!
     * \brief Returns a reference to the derived object, i.e. the object using the CRTP injector.
     * \return a reference to the derived object.
     */
    Parent& as_derived(){
        return *static_cast<Parent*>(this);
    }

    /*!
     * \brief Returns a reference to the derived object, i.e. the object using the CRTP injector.
     * \return a reference to the derived object.
     */
    const Parent& as_derived() const {
        return *static_cast<const Parent*>(this);
    }

protected:
    /*!
     * \brief Pointer to the Conjugate Gradient (CG) context.
     *
     * Needs to be shared because of dyn_rbm
     */
    mutable std::shared_ptr<cg_context<parent_t>> cg_context_ptr;

    /*!
     * \brief Pointer to the Stochastic Gradient Descent (SGD) context.
     *
     * Needs to be shared because of dyn_rbm
     */
    mutable std::shared_ptr<void> sgd_context_ptr;
};

} //end of dll namespace
