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
#include "dll/util/tmp.hpp"            // Every layer description needs TMP
#include "dll/trainer/context_fwd.hpp" // Forward declaration of the context classes

namespace dll {

template <typename T>
T& unique_safe_get(std::unique_ptr<T>& ptr) {
    if (!ptr) {
        ptr = std::make_unique<T>();
    }

    return *ptr;
}

template <typename Parent>
struct layer {
    using parent_t = Parent; ///< The CRTP parent layer

    layer(const layer& rbm) = delete;
    layer& operator=(const layer& rbm) = delete;

    //No moving
    layer(layer&& rbm) = delete;
    layer& operator=(layer&& rbm) = delete;

    /*
     * !\brief Default initialize the layer
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

    // Default function

    template <typename Input>
    auto activate_hidden(const Input& input) const {
        auto output = as_derived().template prepare_one_output<Input>();
        as_derived().activate_hidden(output, input);
        return output;
    }

    template <typename Input, typename Output>
    void test_activate_hidden(Output& output, const Input& input) const {
        as_derived().activate_hidden(output, input);
    }

    template <typename Input, typename Output>
    void train_activate_hidden(Output& output, const Input& input) const {
        as_derived().activate_hidden(output, input);
    }

    template <typename Input>
    auto test_batch_activate_hidden(const Input& input) const {
        return as_derived().batch_activate_hidden(input);
    }

    template <typename Input, typename Output>
    void test_batch_activate_hidden(Output& output, const Input& input) const {
        as_derived().batch_activate_hidden(output, input);
    }

    template <typename Input, typename Output>
    void train_batch_activate_hidden(Output& output, const Input& input) const {
        as_derived().batch_activate_hidden(output, input);
    }

    template <bool Train, typename Input, typename Output, cpp_enable_if(Train)>
    void select_activate_hidden(Output& output, const Input& input) const {
        as_derived().train_activate_hidden(output, input);
    }

    template <bool Train, typename Input, typename Output, cpp_enable_if(!Train)>
    void select_activate_hidden(Output& output, const Input& input) const {
        as_derived().test_activate_hidden(output, input);
    }

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
    //Needs to be shared because of dyn_rbm
    mutable std::shared_ptr<cg_context<parent_t>> cg_context_ptr;

    //Needs to be shared because of dyn_rbm
    mutable std::shared_ptr<void> sgd_context_ptr;
};

} //end of dll namespace
