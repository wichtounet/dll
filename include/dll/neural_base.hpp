//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_NEURAL_BASE_HPP
#define DLL_NEURAL_BASE_HPP

#include <memory>

#include "trainer/cg_context.hpp"  //Context for CG
#include "trainer/sgd_context.hpp" //Context for SGD

namespace dll {

template <typename T>
T& unique_safe_get(std::unique_ptr<T>& ptr) {
    if (!ptr) {
        ptr = std::make_unique<T>();
    }

    return *ptr;
}

template <typename Parent>
struct neural_base {
    using parent_t = Parent;

    //Needs to be shared because of dyn_rbm
    mutable std::shared_ptr<cg_context<parent_t>> cg_context_ptr;

    //Needs to be shared because of dyn_rbm
    mutable std::shared_ptr<void> sgd_context_ptr;

    neural_base(const neural_base& rbm) = delete;
    neural_base& operator=(const neural_base& rbm) = delete;

    //No moving
    neural_base(neural_base&& rbm) = delete;
    neural_base& operator=(neural_base&& rbm) = delete;

    neural_base() {
        //Nothing to do
    }

    //CRTP Deduction

    Parent& as_derived(){
        return *static_cast<Parent*>(this);
    }

    const Parent& as_derived() const {
        return *static_cast<const Parent*>(this);
    }

    // Default function

    template <typename Input, typename Output>
    void test_activate_hidden(Output& output, const Input& input) const {
        return as_derived().activate_hidden(output, input);
    }

    template <typename Input, typename Output>
    void train_activate_hidden(Output& output, const Input& input) const {
        return as_derived().activate_hidden(output, input);
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
    auto prepare_test_output(std::size_t samples) const {
        return as_derived().template prepare_output<Input>(samples);
    }

    template <typename Input>
    auto prepare_one_test_output() const {
        return as_derived().template prepare_one_output<Input>();
    }

    template <typename Input>
    auto prepare_train_output(std::size_t samples) const {
        return as_derived().template prepare_output<Input>(samples);
    }

    template <typename Input>
    auto prepare_one_train_output() const {
        return as_derived().template prepare_one_output<Input>();
    }

    template <bool Train, typename Input, cpp_enable_if(Train)>
    auto select_prepare_output(std::size_t samples) const {
        return as_derived().template prepare_train_output<Input>(samples);
    }

    template <bool Train, typename Input, cpp_enable_if(!Train)>
    auto select_prepare_output(std::size_t samples) const {
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

    void init_cg_context() {
        if (!cg_context_ptr) {
            cg_context_ptr = std::make_shared<cg_context<parent_t>>();
        }
    }

    cg_context<parent_t>& get_cg_context() {
        return *cg_context_ptr;
    }

    const cg_context<parent_t>& get_cg_context() const {
        return *cg_context_ptr;
    }

    //SGD context

    template <typename DBN>
    void init_sgd_context() {
        sgd_context_ptr = std::make_shared<sgd_context<DBN, parent_t>>();
    }

    template <typename DBN>
    sgd_context<DBN, parent_t>& get_sgd_context() {
        return *static_cast<sgd_context<DBN, parent_t>*>(sgd_context_ptr.get());
    }

    template <typename DBN>
    const sgd_context<DBN, parent_t>& get_sgd_context() const {
        return *static_cast<const sgd_context<DBN, parent_t>*>(sgd_context_ptr.get());
    }
};

} //end of dll namespace

#endif
