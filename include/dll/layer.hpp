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
    using parent_t = Parent;

    layer(const layer& rbm) = delete;
    layer& operator=(const layer& rbm) = delete;

    //No moving
    layer(layer&& rbm) = delete;
    layer& operator=(layer&& rbm) = delete;

    layer() {
#ifndef DLL_DENORMALS
        // Disable denormals for performance reason
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
#endif
    }

    void display() const {
        std::cout << as_derived().to_short_string() << std::endl;
    }

    // Default function

    template <typename Input, typename Output>
    void test_activate_hidden(Output& output, const Input& input) const {
        as_derived().activate_hidden(output, input);
    }

    template <typename Input, typename Output>
    void train_activate_hidden(Output& output, const Input& input) const {
        as_derived().activate_hidden(output, input);
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

private:
    //CRTP Deduction

    Parent& as_derived(){
        return *static_cast<Parent*>(this);
    }

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
