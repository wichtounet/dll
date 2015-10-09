//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_NEURAL_BASE_HPP
#define DLL_NEURAL_BASE_HPP

#include <memory>

#include "cg_context.hpp"       //Context for CG
#include "sgd_context.hpp"      //Context for SGD

namespace dll {

template<typename T>
T& unique_safe_get(std::unique_ptr<T>& ptr){
    if(!ptr){
        ptr = std::make_unique<T>();
    }

    return *ptr;
}

template<typename Parent>
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

    neural_base(){
        //Nothing to do
    }

    //CG context

    void init_cg_context(){
        if(!cg_context_ptr){
            cg_context_ptr = std::make_shared<cg_context<parent_t>>();
        }
    }

    cg_context<parent_t>& get_cg_context(){
        return *cg_context_ptr;
    }

    const cg_context<parent_t>& get_cg_context() const {
        return *cg_context_ptr;
    }

    //SGD context

    template<typename DBN>
    void init_sgd_context(){
        sgd_context_ptr = std::make_shared<sgd_context<DBN, parent_t>>();
    }

    template<typename DBN>
    sgd_context<DBN, parent_t>& get_sgd_context(){
        return *static_cast<sgd_context<DBN, parent_t>*>(sgd_context_ptr.get());
    }

    template<typename DBN>
    const sgd_context<DBN, parent_t>& get_sgd_context() const {
        return *static_cast<const sgd_context<DBN, parent_t>*>(sgd_context_ptr.get());
    }
};

} //end of dll namespace

#endif
