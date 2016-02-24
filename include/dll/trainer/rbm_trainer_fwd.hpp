//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_RBM_TRAINER_FWD_HPP
#define DLL_RBM_TRAINER_FWD_HPP

namespace dll {

template <typename RBM, bool EnableWatcher = true, typename RW = void, bool Denoising = false>
struct rbm_trainer;

} //end of dll namespace

#endif
