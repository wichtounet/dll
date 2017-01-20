//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace dll {

template <typename Layer>
struct neural_layer_base_traits;

template <typename Layer>
struct rbm_layer_base_traits;

// Helper is standard dense
// Helper is dense rbm
// Helper is conv rbm
// Helper has same_type
// Helper is_trained
// Helper is_pretrained
// Helper is_multiplex

} //end of dll namespace
