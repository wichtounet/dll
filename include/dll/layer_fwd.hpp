//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

//Forward declaration of the layers

namespace dll {

template <typename Desc>
struct rbm_impl;

template <typename Desc>
struct dyn_rbm_impl;

template <typename Desc>
struct conv_rbm_impl;

template <typename Desc>
struct dyn_conv_rbm_impl;

template <typename Desc>
struct conv_rbm_mp_impl;

template <typename Desc>
struct dyn_conv_rbm_mp_impl;

template <typename Desc>
struct mp_3d_layer;

template <typename Desc>
struct dyn_mp_3d_layer;

template <typename Desc>
struct avgp_3d_layer;

template <typename Desc>
struct dyn_avgp_3d_layer;

template <typename Desc>
struct upsample_3d_layer;

template <typename Desc>
struct dyn_upsample_3d_layer;

template <typename Desc>
struct binarize_layer_impl;

template <typename Desc>
struct normalize_layer_impl;

template <typename Desc>
struct rectifier_layer_impl;

template <typename Desc>
struct random_layer;

template <typename Desc>
struct lcn_layer_impl;

template <typename Desc>
struct dyn_lcn_layer_impl;

template <typename Desc>
struct scale_layer_impl;

template <typename Desc>
struct dense_layer_impl;

template <typename Desc>
struct dyn_dense_layer;

template <typename Desc>
struct conv_layer_impl;

template <typename Desc>
struct dyn_conv_layer_impl;

template <typename Desc>
struct deconv_layer_impl;

template <typename Desc>
struct dyn_deconv_layer_impl;

template <typename Desc>
struct activation_layer_impl;

} //end of dll namespace
