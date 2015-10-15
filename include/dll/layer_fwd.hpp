//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

//Forward declaration of the layers

#ifndef DLL_LAYER_FWD_HPP
#define DLL_LAYER_FWD_HPP

namespace dll {

template <typename Desc>
struct rbm;

template <typename Desc>
struct dyn_rbm;

template <typename Desc>
struct conv_rbm;

template <typename Desc>
struct conv_rbm_mp;

template <typename Desc>
struct mp_layer_3d;

template <typename Desc>
struct avgp_layer_3d;

template <typename Desc>
struct binarize_layer;

template <typename Desc>
struct normalize_layer;

template <typename Desc>
struct scale_layer;

template <typename Desc>
struct patches_layer;

template <typename Desc>
struct patches_layer_padh;

template <typename Desc>
struct dense_layer;

template <typename Desc>
struct conv_layer;

} //end of dll namespace

#endif
