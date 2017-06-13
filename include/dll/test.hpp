//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "cpp_utils/stop_watch.hpp"

namespace dll {

/*!
 * \brief Utility to predict a label from an input
 */
struct predictor {
    /*!
     * \brief Return the predicted label for the given image using the given DBN
     */
    template <typename T, typename V>
    size_t operator()(T& dbn, V& image) {
        return dbn->predict(image);
    }
};

#ifdef DLL_SVM_SUPPORT

/*!
 * \brief Utility to predict a label from an input in SVM mode
 */
struct svm_predictor {
    /*!
     * \brief Return the predicted label for the given image using the given DBN
     * in SVM mode.
     */
    template <typename T, typename V>
    size_t operator()(T& dbn, V& image) {
        return dbn->svm_predict(image);
    }
};

#endif //DLL_SVM_SUPPORT

/*!
 * \brief Utility to predict a label from an input using a DBN with only RBM
 * pretraining.
 */
struct label_predictor {
    /*!
     * \brief Return the predicted label for the given image using the given DBN
     */
    template <typename T, typename V>
    size_t operator()(T& dbn, V& image) {
        return dbn->predict_labels(image, 10);
    }
};

template <typename DBN, typename Functor, typename Samples, typename Labels>
double test_set(DBN& dbn, const Samples& images, const Labels& labels, Functor&& f) {
    return test_set(dbn, images.begin(), images.end(), labels.begin(), labels.end(), std::forward<Functor>(f));
}

template <typename DBN, typename Functor, typename Iterator, typename LIterator>
double test_set(DBN& dbn, Iterator first, Iterator last, LIterator lfirst, LIterator /*llast*/, Functor&& f) {
    size_t success = 0;
    size_t images  = 0;

    while (first != last) {
        const auto& image = *first;
        const auto& label = *lfirst;

        auto predicted = f(dbn, image);

        if (predicted == label) {
            ++success;
        }

        ++images;
        ++first;
        ++lfirst;
    }

    return (images - success) / static_cast<double>(images);
}

template <typename DBN, typename Samples>
double test_set_ae(DBN& dbn, const Samples& images) {
    return test_set_ae(dbn, images.begin(), images.end());
}

template <typename DBN, typename Iterator>
double test_set_ae(DBN& dbn, Iterator first, Iterator last) {
    double rate        = 0.0;
    size_t images = 0;

    while (first != last) {
        decltype(auto) image = *first;
        decltype(auto) rec_image = dbn.features(image);

        rate += etl::mean(abs(image - rec_image));

        ++images;
        ++first;
    }

    return std::abs(rate) / images;
}

} //end of dll namespace
