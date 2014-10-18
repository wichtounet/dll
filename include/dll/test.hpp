//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_TEST_HPP
#define DLL_TEST_HPP

namespace dll {

struct predictor {
    template<typename T, typename V>
    std::size_t operator()(T& dbn, V& image){
        return dbn->predict(image);
    }
};

#ifdef DLL_SVM_SUPPORT

struct svm_predictor {
    template<typename T, typename V>
    std::size_t operator()(T& dbn, V& image){
        return dbn->svm_predict(image);
    }
};

#endif //DLL_SVM_SUPPORT

struct deep_predictor {
    template<typename T, typename V>
    std::size_t operator()(T& dbn, V& image){
        return dbn->deep_predict(image, 5);
    }
};

struct label_predictor {
    template<typename T, typename V>
    std::size_t operator()(T& dbn, V& image){
        return dbn->predict_labels(image, 10);
    }
};

struct deep_label_predictor {
    template<typename T, typename V>
    std::size_t operator()(T& dbn, V& image){
        return dbn->deep_predict_labels(image, 10, 5);
    }
};

template<typename DBN, typename Functor, typename Samples, typename Labels>
double test_set(DBN& dbn, const Samples& images, const Labels& labels, Functor&& f){
    return test_set(dbn, images.begin(), images.end(), labels.begin(), labels.end(), std::forward<Functor>(f));
}

template<typename DBN, typename Functor, typename Iterator, typename LIterator>
double test_set(DBN& dbn, Iterator first, Iterator last, LIterator lfirst, LIterator /*llast*/, Functor&& f){
    cpp::stop_watch<std::chrono::milliseconds> watch;

    std::size_t success = 0;
    std::size_t images = 0;

    while(first != last){
        const auto& image = *first;
        const auto& label = *lfirst;

        auto predicted = f(dbn, image);

        if(predicted == label){
            ++success;
        }

        ++images;
        ++first;
        ++lfirst;
    }

    auto elapsed = watch.elapsed();

    std::cout << "Testing took " << watch.elapsed() << "ms, average: " << (elapsed / images) << "ms" << std::endl;

    return (images - success) / static_cast<double>(images);
}

} //end of dbn namespace

#endif