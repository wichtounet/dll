//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DBN_TEST_HPP
#define DBN_TEST_HPP

namespace dll {

struct predictor {
    template<typename T, typename V>
    size_t operator()(T& dbn, V& image){
        return dbn->predict(image);
    }
};

struct deep_predictor {
    template<typename T, typename V>
    size_t operator()(T& dbn, V& image){
        return dbn->deep_predict(image, 5);
    }
};

struct label_predictor {
    template<typename T, typename V>
    size_t operator()(T& dbn, V& image){
        return dbn->predict_labels(image, 10);
    }
};

struct deep_label_predictor {
    template<typename T, typename V>
    size_t operator()(T& dbn, V& image){
        return dbn->deep_predict_labels(image, 10, 5);
    }
};

template<typename DBN, typename Functor, typename Label>
double test_set(DBN& dbn, std::vector<vector<double>>& images, const std::vector<Label>& labels, Functor&& f){
    stop_watch<std::chrono::milliseconds> watch;

    size_t success = 0;
    for(size_t i = 0; i < images.size(); ++i){
        auto& image = images[i];
        auto& label = labels[i];

        auto predicted = f(dbn, image);

        if(predicted == label){
            ++success;
        }
    }

    auto elapsed = watch.elapsed();

    std::cout << "Testing took " << watch.elapsed() << "ms, average: " << (elapsed / images.size()) << "ms" << std::endl;

    return (images.size() - success) / static_cast<double>(images.size());
}

} //end of dbn namespace

#endif