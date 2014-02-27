//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <memory>

#include "dbn.hpp"
#include "layer.hpp"
#include "conf.hpp"
#include "mnist_reader.hpp"
#include "image_utils.hpp"

namespace {

template<typename V>
struct fake_label_array {
    V value;

    fake_label_array(V v) : value(v) {}

    double operator[](size_t i) const {
        if(i == value){
            return 1.0;
        } else {
            return 0.0;
        }
    }
};

template<typename T>
typename std::vector<fake_label_array<T>> make_fake(const std::vector<T>& values){
    std::vector<fake_label_array<T>> fake;
    fake.reserve(values.size());

    for(auto v: values){
        fake.emplace_back(v);
    }

    return std::move(fake);
}

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

template<typename DBN, typename Functor>
double test_set(DBN& dbn, std::vector<vector<double>>& images, const std::vector<uint8_t>& labels, Functor f){
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

template<typename DBN, typename P1>
void test_all(DBN& dbn, std::vector<vector<double>>& training_images, const std::vector<uint8_t>& training_labels, P1 predictor){
    auto test_images = mnist::read_test_images();
    auto test_labels = mnist::read_test_labels();

    if(test_images.empty() || test_labels.empty()){
        std::cout << "Impossible to read test set" << std::endl;
        return;
    }

    std::cout << "Start testing" << std::endl;

    std::cout << "Training Set" << std::endl;
    auto error_rate = test_set(dbn, training_images, training_labels, predictor);
    std::cout << "\tError rate (normal): " << 100.0 * error_rate << std::endl;

    std::cout << "Test Set" << std::endl;
    error_rate = test_set(dbn, test_images, test_labels, predictor);
    std::cout << "\tError rate (normal): " << 100.0 * error_rate << std::endl;
}

} //end of anonymous namespace

int main(int argc, char* argv[]){
    auto simple = false;
    auto load = false;

    for(int i = 1; i < argc; ++i){
        std::string command(argv[i]);

        if(command == "simple"){
            simple = true;
        }

        if(command == "load"){
            load = true;
        }
    }

    auto training_images = mnist::read_training_images();
    auto training_labels = mnist::read_training_labels();

    if(training_images.empty() || training_labels.empty()){
        return 1;
    }

    binarize_each(training_images);

    if(simple){
        typedef dbn::dbn<
            dbn::layer<dbn::conf<true, 50, true>, 28 * 28, 50>,
            dbn::layer<dbn::conf<true, 50, false>, 50, 50>,
            dbn::layer<dbn::conf<true, 50, false>, 60, 100>> dbn_simple_t;

        auto dbn = std::make_shared<dbn_simple_t>();

        dbn->train_with_labels(training_images, training_labels, 10, 5);

        test_all(dbn, training_images, training_labels, label_predictor());
    } else {
        typedef dbn::dbn<
            dbn::layer<dbn::conf<true, 100, true, true>, 28 * 28, 30>,
            //dbn::layer<dbn::conf<true, 100, false, true>, 300, 300>,
            dbn::layer<dbn::conf<true, 100, false, true>, 30, 30>,
            dbn::layer<dbn::conf<true, 100, false, true, true, dbn::Type::EXP>, 30, 10>> dbn_t;

        auto labels = make_fake(training_labels);

        auto dbn = std::make_shared<dbn_t>();

        dbn->display();

        if(load){
            std::cout << "Load from file" << std::endl;

            std::ifstream is("dbn.data", std::ifstream::binary);
            dbn->load(is);
        } else {
            std::cout << "Start pretraining" << std::endl;
            dbn->pretrain(training_images, 5);

            std::cout << "Start fine-tuning" << std::endl;
            dbn->fine_tune(training_images, labels, 5, 1000);

            std::ofstream os("dbn.data", std::ofstream::binary);
            dbn->store(os);
        }

        test_all(dbn, training_images, training_labels, predictor());
    }

    return 0;
}
