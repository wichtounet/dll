//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

/*!
 * \file processor.hpp
 * \brief This file is made to be included by the dllp generated file only.
 */

#include <string>
#include <vector>
#include <iostream>
#include <iomanip>

#include "dll/rbm/rbm.hpp"
#include "dll/rbm/conv_rbm.hpp"
#include "dll/neural/dense/dense_layer.hpp"
#include "dll/neural/conv/conv_layer.hpp"
#include "dll/dbn.hpp"
#include "dll/text_reader.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

namespace dll {

namespace processor {

struct options {
    bool quiet  = false;
    bool mkl    = false;
    bool cublas = false;
    bool cufft  = false;
    bool cache  = false;
};

template <typename LastLayer, typename Enable = void>
struct sgd_possible {
    static constexpr bool value = false;
};

template <typename LastLayer>
struct sgd_possible<LastLayer, std::enable_if_t<decay_layer_traits<LastLayer>::base_traits::sgd_supported>> {
    static constexpr bool value = true;
};

//These functions are only exposed to be able to unit-test the program
int process_file(const options& opt, const std::vector<std::string>& actions, const std::string& source_file);
std::string process_file_result(const options& opt, const std::vector<std::string>& actions, const std::string& source_file);

constexpr double stupid_default = -666.0;

struct datasource {
    std::string source_file;
    std::string reader;

    bool binarize         = false;
    bool normalize        = false;
    bool scale            = false;
    double scale_d        = 0.0;
    bool shift            = false;
    double shift_d        = 0.0;
    bool normal_noise     = false;
    double normal_noise_d = 0.0;

    long limit = -1;

    datasource() {}
    datasource(std::string source_file, std::string reader)
            : source_file(std::move(source_file)), reader(std::move(reader)) {}

    bool empty() const {
        return source_file.empty();
    }
};

struct datasource_pack {
    datasource samples;
    datasource labels;
};

struct general_desc {
    bool batch_mode       = false;
    size_t big_batch = 1;
};

struct pretraining_desc {
    size_t epochs = 25;
    bool denoising     = false;
};

struct training_desc {
    size_t epochs     = 25;
    double learning_rate   = stupid_default;
    double momentum        = stupid_default;
    size_t batch_size = 0;

    std::string decay     = "none";
    double l1_weight_cost = stupid_default;
    double l2_weight_cost = stupid_default;

    std::string trainer = "none";

    bool verbose = false;
};

struct weights_desc {
    std::string file = "weights.dat";
};

struct task {
    std::vector<std::string> default_actions;

    dll::processor::datasource_pack pretraining;
    dll::processor::datasource_pack pretraining_clean;
    dll::processor::datasource_pack training;
    dll::processor::datasource_pack testing;

    dll::processor::pretraining_desc pt_desc;
    dll::processor::training_desc ft_desc;
    dll::processor::weights_desc w_desc;
    dll::processor::general_desc general_desc;
};

template <bool Three, typename Sample>
bool read_samples(const datasource& ds, std::vector<Sample>& samples) {
    size_t limit = 0;

    if (ds.limit > 0) {
        limit = ds.limit;
    }

    if (ds.reader == "mnist") {
        mnist::read_mnist_image_file<std::vector, Sample>(samples, ds.source_file, limit, [] { return Sample(1 * 28 * 28); });
    } else if(ds.reader == "text"){
        dll::text::read_images_direct<Three>(samples, ds.source_file, limit);
    } else {
        std::cout << "dllp: error: unknown samples reader: " << ds.reader << std::endl;
        return false;
    }

    if (ds.binarize) {
        mnist::binarize_each(samples);
    }

    if (ds.normalize) {
        mnist::normalize_each(samples);
    }

    if (ds.shift) {
        for (auto& vec : samples) {
            for (auto& v : vec) {
                v += ds.shift_d;
            }
        }
    }

    if (ds.scale) {
        for (auto& vec : samples) {
            for (auto& v : vec) {
                v *= ds.scale_d;
            }
        }
    }

    if (ds.normal_noise) {
        mnist::normalize_each(samples);

        std::random_device rd;
        std::default_random_engine rand_engine(rd());
        std::normal_distribution<float> normal_distribution(0.0, ds.normal_noise_d);
        auto noise = std::bind(normal_distribution, rand_engine);

        for (auto& vec : samples) {
            for (auto& noisy_x : vec) {
                noisy_x += noise();
            }
        }

        mnist::normalize_each(samples);
    }

    return !samples.empty();
}

template <typename Label>
bool read_labels(const datasource& ds, std::vector<Label>& labels) {
    size_t limit = 0;

    if (ds.limit > 0) {
        limit = ds.limit;
    }

    if (ds.reader == "mnist") {
        mnist::read_mnist_label_file<std::vector, Label>(labels, ds.source_file, limit);
    } else if (ds.reader == "text") {
        dll::text::read_labels<std::vector, Label>(labels, ds.source_file, limit);
    } else {
        std::cout << "dllp: error: unknown labels reader: " << ds.reader << std::endl;
        return false;
    }

    return !labels.empty();
}

inline void print_title(const std::string& value) {
    std::cout << std::string(25, ' ') << std::endl;
    std::cout << std::string(25, '*') << std::endl;
    std::cout << "* " << value << std::string(25 - value.size() - 3, ' ') << "*" << std::endl;
    std::cout << std::string(25, '*') << std::endl;
    std::cout << std::string(25, ' ') << std::endl;
}

template <typename Container, bool Three, typename DBN>
void execute(DBN& dbn, task& task, const std::vector<std::string>& actions) {
    print_title("Network");
    dbn.display();

    using dbn_t = std::decay_t<DBN>;

    //Execute all the actions sequentially
    for (auto& action : actions) {
        if (action == "pretrain") {
            print_title("Pretraining");

            if (task.pretraining.samples.empty()) {
                std::cout << "dllp: error: pretrain is not possible without a pretraining input" << std::endl;
                return;
            }

            std::vector<Container> pt_samples;

            //Try to read the samples
            if (!read_samples<Three>(task.pretraining.samples, pt_samples)) {
                std::cout << "dllp: error: failed to read the pretraining samples" << std::endl;
                return;
            }

            if (task.pt_desc.denoising) {
                std::vector<Container> clean_samples;

                //Try to read the samples
                if (!read_samples<Three>(task.pretraining_clean.samples, clean_samples)) {
                    std::cout << "dllp: error: failed to read the clean samples" << std::endl;
                    return;
                }

                //Pretrain the network
                if constexpr(dbn_t::pretrain_possible && dbn_t::layers_t::is_denoising) {
                    dbn.pretrain_denoising(pt_samples.begin(), pt_samples.end(), clean_samples.begin(), clean_samples.end(), task.pt_desc.epochs);
                }
            } else {
                if constexpr (dbn_t::pretrain_possible) {
                    //Pretrain the network
                    dbn.pretrain(pt_samples.begin(), pt_samples.end(), task.pt_desc.epochs);
                }
            }
        } else if (action == "train") {
            print_title("Training");

            if (task.training.samples.empty() || task.training.labels.empty()) {
                std::cout << "dllp: error: train is not possible without samples and labels" << std::endl;
                return;
            }

            std::vector<Container> ft_samples;
            std::vector<size_t> ft_labels;

            //Try to read the samples
            if (!read_samples<Three>(task.training.samples, ft_samples)) {
                std::cout << "dllp: error: failed to read the training samples" << std::endl;
                return;
            }

            //Try to read the labels
            if (!read_labels(task.training.labels, ft_labels)) {
                std::cout << "dllp: error: failed to read the training labels" << std::endl;
                return;
            }

            using last_layer = typename dbn_t::template layer_type<dbn_t::layers - 1>;

            if(!sgd_possible<last_layer>::value){
                std::cout << "dllp: error: The network is not trainable by SGD" << std::endl;
                return;
            }

            //Train the network
            if constexpr(sgd_possible<last_layer>::value) {
                auto ft_error = dbn.fine_tune(ft_samples, ft_labels, task.ft_desc.epochs);
                std::cout << "Train Classification Error:" << ft_error << std::endl;
            }
        } else if (action == "test") {
            print_title("Testing");

            if (task.testing.samples.empty() || task.testing.labels.empty()) {
                std::cout << "dllp: error: test is not possible without samples and labels" << std::endl;
                return;
            }

            std::vector<Container> test_samples;
            std::vector<size_t> test_labels;

            //Try to read the samples
            if (!read_samples<Three>(task.testing.samples, test_samples)) {
                std::cout << "dllp: error: failed to read the test samples" << std::endl;
                return;
            }

            //Try to read the labels
            if (!read_labels(task.testing.labels, test_labels)) {
                std::cout << "dllp: error: failed to read the test labels" << std::endl;
                return;
            }

            auto classes = dbn.output_size();

            etl::dyn_matrix<size_t, 2> conf(classes, classes, 0.0);

            size_t n  = test_samples.size();
            size_t tp = 0;

            for (size_t i = 0; i < test_samples.size(); ++i) {
                auto sample = test_samples[i];
                auto label  = test_labels[i];

                auto predicted = dbn.predict(sample);

                if (predicted == label) {
                    ++tp;
                }

                ++conf(label, predicted);
            }

            double test_error = (n - tp) / double(n);

            std::cout << "Error rate: " << test_error << std::endl;
            std::cout << "Accuracy: " << (1.0 - test_error) << std::endl
                      << std::endl;

            std::cout << "Results per class" << std::endl;

            double overall = 0.0;

            std::cout << "   | Accuracy | Error rate |" << std::endl;

            for (size_t l = 0; l < classes; ++l) {
                size_t total = etl::sum(conf(l));
                double acc = (total - conf(l, l)) / double(total);
                std::cout << std::setw(3) << l;
                std::cout << "|" << std::setw(10) << (1.0 - acc) << "|" << std::setw(12) << acc << "|" << std::endl;
                overall += acc;
            }

            std::cout << std::endl;

            std::cout << "Overall Error rate: " << overall / classes << std::endl;
            std::cout << "Overall Accuracy: " << 1.0 - (overall / classes) << std::endl
                      << std::endl;

            std::cout << "Confusion Matrix (%)" << std::endl
                      << std::endl;

            std::cout << "    ";
            for (size_t l = 0; l < classes; ++l) {
                std::cout << std::setw(5) << l << " ";
            }
            std::cout << std::endl;

            for (size_t l = 0; l < classes; ++l) {
                size_t total = etl::sum(conf(l));
                std::cout << std::setw(3) << l << "|";
                for (size_t p = 0; p < classes; ++p) {
                    std::cout << std::setw(5) << std::setprecision(2) << 100.0 * (conf(l, p) / double(total)) << "|";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        } else if (action == "save") {
            print_title("Save Weights");

            dbn.store(task.w_desc.file);
            std::cout << "Weights saved" << std::endl;
        } else if (action == "load") {
            print_title("Load Weights");

            dbn.load(task.w_desc.file);
            std::cout << "Weights loaded" << std::endl;
        } else {
            std::cout << "dllp: error: Invalid action: " << action << std::endl;
        }
    }

    //TODO
}

} //end of namespace processor

} //end of namespace dll
