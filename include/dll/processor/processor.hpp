//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
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

#include "dll/dbn.hpp"

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

namespace dll {

namespace processor {

struct datasource {
    std::string source_file;
    std::string reader;

    bool binarize = false;
    bool normalize = false;

    long limit = -1;

    datasource(){}
    datasource(std::string source_file, std::string reader) : source_file(source_file), reader(reader) {}

    bool empty() const {
        return source_file.empty();
    }
};

struct pretraining_desc {
    std::size_t epochs = 25;
};

struct training_desc {
    std::size_t epochs = 25;
};

struct task {
    dll::processor::datasource pretraining;
    dll::processor::datasource samples;
    dll::processor::datasource labels;

    dll::processor::pretraining_desc pt_desc;
    dll::processor::training_desc ft_desc;
};

template<typename Sample>
bool read_samples(const datasource& ds, std::vector<Sample>& samples){
    if(ds.reader == "mnist"){
        std::size_t limit = 0;

        if(ds.limit > 0){
            limit = ds.limit;
        }

        mnist::read_mnist_image_file<std::vector, Sample>(samples, ds.source_file, limit, []{ return Sample(1 * 28 * 28); });

        if(ds.binarize){
            mnist::binarize_each(samples);
        }

        if(ds.normalize){
            mnist::normalize_each(samples);
        }

        return !samples.empty();
    } else {
        std::cout << "dllp: error: unknown samples reader: " << ds.reader << std::endl;
        return false;
    }
}

template<typename Label>
bool read_labels(const datasource& ds, std::vector<Label>& labels){
    if(ds.reader == "mnist"){
        std::size_t limit = 0;

        if(ds.limit > 0){
            limit = ds.limit;
        }

        mnist::read_mnist_label_file<std::vector, Label>(labels, ds.source_file, limit);

        return !labels.empty();
    } else {
        std::cout << "dllp: error: unknown labels reader: " << ds.reader << std::endl;
        return false;
    }
}

template<typename DBN>
void execute(DBN& dbn, task& task, const std::vector<std::string>& actions){
    std::cout << "Configured network:" << std::endl;
    dbn.display();

    using dbn_t = std::decay_t<DBN>;

    //Execute all the actions sequentially
    for(auto& action : actions){
        if(action == "pretrain"){
            if(task.pretraining.empty()){
                std::cout << "dllp: error: pretrain is not possible with a pretraining input" << std::endl;
                return;
            }

            std::vector<typename dbn_t::input_t> pt_samples;

            //Try to read the samples
            if(!read_samples(task.pretraining, pt_samples)){
                std::cout << "dllp: error: failed to read the pretraining samples" << std::endl;
                return;
            }

            //Pretrain the network
            dbn.pretrain(pt_samples.begin(), pt_samples.end(), task.pt_desc.epochs);
        } else if(action == "train"){
            if(task.samples.empty() || task.labels.empty()){
                std::cout << "dllp: error: train is not possible without samples and labels" << std::endl;
                return;
            }

            std::vector<typename dbn_t::input_t> ft_samples;
            std::vector<std::size_t> ft_labels;

            //Try to read the samples
            if(!read_samples(task.samples, ft_samples)){
                std::cout << "dllp: error: failed to read the training samples" << std::endl;
                return;
            }

            //Try to read the labels
            if(!read_labels(task.labels, ft_labels)){
                std::cout << "dllp: error: failed to read the training labels" << std::endl;
                return;
            }

        } else if(action == "test"){
            if(task.samples.empty() || task.labels.empty()){
                std::cout << "dllp: error: test is not possible without samples and labels" << std::endl;
                return;
            }

        } else {
            std::cout << "dllp: error: Invalid action: " << action << std::endl;
        }
    }


    //TODO
}

} //end of namespace processor

} //end of namespace dll
