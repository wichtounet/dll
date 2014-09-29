//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_DBN_TRAINER_HPP
#define DLL_DBN_TRAINER_HPP

#include "dll/labels.hpp"
#include "dll/test.hpp"
#include "dll/dbn_traits.hpp"

namespace dll {

/*!
 * \brief A generic trainer for Deep Belief Network
 *
 * This trainer use the specified trainer of the DBN to perform supervised
 * fine-tuning.
 */
template<typename DBN>
struct dbn_trainer {
    using dbn_t = DBN;

    template<typename R>
    using trainer_t = typename dbn_t::desc::template trainer_t<R>;

    template<typename R>
    using watcher_t = typename dbn_t::desc::template watcher_t<R>;

    template<typename Samples, typename Labels>
    typename dbn_t::weight train(DBN& dbn, const Samples& training_data, Labels& labels, size_t max_epochs, size_t batch_size) const {
        watcher_t<dbn_t> watcher;

        dbn.momentum = dbn.initial_momentum;

        watcher.fine_tuning_begin(dbn);

        auto trainer = std::make_unique<trainer_t<dbn_t>>(dbn);

        //Initialize the trainer if necessary
        trainer->init_training(batch_size);

        //Convert labels to an useful form
        auto fake_labels = dll::make_fake(labels);

        //Get types for the batch
        using fake_label_t = typename std::remove_reference<decltype(fake_labels)>::type;
        using samples_t = std::vector<etl::dyn_vector<typename Samples::value_type::value_type>>;

        //Convert data to an useful form
        samples_t data;
        data.reserve(training_data.size());

        for(auto& sample : training_data){
            data.emplace_back(sample);
        }

        //Compute the number of batches
        auto batches = data.size() / batch_size + (data.size() % batch_size == 0 ? 0 : 1);

        typename dbn_t::weight error = 0.0;

        //Train for max_epochs epoch
        for(size_t epoch= 0; epoch < max_epochs; ++epoch){
            //Train one mini-batch at a time
            for(size_t i = 0; i < batches; ++i){
                auto start = i * batch_size;
                auto end = std::min(start + batch_size, data.size());

                auto data_batch = make_batch(data.begin() + start, data.begin() + end);
                auto label_batch = make_batch(fake_labels.begin() + start, fake_labels.begin() + end);

                trainer->train_batch(epoch, data_batch, label_batch);
            }

            error = test_set(dbn, data, labels, [](dbn_t& dbn, auto& image){return dbn.predict(image);});

            //After some time increase the momentum
            if(dbn_traits<dbn_t>::has_momentum() && epoch == dbn.final_momentum_epoch){
                dbn.momentum = dbn.final_momentum;
            }

            watcher.ft_epoch_end(epoch, error, dbn);
        }

        watcher.fine_tuning_end(dbn);

        return error;
    }
};

} //end of dbn namespace

#endif