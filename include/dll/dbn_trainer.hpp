//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_DBN_TRAINER_HPP
#define DLL_DBN_TRAINER_HPP

#include "etl/etl.hpp"

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

    template<typename Iterator, typename LIterator>
    typename dbn_t::weight train(DBN& dbn, Iterator first, Iterator last, LIterator lfirst, LIterator llast, std::size_t max_epochs) const {
        constexpr const auto batch_size = std::decay_t<DBN>::batch_size;
        constexpr const auto big_batch_size = std::decay_t<DBN>::big_batch_size;

        //Get types for the batch
        using samples_t = std::vector<typename std::iterator_traits<Iterator>::value_type>;
        using labels_t = std::vector<typename std::iterator_traits<LIterator>::value_type>;

        //Initialize the momentum
        dbn.momentum = dbn.initial_momentum;

        //Initialize the watcher
        watcher_t<dbn_t> watcher;

        watcher.fine_tuning_begin(dbn);

        auto trainer = std::make_unique<trainer_t<dbn_t>>(dbn);

        //Initialize the trainer if necessary
        trainer->init_training(batch_size);

        typename dbn_t::weight error = 0.0;

        if(!dbn_traits<dbn_t>::save_memory()){
            //Convert labels to an useful form
            auto fake_labels = dll::make_fake(lfirst, llast);

            //Make sure data is contiguous
            samples_t data;
            data.reserve(std::distance(first, last));

            std::for_each(first, last, [&data](auto& sample){
                data.emplace_back(sample);
            });

            //Compute the number of batches
            auto batches = data.size() / batch_size + (data.size() % batch_size == 0 ? 0 : 1);

            //Train for max_epochs epoch
            for(std::size_t epoch= 0; epoch < max_epochs; ++epoch){
                //Train one mini-batch at a time
                for(std::size_t i = 0; i < batches; ++i){
                    auto start = i * batch_size;
                    auto end = std::min(start + batch_size, data.size());

                    auto data_batch = make_batch(data.begin() + start, data.begin() + end);
                    auto label_batch = make_batch(fake_labels.begin() + start, fake_labels.begin() + end);

                    trainer->train_batch(epoch, data_batch, label_batch);
                }

                error = test_set(dbn, first, last, lfirst, llast,
                    [](dbn_t& dbn, auto& image){ return dbn.predict(image); });

                //After some time increase the momentum
                if(dbn_traits<dbn_t>::has_momentum() && epoch == dbn.final_momentum_epoch){
                    dbn.momentum = dbn.final_momentum;
                }

                watcher.ft_epoch_end(epoch, error, dbn);

                //Once the goal is reached, stop training
                if(error == 0.0){
                    break;
                }
            }
        } else {
            auto total_batch_size = big_batch_size * batch_size;

            //Prepare some space for converted data
            samples_t input_cache(total_batch_size);
            labels_t label_cache(total_batch_size);

            //Train for max_epochs epoch
            for(std::size_t epoch= 0; epoch < max_epochs; ++epoch){
                auto it = first;

                auto lit = lfirst;

                //Train all mini-batches
                while(it != last){
                    //Fill the input caches
                    std::size_t i = 0;
                    while(it != last && i < total_batch_size){
                        input_cache[i] = *it++;
                        label_cache[i] = *lit++;
                        ++i;
                    }

                    //Convert labels to an useful form
                    auto fake_labels = dll::make_fake(label_cache.begin(), label_cache.end());

                    auto full_batches = i / batch_size;

                    //Train all the full batches
                    for(std::size_t b = 0; b < full_batches; ++b){
                        auto data_batch = make_batch(input_cache.begin() + b * batch_size, input_cache.begin() + (b+1) * batch_size);
                        auto label_batch = make_batch(fake_labels.begin() + b * batch_size, fake_labels.begin() + (b+1) * batch_size);

                        trainer->train_batch(epoch, data_batch, label_batch);
                    }

                    //Train the last incomplete batch, if any
                    if(i % batch_size > 0){
                        auto data_batch = make_batch(input_cache.begin() + full_batches * batch_size, input_cache.begin() + i);
                        auto label_batch = make_batch(fake_labels.begin() + full_batches * batch_size, fake_labels.begin()+ i);

                        trainer->train_batch(epoch, data_batch, label_batch);
                    }
                }

                error = test_set(dbn, first, last, lfirst, llast,
                    [](dbn_t& dbn, auto& image){ return dbn.predict(image); });

                //After some time increase the momentum
                if(dbn_traits<dbn_t>::has_momentum() && epoch == dbn.final_momentum_epoch){
                    dbn.momentum = dbn.final_momentum;
                }

                watcher.ft_epoch_end(epoch, error, dbn);

                //Once the goal is reached, stop training
                if(error == 0.0){
                    break;
                }
            }
        }

        watcher.fine_tuning_end(dbn);

        return error;
    }
};

} //end of dll namespace

#endif
