//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DBN_DBN_TRAINER_HPP
#define DBN_DBN_TRAINER_HPP

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
    using trainer_t = typename dbn_t::template trainer_t<R>;

    template<typename R>
    using watcher_t = typename dbn_t::template watcher_t<R>;

    template<typename Label>
    typename dbn_t::weight train(DBN& dbn, const std::vector<vector<weight>>& training_data, std::vector<Label>& labels, size_t max_epochs, size_t batch_size) const {
        watcher_t<dbn_t> watcher;

        watcher.training_begin(dbn);

        auto trainer = make_unique<trainer_t<dbn_t>>(dbn);

        //Compute the number of batches
        auto batches = training_data.size() / batch_size + (training_data.size() % batch_size == 0 ? 0 : 1);

        //Train for max_epochs epoch
        for(size_t epoch= 0; epoch < max_epochs; ++epoch){
            //Train one mini-batch at a time
            for(size_t i = 0; i < batches; ++i){
                auto start = i * batch_size;
                auto end = std::min(start + batch_size, training_data.size());

                dll::batch<vector<typename dbn_t::weight>> data_batch(training_data.begin() + start, training_data.begin() + end);
                dll::batch<Label> label_batch(labels.begin() + start, labels.begin() + end),

                trainer->train_batch(epoch, data_batch, label_batch, dbn);
            }

            watcher.epoch_end(epoch, dbn);
        }

        watcher.training_end(rbm);

        return 0.0; //TODO
    }
};

} //end of dbn namespace

#endif