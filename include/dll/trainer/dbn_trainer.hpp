//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "cpp_utils/algorithm.hpp" // For parallel_shuffle

#include "etl/etl.hpp"

#include "dll/util/labels.hpp"
#include "dll/util/batch.hpp" // For make_batch
#include "dll/test.hpp"
#include "dll/dbn_traits.hpp"

namespace dll {

template <typename Iterator>
struct range {
private:
    Iterator first;
    Iterator last;

public:
    range(Iterator first, Iterator last)
            : first(first), last(last) {}

    Iterator begin() const {
        return first;
    }

    Iterator end() const {
        return last;
    }
};

template <typename Iterator>
range<Iterator> make_range(Iterator first, Iterator last) {
    return {first, last};
}

/*!
 * \brief A generic trainer for Deep Belief Network
 *
 * This trainer use the specified trainer of the DBN to perform supervised
 * fine-tuning.
 */
template <typename DBN>
struct dbn_trainer {
    using dbn_t = DBN;
    using error_type = typename dbn_t::weight;

    template <typename R>
    using trainer_t = typename dbn_t::desc::template trainer_t<R>;

    template <typename R>
    using watcher_t = typename dbn_t::desc::template watcher_t<R>;

    template <typename Iterator, typename LIterator>
    error_type train(DBN& dbn, Iterator first, Iterator last, LIterator lfirst, LIterator llast, std::size_t max_epochs) const {
        auto error_function = [&dbn, first, last, lfirst, llast]() {
            return test_set(dbn, first, last, lfirst, llast,
                            [](dbn_t& dbn, auto& image) { return dbn.predict(image); });
        };

        auto label_transformer = [](auto first, auto last) {
            return dll::make_fake(first, last);
        };

        auto input_transformer = [](const auto& /*value*/){
            // NOP
        };

        return train_impl(dbn, false, first, last, lfirst, llast, max_epochs, error_function, input_transformer, label_transformer);
    }

    template <typename Iterator>
    error_type train_ae(DBN& dbn, Iterator first, Iterator last, std::size_t max_epochs) const {
        auto error_function = [&dbn, first, last]() {
            return test_set_ae(dbn, first, last);
        };

        auto label_transformer = [](auto first, auto last) {
            return make_range(first, last);
        };

        auto input_transformer = [](const auto& /*value*/){
            // NOP
        };

        return train_impl(dbn, true, first, last, first, last, max_epochs, error_function, input_transformer, label_transformer);
    }

    template <typename Iterator>
    error_type train_dae(DBN& dbn, Iterator first, Iterator last, std::size_t max_epochs, double corrupt) const {
        auto error_function = [&dbn, first, last]() {
            return test_set_ae(dbn, first, last);
        };

        auto label_transformer = [](auto first, auto last) {
            return make_range(first, last);
        };

        auto input_transformer = [corrupt](auto&& value){
            static std::random_device rd;
            static std::default_random_engine g(rd());

            std::uniform_real_distribution<double> dist(0.0, 1000.0);

            for(auto& v :  value){
                v *= dist(g) < corrupt * 1000.0 ? 0.0 : 1.0;
            }
        };

        return train_impl(dbn, true, first, last, first, last, max_epochs, error_function, input_transformer, label_transformer);
    }

    template <typename Iterator, typename LIterator, typename Error, typename InputTransformer, typename LabelTransformer>
    error_type train_impl(DBN& dbn, bool ae, Iterator first, Iterator last, LIterator lfirst, LIterator llast, std::size_t max_epochs, Error error_function, InputTransformer input_transformer, LabelTransformer label_transformer) const {
        constexpr const auto batch_size     = std::decay_t<DBN>::batch_size;
        constexpr const auto big_batch_size = std::decay_t<DBN>::big_batch_size;

        //Get types for the batch
        using samples_t = std::vector<std::remove_cv_t<typename std::iterator_traits<Iterator>::value_type>>;
        using labels_t  = std::vector<std::remove_cv_t<typename std::iterator_traits<LIterator>::value_type>>;

        //Initialize the momentum
        dbn.momentum = dbn.initial_momentum;

        //Initialize the watcher
        watcher_t<dbn_t> watcher;

        watcher.fine_tuning_begin(dbn);

        auto trainer = std::make_unique<trainer_t<dbn_t>>(dbn);

        trainer->set_autoencoder(ae);

        //Initialize the trainer if necessary
        trainer->init_training(batch_size);

        error_type error = 0.0;

        if (!dbn.batch_mode()) {
            //Make sure data is contiguous
            samples_t data;
            data.reserve(std::distance(first, last));

            std::for_each(first, last, [&data](auto& sample) {
                data.emplace_back(sample);
            });

            //Convert labels to an useful form
            auto fake_labels = label_transformer(lfirst, llast);

            //Make sure labels are contiguous
            std::vector<std::decay_t<decltype(*fake_labels.begin())>> labels;
            labels.reserve(std::distance(fake_labels.begin(), fake_labels.end()));

            std::for_each(fake_labels.begin(), fake_labels.end(), [&labels](auto& label) {
                labels.emplace_back(label);
            });

            //Compute the number of batches
            auto batches = data.size() / batch_size + (data.size() % batch_size == 0 ? 0 : 1);

            //Train for max_epochs epoch
            for (std::size_t epoch = 0; epoch < max_epochs; ++epoch) {
                // Shuffle before the epoch if necessary
                if(dbn_traits<dbn_t>::shuffle()){
                    static std::random_device rd;
                    static std::mt19937_64 g(rd());

                    cpp::parallel_shuffle(data.begin(), data.end(), labels.begin(), labels.end(), g);
                }

                double loss = 0;

                //Train one mini-batch at a time
                for (std::size_t i = 0; i < batches; ++i) {
                    auto start = i * batch_size;
                    auto end   = std::min(start + batch_size, data.size());

                    auto data_batch  = make_batch(data.begin() + start, data.begin() + end);
                    auto label_batch = make_batch(labels.begin() + start, labels.begin() + end);

                    double batch_error;
                    double batch_loss;
                    std::tie(batch_error, batch_loss) = trainer->train_batch(epoch, data_batch, label_batch, input_transformer);

                    if(dbn_traits<dbn_t>::is_verbose()){
                        watcher.ft_batch_end(epoch, i, batches, batch_error, batch_loss, error_function(), dbn);
                    }

                    loss += batch_loss;
                }

                loss /= batches;

                auto last_error = error;
                error           = error_function();

                //After some time increase the momentum
                if (dbn_traits<dbn_t>::has_momentum() && epoch == dbn.final_momentum_epoch) {
                    dbn.momentum = dbn.final_momentum;
                }

                watcher.ft_epoch_end(epoch, error, loss, dbn);

                //Once the goal is reached, stop training
                if (error <= dbn.goal) {
                    break;
                }

                if (dbn_traits<dbn_t>::lr_driver() == lr_driver_type::BOLD) {
                    if (epoch) {
                        if (error > last_error + 1e-8) {
                            //Error increased
                            dbn.learning_rate *= dbn.lr_bold_dec;
                            watcher.lr_adapt(dbn);
                            dbn.restore_weights();
                        } else if (error < last_error - 1e-10) {
                            //Error decreased
                            dbn.learning_rate *= dbn.lr_bold_inc;
                            watcher.lr_adapt(dbn);
                            dbn.backup_weights();
                        } else {
                            //Error didn't change enough
                            dbn.backup_weights();
                        }
                    } else {
                        dbn.backup_weights();
                    }
                }

                if (dbn_traits<dbn_t>::lr_driver() == lr_driver_type::STEP) {
                    if (epoch && epoch % dbn.lr_step_size == 0) {
                        dbn.learning_rate *= dbn.lr_step_gamma;
                        watcher.lr_adapt(dbn);
                    }
                }
            }
        } else {
            auto total_batch_size = big_batch_size * batch_size;

            //Prepare some space for converted data
            samples_t input_cache(total_batch_size);
            labels_t label_cache(total_batch_size);

            //Train for max_epochs epoch
            for (std::size_t epoch = 0; epoch < max_epochs; ++epoch) {
                auto it = first;
                auto lit = lfirst;

                double loss = 0.0;

                std::size_t n = 0;

                //Train all mini-batches
                while (it != last) {
                    //Fill the input caches
                    std::size_t i = 0;
                    while (it != last && i < total_batch_size) {
                        input_cache[i] = *it++;
                        label_cache[i] = *lit++;
                        ++i;
                        ++n;
                    }

                    //Convert labels to an useful form
                    auto fake_labels = label_transformer(label_cache.begin(), label_cache.end());

                    auto full_batches = i / batch_size;

                    //Train all the full batches
                    for (std::size_t b = 0; b < full_batches; ++b) {
                        auto data_batch  = make_batch(input_cache.begin() + b * batch_size, input_cache.begin() + (b + 1) * batch_size);
                        auto label_batch = make_batch(fake_labels.begin() + b * batch_size, fake_labels.begin() + (b + 1) * batch_size);

                        double batch_error;
                        double batch_loss;
                        std::tie(batch_error, batch_loss) = trainer->train_batch(epoch, data_batch, label_batch, input_transformer);

                        if(dbn_traits<dbn_t>::is_verbose()){
                            watcher.ft_batch_end(epoch, batch_error, batch_loss, error_function(), dbn);
                        }

                        loss += batch_loss;
                    }

                    //Train the last incomplete batch, if any
                    if (i % batch_size > 0) {
                        auto data_batch  = make_batch(input_cache.begin() + full_batches * batch_size, input_cache.begin() + i);
                        auto label_batch = make_batch(fake_labels.begin() + full_batches * batch_size, fake_labels.begin() + i);

                        double batch_error;
                        double batch_loss;
                        std::tie(batch_error, batch_loss) = trainer->train_batch(epoch, data_batch, label_batch, input_transformer);

                        if(dbn_traits<dbn_t>::is_verbose()){
                            watcher.ft_batch_end(epoch, batch_error, batch_loss, error_function(), dbn);
                        }

                        loss += batch_loss;
                    }
                }

                error = error_function();
                loss /= n;

                //After some time increase the momentum
                if (dbn_traits<dbn_t>::has_momentum() && epoch == dbn.final_momentum_epoch) {
                    dbn.momentum = dbn.final_momentum;
                }

                watcher.ft_epoch_end(epoch, error, loss, dbn);

                //Once the goal is reached, stop training
                if (error <= dbn.goal) {
                    break;
                }
            }
        }

        watcher.fine_tuning_end(dbn);

        return error;
    }
};

} //end of dll namespace
