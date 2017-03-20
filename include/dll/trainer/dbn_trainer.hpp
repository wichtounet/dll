//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "cpp_utils/algorithm.hpp" // For parallel_shuffle

#include "etl/etl.hpp"

#include "dll/util/labels.hpp"
#include "dll/util/timers.hpp"
#include "dll/util/batch.hpp" // For make_batch
#include "dll/test.hpp"
#include "dll/dbn_traits.hpp"

#include "dll/trainer/sgd_context.hpp" // TODO Remove later

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
    error_type train(DBN& dbn, Iterator first, Iterator last, LIterator lfirst, LIterator llast, size_t max_epochs) const {
        auto error_function = [&dbn, first, last, lfirst, llast]() {
            return test_set(dbn, first, last, lfirst, llast,
                            [](dbn_t& dbn, auto& image) { return dbn.predict(image); });
        };

        auto label_transformer = [](const auto& value, size_t n) {
            return dll::make_fake_etl(value, n);
        };

        auto input_transformer = [](const auto& /*value*/){
            // NOP
        };

        return train_impl(dbn, false, first, last, lfirst, llast, max_epochs, error_function, input_transformer, label_transformer);
    }

    template <typename Iterator>
    error_type train_ae(DBN& dbn, Iterator first, Iterator last, size_t max_epochs) const {
        auto error_function = [&dbn, first, last]() {
            return test_set_ae(dbn, first, last);
        };

        auto label_transformer = [](const auto& value, size_t /*n*/) {
            return value;
        };

        auto input_transformer = [](const auto& /*value*/){
            // NOP
        };

        return train_impl(dbn, true, first, last, first, last, max_epochs, error_function, input_transformer, label_transformer);
    }

    template <typename Iterator>
    error_type train_dae(DBN& dbn, Iterator first, Iterator last, size_t max_epochs, double corrupt) const {
        auto error_function = [&dbn, first, last]() {
            return test_set_ae(dbn, first, last);
        };

        auto label_transformer = [](const auto& value, size_t /*n*/) {
            return value;
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

    template <typename D, typename It>
    static void copy_inputs(D& dest, It first, It last) {
        size_t i = 0;

        while (first != last) {
            dest(i++) = *first++;
        }
    }

    template <typename Iterator, typename LIterator, typename Error, typename InputTransformer, typename LabelTransformer>
    error_type train_impl(DBN& dbn, bool ae, Iterator first, Iterator last, LIterator lfirst, LIterator /*llast*/, size_t max_epochs, Error error_function, InputTransformer input_transformer, LabelTransformer label_transformer) const {
        dll::auto_timer timer("dbn::trainer::train_impl");

        decltype(auto) input_layer = dbn.template layer_get<dbn_t::input_layer_n>();
        decltype(auto) output_layer = dbn.template layer_get<dbn_t::output_layer_n>();

        using input_layer_t  = typename dbn_t::template layer_type<dbn_t::input_layer_n>;
        using output_layer_t = typename dbn_t::template layer_type<dbn_t::output_layer_n>;

        constexpr const auto batch_size     = std::decay_t<DBN>::batch_size;
        constexpr const auto big_batch_size = std::decay_t<DBN>::big_batch_size;

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
            dll::auto_timer timer("dbn::trainer::train_impl::fast");

            // The number of elements on which to train
            const size_t n = std::distance(first, last);

            // Prepare the data

            // TODO Create correctly the type for conv
            etl::dyn_matrix<typename input_layer_t::weight, 2> data(n, input_layer.input_size());

            for(size_t l = 0; l < n; ++l){
                data(l) = *first++;
            }

            // Prepare the labels

            etl::dyn_matrix<typename output_layer_t::weight, 2> labels(n, output_layer.output_size(), typename output_layer_t::weight(0.0));

            for(size_t l = 0; l < n; ++l){
                labels(l) = label_transformer(*lfirst++, output_layer.output_size());
            }

            //Compute the number of batches
            const auto batches = n / batch_size + (n % batch_size == 0 ? 0 : 1);

            // Prepare a fast error function using the contiguous memory

            auto error_function_batch = [&] () {
                double error = 0.0;

                // Compute the error on one mini-batch at a time
                for (size_t i = 0; i < n / batch_size; ++i) {
                    const size_t start = i * batch_size;
                    const size_t end   = start + batch_size;

                    decltype(auto) output = dbn.forward_batch(slice(data, start, end));

                    if(ae){
                        for(size_t b = 0; b < end - start; ++b){
                            error += mean(abs(labels(start + b) - output(b)));
                        }
                    } else {
                        // TODO Review this calculation
                        // The result is correct, but can probably be done in a more clean way

                        for(size_t b = 0; b < end - start; ++b){
                            error += std::min(1.0, (double) etl::sum(abs(labels(start + b) - one_if_max(output(b)))));
                        }
                    }
                }

                // Complete the computation for incomplete batches
                if (n % batch_size > 0) {
                    const auto start = n - n % batch_size;
                    const auto end   = n;

                    for(size_t i = start; i < end; ++i){
                        decltype(auto) output = dbn.forward(data(i));

                        if(ae){
                            error += mean(abs(labels(i) - output));
                        } else {
                            // TODO Review this calculation
                            // The result is correct, but can probably be done in a more clean way

                            error += std::min(1.0, (double) etl::sum(abs(labels(i) - one_if_max(output))));
                        }
                    }
                }

                return error / n;
            };

            //Train for max_epochs epoch
            for (size_t epoch = 0; epoch < max_epochs; ++epoch) {
                dll::auto_timer timer("dbn::trainer::train_impl::epoch");

                // Shuffle before the epoch if necessary
                if(dbn_traits<dbn_t>::shuffle()){
                    static std::random_device rd;
                    static std::mt19937_64 g(rd());

                    etl::parallel_shuffle(data, labels);
                }

                double loss = 0;

                //Train one mini-batch at a time
                for (size_t i = 0; i < batches; ++i) {
                    dll::auto_timer timer("dbn::trainer::train_impl::epoch::batch");

                    const auto start = i * batch_size;
                    const auto end   = std::min(start + batch_size, n);

                    double batch_error;
                    double batch_loss;
                    std::tie(batch_error, batch_loss) = trainer->train_batch(
                        epoch,
                        slice(data, start, end),
                        slice(labels, start, end),
                        input_transformer);

                    if(dbn_traits<dbn_t>::is_verbose()){
                        watcher.ft_batch_end(epoch, i, batches, batch_error, batch_loss, error_function_batch(), dbn);
                    }

                    loss += batch_loss;
                }

                loss /= batches;

                // Save the last error
                auto last_error = error;

                // Compute the error at this epoch

                {
                    dll::auto_timer timer("dbn::trainer::train_impl::epoch::error");

                    error = error_function_batch();
                }

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
            etl::dyn_matrix<typename input_layer_t::weight, 2> input_cache(total_batch_size, input_layer.input_size());
            etl::dyn_matrix<typename output_layer_t::weight, 2> label_cache(total_batch_size, output_layer.output_size());

            //Train for max_epochs epoch
            for (size_t epoch = 0; epoch < max_epochs; ++epoch) {
                auto it = first;
                auto lit = lfirst;

                double loss = 0.0;

                size_t n = 0;

                //Train all mini-batches
                while (it != last) {
                    label_cache = 0.0;

                    //Fill the input caches
                    size_t i = 0;
                    while (it != last && i < total_batch_size) {
                        input_cache(i) = *it++;
                        label_cache(i) = label_transformer(*lit++, output_layer.output_size());

                        ++i;
                        ++n;
                    }

                    auto full_batches = i / batch_size;

                    //Train all the full batches
                    for (size_t b = 0; b < full_batches; ++b) {
                        const auto start       = b * batch_size;
                        const auto end         = (b + 1) * batch_size;

                        double batch_error;
                        double batch_loss;
                        std::tie(batch_error, batch_loss) = trainer->train_batch(
                            epoch,
                            slice(input_cache, start, end),
                            slice(label_cache, start, end),
                            input_transformer);

                        if(dbn_traits<dbn_t>::is_verbose()){
                            watcher.ft_batch_end(epoch, batch_error, batch_loss, error_function(), dbn);
                        }

                        loss += batch_loss;
                    }

                    //Train the last incomplete batch, if any
                    if (i % batch_size > 0) {
                        const auto start       = full_batches * batch_size;
                        const auto end         = i;

                        double batch_error;
                        double batch_loss;
                        std::tie(batch_error, batch_loss) = trainer->train_batch(
                            epoch,
                            slice(input_cache, start, end),
                            slice(label_cache, start, end),
                            input_transformer);

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
