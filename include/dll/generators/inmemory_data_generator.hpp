//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <atomic>
#include <thread>

namespace dll {

/*!
 * \brief a in-memory data generator
 */
template<typename Iterator, typename LIterator, typename Desc, typename Enable = void>
struct inmemory_data_generator;

/*!
 * \copydoc inmemory_data_generator
 */
template<typename Iterator, typename LIterator, typename Desc>
struct inmemory_data_generator <Iterator, LIterator, Desc, std::enable_if_t<!is_augmented<Desc>::value>> {
    using desc = Desc;
    using weight = etl::value_t<typename Iterator::value_type>;
    using data_cache_helper_t = cache_helper<Desc, Iterator>;
    using label_cache_helper_t = label_cache_helper<Desc, weight, LIterator>;

    using data_cache_type = typename data_cache_helper_t::cache_type;
    using label_cache_type = typename label_cache_helper_t::cache_type;

    static constexpr bool dll_generator = true;

    static constexpr size_t batch_size = desc::BatchSize;

    data_cache_type input_cache;
    label_cache_type label_cache;

    size_t current = 0;

    inmemory_data_generator(Iterator first, Iterator last, LIterator lfirst, LIterator llast, size_t n_classes){
        const size_t n = std::distance(first, last);

        data_cache_helper_t::init(n, first, input_cache);
        label_cache_helper_t::init(n, n_classes, lfirst, label_cache);

        size_t i = 0;
        while(first != last){
            input_cache(i) = *first;

            pre_scaler<desc>::transform(input_cache(i));
            pre_normalizer<desc>::transform(input_cache(i));
            pre_binarizer<desc>::transform(input_cache(i));

            label_cache_helper_t::set(i, lfirst, label_cache);

            // In case of auto-encoders, the label images also need to be transformed
            cpp::static_if<desc::AutoEncoder>([&](auto f){
                pre_scaler<desc>::transform(f(label_cache)(i));
                pre_normalizer<desc>::transform(f(label_cache)(i));
                pre_binarizer<desc>::transform(f(label_cache)(i));
            });

            ++i;
            ++first;
            ++lfirst;
        }

        cpp_unused(llast);
    }

    inmemory_data_generator(const inmemory_data_generator& rhs) = delete;
    inmemory_data_generator operator=(const inmemory_data_generator& rhs) = delete;

    inmemory_data_generator(inmemory_data_generator&& rhs) = delete;
    inmemory_data_generator operator=(inmemory_data_generator&& rhs) = delete;

    void set_test(){
        // Nothing to do
    }

    void set_train(){
        // Nothing to do
    }

    /*!
     * \brief Reset the generator to the beginning
     */
    void reset(){
        current = 0;
    }

    /*
     * \brief Reset the generator and shuffle the order of samples
     */
    void reset_shuffle(){
        current = 0;
        shuffle();
    }

    /*!
     * \brief Shuffle the order of the samples.
     *
     * This should only be done when the generator is at the beginning.
     */
    void shuffle(){
        cpp_assert(!current, "Shuffle should only be performed on start of generation");

        etl::parallel_shuffle(input_cache, label_cache);
    }

    /*!
     * \brief Return the index of the current batch in the generation
     * \return The current batch index
     */
    size_t current_batch() const {
        return current / batch_size;
    }

    /*
     * \brief Returns the number of elements in the generator
     * \return The number of elements in the generator
     */
    size_t size() const {
        return etl::dim<0>(input_cache);
    }

    /*
     * \brief Returns the augmented number of elements in the generator.
     *
     * This number may be an estimate, depending on which augmentation
     * techniques are enabled.
     *
     * \return The augmented number of elements in the generator
     */
    size_t augmented_size() const {
        return etl::dim<0>(input_cache);
    }

    /*
     * \brief Returns the number of batches in the generator.
     * \return The number of batches in the generator
     */
    size_t batches() const {
        return size() / batch_size + (size() % batch_size == 0 ? 0 : 1);
    }

    /*
     * \brief Indicates if the generator has a next batch or not
     * \return true if the generator has a next batch, false otherwise
     */
    bool has_next_batch() const {
        return current < size() - 1;
    }

    /*!
     * \brief Moves to the next batch.
     *
     * This should only be called if the generator has a next batch.
     */
    void next_batch() {
        current += batch_size;
    }

    /*!
     * \brief Returns the current data batch
     * \return a a batch of data.
     */
    auto data_batch() const {
        return etl::slice(input_cache, current, std::min(current + batch_size, size()));
    }

    /*!
     * \brief Returns the current label batch
     * \return a a batch of label.
     */
    auto label_batch() const {
        return etl::slice(label_cache, current, std::min(current + batch_size, size()));
    }

    /*!
     * \brief Returns the number of dimensions of the input.
     * \return The number of dimensions of the input.
     */
    static constexpr size_t dimensions() {
        return etl::dimensions<data_cache_type>() - 1;
    }
};

/*!
 * \copydoc inmemory_data_generator
 */
template<typename Iterator, typename LIterator, typename Desc>
struct inmemory_data_generator <Iterator, LIterator, Desc, std::enable_if_t<is_augmented<Desc>::value>> {
    using desc = Desc;
    using weight = etl::value_t<typename Iterator::value_type>;
    using data_cache_helper_t = cache_helper<desc, Iterator>;
    using label_cache_helper_t = label_cache_helper<desc, weight, LIterator>;

    using data_cache_type  = typename data_cache_helper_t::cache_type;
    using big_cache_type   = typename data_cache_helper_t::big_cache_type;
    using label_cache_type = typename label_cache_helper_t::cache_type;

    static constexpr bool dll_generator = true;

    static constexpr size_t batch_size = desc::BatchSize;
    static constexpr size_t big_batch_size = desc::BigBatchSize;

    data_cache_type input_cache;
    big_cache_type batch_cache;
    label_cache_type label_cache;

    random_cropper<Desc> cropper;
    random_mirrorer<Desc> mirrorer;
    elastic_distorter<Desc> distorter;
    random_noise<Desc> noiser;

    size_t current = 0;

    mutable volatile bool status[big_batch_size];
    mutable volatile size_t indices[big_batch_size];

    mutable std::mutex main_lock;
    mutable std::condition_variable condition;
    mutable std::condition_variable ready_condition;

    volatile bool stop_flag = false;

    std::thread main_thread;
    bool train_mode = false;

    inmemory_data_generator(Iterator first, Iterator last, LIterator lfirst, LIterator llast, size_t n_classes) : cropper(*first), mirrorer(*first), distorter(*first), noiser(*first) {
        const size_t n = std::distance(first, last);

        data_cache_helper_t::init(n, first, input_cache);
        data_cache_helper_t::init_big(big_batch_size, batch_size, first, batch_cache);

        label_cache_helper_t::init(n, n_classes, lfirst, label_cache);

        size_t i = 0;
        while(first != last){
            input_cache(i) = *first;

            pre_scaler<desc>::transform(input_cache(i));
            pre_normalizer<desc>::transform(input_cache(i));
            pre_binarizer<desc>::transform(input_cache(i));

            label_cache_helper_t::set(i, lfirst, label_cache);

            // In case of auto-encoders, the label images also need to be transformed
            cpp::static_if<desc::AutoEncoder>([&](auto f){
                pre_scaler<desc>::transform(f(label_cache)(i));
                pre_normalizer<desc>::transform(f(label_cache)(i));
                pre_binarizer<desc>::transform(f(label_cache)(i));
            });

            ++i;
            ++first;
            ++lfirst;
        }

        for(size_t b = 0; b < big_batch_size; ++b){
            status[b] = false;
            indices[b] = b;
        }

        cpp_unused(llast);

        main_thread = std::thread([this]{
            while(true){
                // The index of the batch inside the batch cache
                size_t index = 0;

                {
                    std::unique_lock<std::mutex> ulock(main_lock);

                    bool found = false;

                    // Try to find a read batch first
                    for(size_t b = 0; b < big_batch_size; ++b){
                        if(!status[b] && indices[b] * batch_size < size()){
                            index = b;
                            found = true;
                            break;
                        }
                    }

                    // if not found, wait for a ready to compute batch

                    if(!found){
                        // Wait for the end or for some work
                        condition.wait(ulock, [this, &index] {
                            if(stop_flag){
                                return true;
                            }

                            for(size_t b = 0; b < big_batch_size; ++b){
                                if(!status[b] && indices[b] * batch_size < size()){
                                    index = b;
                                    return true;
                                }
                            }

                            return false;
                        });

                        // If there is no more thread for the thread, exit
                        if (stop_flag) {
                            return;
                        }
                    }
                }

                // Get the batch that needs to be read
                const size_t batch = indices[index];

                // Get the index from where to read inside the input cache
                const size_t input_n = batch * batch_size;

                for(size_t i = 0; i < batch_size && input_n < size(); ++i){
                    if(train_mode){
                        // Random crop the image
                        cropper.transform_first(batch_cache(index)(i), input_cache(input_n + i));

                        // Mirror the image
                        mirrorer.transform(batch_cache(index)(i));

                        // Distort the image
                        distorter.transform(batch_cache(index)(i));

                        // Noise the image
                        noiser.transform(batch_cache(index)(i));
                    } else {
                        // Center crop the image
                        cropper.transform_first_test(batch_cache(index)(i), input_cache(input_n + i));
                    }
                }

                // Notify a waiter that one batch is ready

                {
                    std::unique_lock<std::mutex> ulock(main_lock);

                    status[index] = true;

                    ready_condition.notify_one();
                }
            }
        });
    }

    inmemory_data_generator(const inmemory_data_generator& rhs) = delete;
    inmemory_data_generator operator=(const inmemory_data_generator& rhs) = delete;

    inmemory_data_generator(inmemory_data_generator&& rhs) = delete;
    inmemory_data_generator operator=(inmemory_data_generator&& rhs) = delete;

    void set_test(){
        train_mode = false;
    }

    void set_train(){
        train_mode = true;
    }

    ~inmemory_data_generator(){
        cpp::with_lock(main_lock, [this] { stop_flag = true; });

        condition.notify_all();

        main_thread.join();
    }

    void reset_generation(){
        std::unique_lock<std::mutex> ulock(main_lock);

        for(size_t b = 0; b < big_batch_size; ++b){
            status[b] = false;
            indices[b] = b;
        }

        condition.notify_one();
    }

    /*!
     * \brief Reset the generator to the beginning
     */
    void reset(){
        current = 0;
        reset_generation();
    }

    /*
     * \brief Reset the generator and shuffle the order of samples
     */
    void reset_shuffle(){
        current = 0;
        shuffle();
        reset_generation();
    }

    /*!
     * \brief Shuffle the order of the samples.
     *
     * This should only be done when the generator is at the beginning.
     */
    void shuffle(){
        cpp_assert(!current, "Shuffle should only be performed on start of generation");

        etl::parallel_shuffle(input_cache, label_cache);
    }

    /*!
     * \brief Return the index of the current batch in the generation
     * \return The current batch index
     */
    size_t current_batch() const {
        return current / batch_size;
    }

    /*
     * \brief Returns the number of elements in the generator
     * \return The number of elements in the generator
     */
    size_t size() const {
        return etl::dim<0>(input_cache);
    }

    /*
     * \brief Returns the augmented number of elements in the generator.
     *
     * This number may be an estimate, depending on which augmentation
     * techniques are enabled.
     *
     * \return The augmented number of elements in the generator
     */
    size_t augmented_size() const {
        return cropper.scaling() * mirrorer.scaling() * noiser.scaling() * distorter.scaling() * etl::dim<0>(input_cache);
    }

    /*
     * \brief Returns the number of batches in the generator.
     * \return The number of batches in the generator
     */
    size_t batches() const {
        return size() / batch_size + (size() % batch_size == 0 ? 0 : 1);
    }

    /*
     * \brief Indicates if the generator has a next batch or not
     * \return true if the generator has a next batch, false otherwise
     */
    bool has_next_batch() const {
        return current < size() - 1;
    }

    /*!
     * \brief Moves to the next batch.
     *
     * This should only be called if the generator has a next batch.
     */
    void next_batch() {
        // Get information from batch that has been consumed
        const auto batch = current / batch_size;
        const auto b = batch % big_batch_size;

        {
            std::unique_lock<std::mutex> ulock(main_lock);

            status[b] = false;
            indices[b] += big_batch_size;

            condition.notify_one();
        }

        current += batch_size;
    }

    /*!
     * \brief Returns the current data batch
     * \return a a batch of data.
     */
    auto data_batch() const {
        std::unique_lock<std::mutex> ulock(main_lock);

        const auto batch = current / batch_size;
        const auto b = batch % big_batch_size;

        if(status[b]){
            return batch_cache(b);
        }

        ready_condition.wait(ulock, [this, b] {
            return status[b];
        });

        return batch_cache(b);
    }

    /*!
     * \brief Returns the current label batch
     * \return a a batch of label.
     */
    auto label_batch() const {
        return etl::slice(label_cache, current, std::min(current + batch_size, size()));
    }

    /*!
     * \brief Returns the number of dimensions of the input.
     * \return The number of dimensions of the input.
     */
    static constexpr size_t dimensions() {
        return etl::dimensions<data_cache_type>() - 1;
    }
};

/*!
 * \brief Descriptor for a inmemory_data_generator
 */
template <typename... Parameters>
struct inmemory_data_generator_desc {
    /*!
     * A list of all the parameters of the descriptor
     */
    using parameters = cpp::type_list<Parameters...>;

    /*!
     * \brief The size of a batch
     */
    static constexpr size_t BatchSize = detail::get_value<batch_size<1>, Parameters...>::value;

    /*!
     * \brief The number of batch in cache
     */
    static constexpr size_t BigBatchSize = detail::get_value<big_batch_size<1>, Parameters...>::value;

    /*!
     * \brief Indicates if the generators must make the labels categorical
     */
    static constexpr bool Categorical = parameters::template contains<categorical>();

    /*!
     * \brief Indicates if horizontal mirroring should be used as augmentation.
     */
    static constexpr bool HorizontalMirroring = parameters::template contains<horizontal_mirroring>();

    /*!
     * \brief Indicates if vertical mirroring should be used as augmentation.
     */
    static constexpr bool VerticalMirroring = parameters::template contains<vertical_mirroring>();

    /*!
     * \brief The random cropping X
     */
    static constexpr size_t random_crop_x = detail::get_value_1<random_crop<0,0>, Parameters...>::value;

    /*!
     * \brief The random cropping Y
     */
    static constexpr size_t random_crop_y = detail::get_value_2<random_crop<0,0>, Parameters...>::value;

    /*!
     * \brief The elastic distortion kernel
     */
    static constexpr size_t ElasticDistortion = detail::get_value<elastic_distortion<0>, Parameters...>::value;

    /*!
     * \brief The noise
     */
    static constexpr size_t Noise = detail::get_value<noise<0>, Parameters...>::value;

    /*!
     * \brief The scaling
     */
    static constexpr size_t ScalePre = detail::get_value<scale_pre<0>, Parameters...>::value;

    /*!
     * \brief The scaling
     */
    static constexpr size_t BinarizePre = detail::get_value<binarize_pre<0>, Parameters...>::value;

    /*!
     * \brief Indicates if input are normalized
     */
    static constexpr bool NormalizePre = parameters::template contains<normalize_pre>();

    /*!
     * \brief Indicates if this is an auto-encoder task
     */
    static constexpr bool AutoEncoder = parameters::template contains<autoencoder>();

    static_assert(BatchSize > 0, "The batch size must be larger than one");

    //Make sure only valid types are passed to the configuration list
    static_assert(
        detail::is_valid<cpp::type_list<batch_size_id, big_batch_size_id, horizontal_mirroring_id, vertical_mirroring_id, random_crop_id, elastic_distortion_id, categorical_id, noise_id, nop_id, normalize_pre_id, binarize_pre_id, scale_pre_id, autoencoder_id>,
                         Parameters...>::value,
        "Invalid parameters type for rbm_desc");

    /*!
     * The generator type
     */
    template<typename Iterator, typename LIterator>
    using generator_t = inmemory_data_generator<Iterator, LIterator, inmemory_data_generator_desc<Parameters...>>;
};

template<typename Iterator, typename LIterator, typename... Parameters>
auto make_generator(Iterator first, Iterator last, LIterator lfirst, LIterator llast, size_t n_classes, const inmemory_data_generator_desc<Parameters...>& /*desc*/){
    using generator_t = typename inmemory_data_generator_desc<Parameters...>::template generator_t<Iterator, LIterator>;
    return std::make_unique<generator_t>(first, last, lfirst, llast, n_classes);
}

template<typename Container, typename LContainer, typename... Parameters>
auto make_generator(const Container& container, const LContainer& lcontainer, size_t n_classes, const inmemory_data_generator_desc<Parameters...>& /*desc*/){
    using generator_t = typename inmemory_data_generator_desc<Parameters...>::template generator_t<typename Container::const_iterator, typename LContainer::const_iterator>;
    return std::make_unique<generator_t>(container.begin(), container.end(), lcontainer.begin(), lcontainer.end(), n_classes);
}

// The following are simply helpers for creating generic generators

template<typename Iterator, typename LIterator, typename... Parameters>
auto make_generator(Iterator first, Iterator last, LIterator lfirst, LIterator llast, size_t n, size_t n_classes, const inmemory_data_generator_desc<Parameters...>& /*desc*/){
    cpp_unused(n);

    using generator_t = typename inmemory_data_generator_desc<Parameters...>::template generator_t<Iterator, LIterator>;
    return std::make_unique<generator_t>(first, last, lfirst, llast, n_classes);
}

template<typename Container, typename LContainer, typename... Parameters>
auto make_generator(const Container& container, const LContainer& lcontainer, size_t n, size_t n_classes, const inmemory_data_generator_desc<Parameters...>& /*desc*/){
    cpp_unused(n);

    using generator_t = typename inmemory_data_generator_desc<Parameters...>::template generator_t<typename Container::const_iterator, typename LContainer::const_iterator>;
    return std::make_unique<generator_t>(container.begin(), container.end(), lcontainer.begin(), lcontainer.end(), n_classes);
}

} //end of dll namespace
