//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Implementation of an in-memory single data generator
 */

#pragma once

#include <atomic>
#include <thread>

namespace dll {

/*!
 * \brief a in-memory data generator
 */
template <typename Iterator, typename Desc, typename Enable = void>
struct inmemory_single_data_generator;

/*!
 * \copydoc inmemory_single_data_generator
 */
template <typename Iterator, typename Desc>
struct inmemory_single_data_generator<Iterator, Desc, std::enable_if_t<!is_augmented<Desc>>> {
    using desc                 = Desc;                                                              ///< The generator descriptor
    using weight               = etl::value_t<typename std::iterator_traits<Iterator>::value_type>; ///< The data type
    using data_cache_helper_t  = cache_helper<Desc, Iterator>;                                      ///< The helper for the data cache

    using data_cache_type  = typename data_cache_helper_t::cache_type;  ///< The type of the data cache

    static constexpr bool dll_generator = true; ///< Simple flag to indicate that the class is a DLL generator

    static constexpr inline size_t batch_size = desc::BatchSize; ///< The size of the generated batches

    data_cache_type input_cache;  ///< The input cache

    size_t current = 0;     ///< The current index
    bool is_safe   = false; ///< Indicates if the generator is safe to reclaim memory from

    template <typename Input>
    inmemory_single_data_generator(const Input& input, size_t n){
        // Initialize cache for enough elements
        data_cache_helper_t::init(n, &input, input_cache);
    }

    /*!
     * \brief Construct an inmemory data generator
     */
    inmemory_single_data_generator(Iterator first, Iterator last){
        const size_t n = std::distance(first, last);

        data_cache_helper_t::init(n, first, input_cache);

        // Fill the cache

        size_t i = 0;
        while (first != last) {
            input_cache(i) = *first;

            ++i;
            ++first;
        }

        // Transform if necessary

        pre_scaler<desc>::transform_all(input_cache);
        pre_normalizer<desc>::transform_all(input_cache);
        pre_binarizer<desc>::transform_all(input_cache);
    }

    inmemory_single_data_generator(const inmemory_single_data_generator& rhs) = delete;
    inmemory_single_data_generator operator=(const inmemory_single_data_generator& rhs) = delete;

    inmemory_single_data_generator(inmemory_single_data_generator&& rhs) = delete;
    inmemory_single_data_generator operator=(inmemory_single_data_generator&& rhs) = delete;

    /*!
     * \brief Display a description of the generator in the given stream
     * \param stream The stream to print to
     * \return stream
     */
    std::ostream& display(std::ostream& stream) const {
        stream << "In-Memory Data Generator" << std::endl;
        stream << "              Size: " << size() << std::endl;
        stream << "           Batches: " << batches() << std::endl;

        if (augmented_size() != size()) {
            stream << "    Augmented Size: " << augmented_size() << std::endl;
        }

        return stream;
    }

    /*!
     * \brief Display a description of the generator in the standard output.
     */
    void display() const {
        display(std::cout);
    }

    /*!
     * \brief Indicates that it is safe to destroy the memory of the generator
     * when not used by the pretraining phase
     */
    void set_safe() {
        is_safe = true;
    }

    /*!
     * \brier Clear the memory of the generator.
     *
     * This is only done if the generator is marked as safe it is safe.
     */
    void clear() {
        if (is_safe) {
            input_cache.clear();
        }
    }

    /*!
     * brief Sets the generator in test mode
     */
    void set_test() {
        // Nothing to do
    }

    /*!
     * brief Sets the generator in train mode
     */
    void set_train() {
        // Nothing to do
    }

    /*!
     * \brief Reset the generator to the beginning
     */
    void reset() {
        current = 0;
    }

    /*!
     * \brief Reset the generator and shuffle the order of samples
     */
    void reset_shuffle() {
        current = 0;
        shuffle();
    }

    /*!
     * \brief Shuffle the order of the samples.
     *
     * This should only be done when the generator is at the beginning.
     */
    void shuffle() {
        cpp_assert(!current, "Shuffle should only be performed on start of generation");

        etl::shuffle(input_cache, dll::rand_engine());
    }

    /*!
     * \brief Prepare the dataset for an epoch
     */
    void prepare_epoch(){
        input_cache.ensure_gpu_up_to_date();
    }

    /*!
     * \brief Return the index of the current batch in the generation
     * \return The current batch index
     */
    size_t current_batch() const {
        return current / batch_size;
    }

    /*!
     * \brief Returns the number of elements in the generator
     * \return The number of elements in the generator
     */
    size_t size() const {
        return etl::dim<0>(input_cache);
    }

    /*!
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

    /*!
     * \brief Returns the number of batches in the generator.
     * \return The number of batches in the generator
     */
    size_t batches() const {
        return size() / batch_size + (size() % batch_size == 0 ? 0 : 1);
    }

    /*!
     * \brief Indicates if the generator has a next batch or not
     * \return true if the generator has a next batch, false otherwise
     */
    bool has_next_batch() const {
        return current < size();
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
        return data_batch();
    }

    /*!
     * \brief Set some part of the data to a new set of value
     * \param i The beginning at which to start storing the new data
     * \param input_batch An input batch
     */
    template <typename Input>
    void set_data_batch(size_t i, Input&& input_batch) {
        etl::slice(input_cache, i, i + etl::dim<0>(input_batch)) = input_batch;
    }

    /*!
     * \brief Finalize the dataset if it was filled directly after having being prepared.
     */
    void finalize_prepared_data() {
        pre_scaler<desc>::transform_all(input_cache);
        pre_normalizer<desc>::transform_all(input_cache);
        pre_binarizer<desc>::transform_all(input_cache);
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
 * \copydoc inmemory_single_data_generator
 */
template <typename Iterator, typename Desc>
struct inmemory_single_data_generator<Iterator, Desc, std::enable_if_t<is_augmented<Desc>>> {
    using desc                 = Desc;                                        ///< The generator descriptor
    using weight               = etl::value_t<typename Iterator::value_type>; ///< The data type
    using data_cache_helper_t  = cache_helper<desc, Iterator>;                ///< The helper for the data cache

    using data_cache_type  = typename data_cache_helper_t::cache_type;     ///< The type of the data cache
    using big_cache_type   = typename data_cache_helper_t::big_cache_type; ///< The type of big data cache

    static constexpr bool dll_generator    = true;               ///< Simple flag to indicate that the class is a DLL generator

    static constexpr size_t inline batch_size     = desc::BatchSize;    ///< The size of the generated batches
    static constexpr size_t inline big_batch_size = desc::BigBatchSize; ///< The number of batches kept in cache

    data_cache_type input_cache;  ///< The data cache
    big_cache_type batch_cache;   ///< The data batch cache

    random_cropper<Desc> cropper;      ///< The random cropper
    random_mirrorer<Desc> mirrorer;    ///< The random mirrorer
    elastic_distorter<Desc> distorter; ///< The elastic distorter
    random_noise<Desc> noiser;         ///< The random noiser


    size_t current = 0;     ///< The current index
    bool is_safe   = false; ///< Indicates if the generator is safe to reclaim memory from

    mutable volatile bool status[big_batch_size];    ///< Status of each batch
    mutable volatile size_t indices[big_batch_size]; ///< Indices of each batch

    mutable std::mutex main_lock;                    ///< The main lock
    mutable std::condition_variable condition;       ///< The condition variable for the thread to wait for some space
    mutable std::condition_variable ready_condition; ///< The condition variable for a reader to wait for ready data

    volatile bool stop_flag = false; ///< Boolean flag indicating to the thread to stop

    std::thread main_thread; ///< The main thread
    bool train_mode = false; ///< The train mode status

    /*!
     * \brief Construct an inmemory data generator
     */
    inmemory_single_data_generator(Iterator first, Iterator last)
            : cropper(*first), mirrorer(*first), distorter(*first), noiser(*first) {
        const size_t n = std::distance(first, last);

        data_cache_helper_t::init(n, first, input_cache);
        data_cache_helper_t::init_big(first, batch_cache);

        // Fill the cache

        size_t i = 0;
        while (first != last) {
            input_cache(i) = *first;

            ++i;
            ++first;
        }

        // Transform if necessary

        pre_scaler<desc>::transform_all(input_cache);
        pre_normalizer<desc>::transform_all(input_cache);
        pre_binarizer<desc>::transform_all(input_cache);

        for (size_t b = 0; b < big_batch_size; ++b) {
            status[b]  = false;
            indices[b] = b;
        }

        main_thread = std::thread([this] {
            while (true) {
                // The index of the batch inside the batch cache
                size_t index = 0;

                {
                    std::unique_lock<std::mutex> ulock(main_lock);

                    bool found = false;

                    // Try to find a read batch first
                    for (size_t b = 0; b < big_batch_size; ++b) {
                        if (!status[b] && indices[b] * batch_size < size()) {
                            index = b;
                            found = true;
                            break;
                        }
                    }

                    // if not found, wait for a ready to compute batch

                    if (!found) {
                        // Wait for the end or for some work
                        condition.wait(ulock, [this, &index] {
                            if (stop_flag) {
                                return true;
                            }

                            for (size_t b = 0; b < big_batch_size; ++b) {
                                if (!status[b] && indices[b] * batch_size < size()) {
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

                for (size_t i = 0; i < batch_size && input_n + i < size(); ++i) {
                    if (train_mode) {
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

    inmemory_single_data_generator(const inmemory_single_data_generator& rhs) = delete;
    inmemory_single_data_generator operator=(const inmemory_single_data_generator& rhs) = delete;

    inmemory_single_data_generator(inmemory_single_data_generator&& rhs) = delete;
    inmemory_single_data_generator operator=(inmemory_single_data_generator&& rhs) = delete;

    /*!
     * \brief Display a description of the generator in the given stream
     * \param stream The stream to print to
     * \return stream
     */
    std::ostream& display(std::ostream& stream) const {
        stream << "In-Memory Data Generator" << std::endl;
        stream << "              Size: " << size() << std::endl;
        stream << "           Batches: " << batches() << std::endl;

        if (augmented_size() != size()) {
            stream << "    Augmented Size: " << augmented_size() << std::endl;
        }

        return stream;
    }

    /*!
     * \brief Display a description of the generator in the standard output.
     */
    void display() const {
        display(std::cout);
    }

    /*!
     * \brief Indicates that it is safe to destroy the memory of the generator
     * when not used by the pretraining phase
     */
    void set_safe() {
        is_safe = true;
    }

    /*!
     * \brier Clear the memory of the generator.
     *
     * This is only done if the generator is marked as safe it is safe.
     */
    void clear() {
        if (is_safe) {
            input_cache.clear();
            batch_cache.clear();
        }
    }

    /*!
     * brief Sets the generator in test mode
     */
    void set_test() {
        train_mode = false;
    }

    /*!
     * brief Sets the generator in train mode
     */
    void set_train() {
        train_mode = true;
    }

    /*!
     * \brief Destructs the inmemory_single_data_generator
     */
    ~inmemory_single_data_generator() {
        cpp::with_lock(main_lock, [this] { stop_flag = true; });

        condition.notify_all();

        main_thread.join();
    }

    /*!
     * \brief Reset the generation to its beginning
     */
    void reset_generation() {
        std::unique_lock<std::mutex> ulock(main_lock);

        for (size_t b = 0; b < big_batch_size; ++b) {
            status[b]  = false;
            indices[b] = b;
        }

        condition.notify_one();
    }

    /*!
     * \brief Reset the generator to the beginning
     */
    void reset() {
        current = 0;
        reset_generation();
    }

    /*!
     * \brief Reset the generator and shuffle the order of samples
     */
    void reset_shuffle() {
        current = 0;
        shuffle();
        reset_generation();
    }

    /*!
     * \brief Shuffle the order of the samples.
     *
     * This should only be done when the generator is at the beginning.
     */
    void shuffle() {
        cpp_assert(!current, "Shuffle should only be performed on start of generation");

        etl::shuffle(input_cache, dll::rand_engine());
    }

    /*!
     * \brief Prepare the dataset for an epoch
     */
    void prepare_epoch(){
        // Nothing can be done here
    }

    /*!
     * \brief Return the index of the current batch in the generation
     * \return The current batch index
     */
    size_t current_batch() const {
        return current / batch_size;
    }

    /*!
     * \brief Returns the number of elements in the generator
     * \return The number of elements in the generator
     */
    size_t size() const {
        return etl::dim<0>(input_cache);
    }

    /*!
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

    /*!
     * \brief Returns the number of batches in the generator.
     * \return The number of batches in the generator
     */
    size_t batches() const {
        return size() / batch_size + (size() % batch_size == 0 ? 0 : 1);
    }

    /*!
     * \brief Indicates if the generator has a next batch or not
     * \return true if the generator has a next batch, false otherwise
     */
    bool has_next_batch() const {
        return current < size();
    }

    /*!
     * \brief Moves to the next batch.
     *
     * This should only be called if the generator has a next batch.
     */
    void next_batch() {
        // Get information from batch that has been consumed
        const auto batch = current / batch_size;
        const auto b     = batch % big_batch_size;

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
        const auto b     = batch % big_batch_size;

        if (status[b]) {
            const auto input_n = indices[b] * batch_size + batch_size;

            if (input_n > size()) {
                return etl::slice(batch_cache(b), 0, batch_size - (input_n - size()));
            } else {
                return etl::slice(batch_cache(b), 0, batch_size);
            }
        }

        ready_condition.wait(ulock, [this, b] {
            return status[b];
        });

        const auto input_n = indices[b] * batch_size + batch_size;

        if (input_n > size()) {
            return etl::slice(batch_cache(b), 0, batch_size - (input_n - size()));
        } else {
            return etl::slice(batch_cache(b), 0, batch_size);
        }
    }

    /*!
     * \brief Returns the current label batch
     * \return a a batch of label.
     */
    auto label_batch() const {
        return data_batch();
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
 * \brief Display the given generator on the given stream
 * \param os The output stream
 * \param generator The generator to display
 * \return os
 */
template <typename Iterator, typename Desc>
std::ostream& operator<<(std::ostream& os, inmemory_single_data_generator<Iterator, Desc>& generator) {
    return generator.display(os);
}

/*!
 * \brief Descriptor for a inmemory_single_data_generator
 */
template <typename... Parameters>
struct inmemory_single_data_generator_desc {
    /*!
     * A list of all the parameters of the descriptor
     */
    using parameters = cpp::type_list<Parameters...>;

    /*!
     * \brief The size of a batch
     */
    static constexpr size_t BatchSize = detail::get_value_v<batch_size<1>, Parameters...>;

    /*!
     * \brief The number of batch in cache
     */
    static constexpr size_t BigBatchSize = detail::get_value_v<big_batch_size<1>, Parameters...>;

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
    static constexpr size_t random_crop_x = detail::get_value_1_v<random_crop<0, 0>, Parameters...>;

    /*!
     * \brief The random cropping Y
     */
    static constexpr size_t random_crop_y = detail::get_value_2_v<random_crop<0, 0>, Parameters...>;

    /*!
     * \brief The elastic distortion kernel
     */
    static constexpr size_t ElasticDistortion = detail::get_value_v<elastic_distortion<0>, Parameters...>;

    /*!
     * \brief The noise
     */
    static constexpr size_t Noise = detail::get_value_v<noise<0>, Parameters...>;

    /*!
     * \brief The scaling
     */
    static constexpr size_t ScalePre = detail::get_value_v<scale_pre<0>, Parameters...>;

    /*!
     * \brief The scaling
     */
    static constexpr size_t BinarizePre = detail::get_value_v<binarize_pre<0>, Parameters...>;

    /*!
     * \brief Indicates if input are normalized
     */
    static constexpr bool NormalizePre = parameters::template contains<normalize_pre>();

    /*!
     * \brief Indicates if this is an auto-encoder task
     */
    static constexpr bool AutoEncoder = parameters::template contains<autoencoder>();

    static_assert(BatchSize > 0, "The batch size must be larger than one");
    static_assert(BigBatchSize > 0, "The big batch size must be larger than one");
    static_assert(!(AutoEncoder && (random_crop_x || random_crop_y)), "autoencoder mode is not compatible with random crop");

    //Make sure only valid types are passed to the configuration list
    static_assert(
        detail::is_valid_v<
            cpp::type_list<
                batch_size_id, big_batch_size_id, horizontal_mirroring_id, vertical_mirroring_id, random_crop_id, elastic_distortion_id,
                noise_id, nop_id, normalize_pre_id, binarize_pre_id, scale_pre_id, autoencoder_id>,
            Parameters...>,
        "Invalid parameters type for rbm_desc");

    /*!
     * The generator type
     */
    template <typename Iterator>
    using generator_t = inmemory_single_data_generator<Iterator, inmemory_single_data_generator_desc<Parameters...>>;
};

/*!
 * \brief Make an out of memory data generator from iterators
 */
template <typename Iterator, typename... Parameters>
auto make_generator(Iterator first, Iterator last, const inmemory_single_data_generator_desc<Parameters...>& /*desc*/) {
    using generator_t = typename inmemory_single_data_generator_desc<Parameters...>::template generator_t<Iterator>;
    return std::make_unique<generator_t>(first, last);
}

/*!
 * \brief Make an out of memory data generator from containers
 */
template <typename Container, typename... Parameters>
auto make_generator(const Container& container, const inmemory_single_data_generator_desc<Parameters...>& /*desc*/) {
    using generator_t = typename inmemory_single_data_generator_desc<Parameters...>::template generator_t<typename Container::const_iterator>;
    return std::make_unique<generator_t>(container.begin(), container.end());
}

// The following are simply helpers for creating generic generators

/*!
 * \brief Make an out of memory data generator from iterators
 */
template <typename Iterator, typename... Parameters>
auto make_generator(Iterator first, Iterator last, [[maybe_unused]] size_t n,
                    const inmemory_single_data_generator_desc<Parameters...>& /*desc*/) {
    using generator_t = typename inmemory_single_data_generator_desc<Parameters...>::template generator_t<Iterator>;
    return std::make_unique<generator_t>(first, last);
}

/*!
 * \brief Make an out of memory data generator from containers
 */
template <typename Container, typename... Parameters>
auto make_generator(const Container& container, [[maybe_unused]] size_t n,
                    const inmemory_single_data_generator_desc<Parameters...>& /*desc*/) {
    using generator_t = typename inmemory_single_data_generator_desc<Parameters...>::template generator_t<typename Container::const_iterator>;
    return std::make_unique<generator_t>(container.begin(), container.end());
}

/*!
 * \brief Prepare an in-memory data generator from an example. The generator
 * will be constructed to hold the given size and can then be filled.
 */
template <typename Input, typename... Parameters>
auto prepare_generator(const Input& input, size_t n, const inmemory_single_data_generator_desc<Parameters...>& /*desc*/) {
    // Create fake iterators for the type (won't be iterated
    using Iterator  = const Input*;

    // The generator type
    using generator_t = typename inmemory_single_data_generator_desc<Parameters...>::template generator_t<Iterator>;

    return std::make_unique<generator_t>(input, n);
}

} //end of dll namespace
