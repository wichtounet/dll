//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Implementation of an out-of-memory data generator
 */

#pragma once

#include <atomic>
#include <thread>

namespace dll {

/*!
 * \brief a out-of-memory data generator
 */
template <typename Iterator, typename LIterator, typename Desc, typename Enable = void>
struct outmemory_data_generator;

/*!
 * \copydoc outmemory_data_generator
 */
template <typename Iterator, typename LIterator, typename Desc>
struct outmemory_data_generator<Iterator, LIterator, Desc, std::enable_if_t<!is_augmented<Desc> && !is_threaded<Desc>>> {
    using desc                 = Desc;                                        ///< The generator descriptor
    using weight               = etl::value_t<typename Iterator::value_type>; ///< The data type
    using data_cache_helper_t  = cache_helper<Desc, Iterator>;                ///< The helper for the data cache
    using label_cache_helper_t = label_cache_helper<Desc, weight, LIterator>; ///< The helper for the label cache

    using big_data_cache_type  = typename data_cache_helper_t::big_cache_type;  ///< The type of the big data cache
    using big_label_cache_type = typename label_cache_helper_t::big_cache_type; ///< The type of the big label cache

    static constexpr bool dll_generator    = true;               ///< Simple flag to indicate that the class is a DLL generator
    static inline constexpr size_t batch_size     = desc::BatchSize;    ///< The size of the batch
    static inline constexpr size_t big_batch_size = desc::BigBatchSize; ///< The number of batches kept in cache

    big_data_cache_type batch_cache;  ///< The data batch cache
    big_label_cache_type label_cache; ///< The label batch cache

    size_t current      = 0;     ///< The current index
    size_t current_real = 0;     ///< The current real index
    size_t current_b    = 0;     ///< The current batch
    bool is_safe        = false; ///< Indicates if the generator is safe to reclaim memory from

    const size_t _size; ///< The size of the dataset
    Iterator orig_it;   ///< The original first iterator on data
    LIterator orig_lit; ///< The original first iterator on label
    Iterator it;        ///< The current iterator on data
    LIterator lit;      ///< The current iterator on label


    /*!
     * \brief Construct an outmemory_data_generator
     * \param first The iterator on the beginning on data
     * \param last The iterator on the end  on data
     * \param lfirst The iterator on the beginning on labels
     * \param llast The iterator on the end  on labels
     * \param n_classes The number of classes
     * \param size The size of the entire dataset
     */
    outmemory_data_generator(Iterator first, [[maybe_unused]] Iterator last, LIterator lfirst, [[maybe_unused]] LIterator llast, size_t n_classes, size_t size)
            : _size(size), orig_it(first), orig_lit(lfirst), it(orig_it), lit(orig_lit) {
        data_cache_helper_t::init_big(first, batch_cache);
        label_cache_helper_t::init_big(n_classes, lfirst, label_cache);

        reset();
    }

    outmemory_data_generator(const outmemory_data_generator& rhs) = delete;
    outmemory_data_generator operator=(const outmemory_data_generator& rhs) = delete;

    outmemory_data_generator(outmemory_data_generator&& rhs) = delete;
    outmemory_data_generator operator=(outmemory_data_generator&& rhs) = delete;

    /*!
     * \brief Display a description of the generator in the given stream
     * \param stream The stream to print to
     * \return stream
     */
    std::ostream& display(std::ostream& stream) const {
        stream << "Out-Of-Memory Data Generator" << std::endl;
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
            batch_cache.clear();
            label_cache.clear();
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
     * \brief Fetch the next batch
     */
    void fetch_next() {
        current_b = 0;

        for (size_t b = 0; b < big_batch_size && current_real < _size; ++b) {
            for (size_t i = 0; i < batch_size && current_real < _size;) {
                auto sub = batch_cache(b)(i);

                sub = *it;

                pre_scaler<desc>::transform(sub);
                pre_normalizer<desc>::transform(sub);
                pre_binarizer<desc>::transform(sub);

                label_cache_helper_t::set(i, lit, label_cache(b));

                // In case of auto-encoders, the label images also need to be transformed
                if constexpr (desc::AutoEncoder) {
                    pre_scaler<desc>::transform(label_cache(b)(i));
                    pre_normalizer<desc>::transform(label_cache(b)(i));
                    pre_binarizer<desc>::transform(label_cache(b)(i));
                }

                ++i;
                ++current_real;
                ++it;
                ++lit;
            }
        }
    }

    /*!
     * \brief Reset the generator to the beginning
     */
    void reset() {
        current      = 0;
        current_real = 0;

        it  = orig_it;
        lit = orig_lit;

        fetch_next();
    }

    /*!
     * \brief Reset the generator and shuffle the order of samples
     */
    void reset_shuffle() {
        cpp_unreachable("Impossible to shuffle out-of-memory data set");
    }

    /*!
     * \brief Shuffle the order of the samples.
     *
     * This should only be done when the generator is at the beginning.
     */
    void shuffle() {
        cpp_unreachable("Impossible to shuffle out-of-memory data set");
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
        return _size;
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
        return _size;
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
        ++current_b;

        if (current_b == big_batch_size) {
            fetch_next();
        }

        current += batch_size;
    }

    /*!
     * \brief Returns the current data batch
     * \return a a batch of data.
     */
    auto data_batch() const {
        return etl::slice(batch_cache(current_b), 0, std::min(batch_size, current_real - current));
    }

    /*!
     * \brief Returns the current label batch
     * \return a a batch of label.
     */
    auto label_batch() const {
        return etl::slice(label_cache(current_b), 0, std::min(batch_size, current_real - current));
    }

    /*!
     * \brief Returns the number of dimensions of the input.
     * \return The number of dimensions of the input.
     */
    static constexpr size_t dimensions() {
        return etl::dimensions<big_data_cache_type>() - 2;
    }
};

/*!
 * \copydoc outmemory_data_generator
 */
template <typename Iterator, typename LIterator, typename Desc>
struct outmemory_data_generator<Iterator, LIterator, Desc, std::enable_if_t<is_augmented<Desc> || is_threaded<Desc>>> {
    using desc                 = Desc;                                        ///< The generator descriptor
    using weight               = etl::value_t<typename Iterator::value_type>; ///< The data type
    using data_cache_helper_t  = cache_helper<desc, Iterator>;                ///< The helper for the data cache
    using label_cache_helper_t = label_cache_helper<desc, weight, LIterator>; ///< The helper for the label cache

    using big_data_cache_type  = typename data_cache_helper_t::big_cache_type;  ///< The type of the big data cache
    using big_label_cache_type = typename label_cache_helper_t::big_cache_type; ///< The type of the big label cache

    static constexpr bool dll_generator    = true;               ///< Simple flag to indicate that the class is a DLL generator
    static inline constexpr size_t batch_size     = desc::BatchSize;    ///< The size of the generated batches
    static inline constexpr size_t big_batch_size = desc::BigBatchSize; ///< The number of batches kept in cache

    big_data_cache_type batch_cache;  ///< The data batch cache
    big_label_cache_type label_cache; ///< The label batch cache

    size_t current      = 0;     ///< The current index
    size_t current_read = 0;     ///< The current index read
    bool is_safe        = false; ///< Indicates if the generator is safe to reclaim memory from

    mutable volatile bool status[big_batch_size];    ///< Status of each batch
    mutable volatile size_t indices[big_batch_size]; ///< Indices of each batch

    mutable std::mutex main_lock;                    ///< The main lock
    mutable std::condition_variable condition;       ///< The condition variable for the thread to wait for some space
    mutable std::condition_variable ready_condition; ///< The condition variable for a reader to wait for ready data

    volatile bool stop_flag = false; ///< Boolean flag indicating to the thread to stop

    std::thread main_thread; ///< The main thread
    bool train_mode = false; ///< The train mode status

    const size_t _size; ///< The size of the dataset
    Iterator orig_it;   ///< The original first iterator on data
    LIterator orig_lit; ///< The original first iterator on label
    Iterator it;        ///< The current iterator on data
    LIterator lit;      ///< The current iterator on label

    random_cropper<Desc> cropper;      ///< The random cropper
    random_mirrorer<Desc> mirrorer;    ///< The random mirrorer
    elastic_distorter<Desc> distorter; ///< The elastic distorter
    random_noise<Desc> noiser;         ///< The random noiser

    /*!
     * \brief Construct an outmemory_data_generator
     * \param first The iterator on the beginning on data
     * \param last The iterator on the end  on data
     * \param lfirst The iterator on the beginning on labels
     * \param llast The iterator on the end  on labels
     * \param n_classes The number of classes
     * \param size The size of the entire dataset
     */
    outmemory_data_generator(Iterator first, [[maybe_unused]] Iterator last, LIterator lfirst, [[maybe_unused]] LIterator llast, size_t n_classes, size_t size)
            : _size(size), orig_it(first), orig_lit(lfirst), it(orig_it), lit(orig_lit), cropper(*first), mirrorer(*first), distorter(*first), noiser(*first) {
        data_cache_helper_t::init_big(first, batch_cache);
        label_cache_helper_t::init_big(n_classes, lfirst, label_cache);

        main_thread = std::thread([this] {
            while (true) {
                // The index of the batch inside the batch cache
                size_t index = 0;

                {
                    std::unique_lock<std::mutex> ulock(main_lock);

                    bool found = false;

                    // Try to find a read batch first
                    for (size_t b = 0; b < big_batch_size; ++b) {
                        if (!status[b] && indices[b] * batch_size < _size) {
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
                                if (!status[b] && indices[b] * batch_size < _size) {
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

                SERIAL_SECTION {
                    for (size_t i = 0; i < batch_size && current_read < _size; ++i) {
                        auto sub = batch_cache(index)(i);

                        if (train_mode) {
                            // Random crop the image
                            cropper.transform_first(sub, *it);

                            pre_scaler<desc>::transform(sub);
                            pre_normalizer<desc>::transform(sub);
                            pre_binarizer<desc>::transform(sub);

                            // Mirror the image
                            mirrorer.transform(sub);

                            // Distort the image
                            distorter.transform(sub);

                            // Noise the image
                            noiser.transform(sub);
                        } else {
                            // Center crop the image
                            cropper.transform_first_test(sub, *it);

                            pre_scaler<desc>::transform(sub);
                            pre_normalizer<desc>::transform(sub);
                            pre_binarizer<desc>::transform(sub);
                        }

                        label_cache_helper_t::set(i, lit, label_cache(index));

                        // In case of auto-encoders, the label images also need to be transformed
                        if constexpr (desc::AutoEncoder){
                            pre_scaler<desc>::transform(label_cache(index)(i));
                            pre_normalizer<desc>::transform(label_cache(index)(i));
                            pre_binarizer<desc>::transform(label_cache(index)(i));
                        }

                        ++it;
                        ++lit;
                        ++current_read;
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

    outmemory_data_generator(const outmemory_data_generator& rhs) = delete;
    outmemory_data_generator& operator=(const outmemory_data_generator& rhs) = delete;

    outmemory_data_generator(outmemory_data_generator&& rhs) = delete;
    outmemory_data_generator& operator=(outmemory_data_generator&& rhs) = delete;

    /*!
     * \brief Destructs the outmemory_data_generator
     */
    ~outmemory_data_generator() {
        cpp::with_lock(main_lock, [this] { stop_flag = true; });

        condition.notify_all();

        main_thread.join();
    }

    /*!
     * \brief Display a description of the generator in the given stream
     * \param stream The stream to print to
     * \return stream
     */
    std::ostream& display(std::ostream& stream) const {
        stream << "Out-Of-Memory Data Generator" << std::endl;
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
            batch_cache.clear();
            label_cache.clear();
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
     * \brief Reset the generation
     */
    void reset_generation() {
        std::unique_lock<std::mutex> ulock(main_lock);

        current_read = 0;
        it           = orig_it;
        lit          = orig_lit;

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
        cpp_unreachable("Out-of-memory generator cannot be shuffled");
    }

    /*!
     * \brief Shuffle the order of the samples.
     *
     * This should only be done when the generator is at the beginning.
     */
    void shuffle() {
        cpp_unreachable("Out-of-memory generator cannot be shuffled");
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
        return _size;
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
        return cropper.scaling() * mirrorer.scaling() * noiser.scaling() * distorter.scaling() * size();
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
            return etl::slice(batch_cache(b), 0, std::min(batch_size, _size - current));
        }

        ready_condition.wait(ulock, [this, b] {
            return status[b];
        });

        return etl::slice(batch_cache(b), 0, std::min(batch_size, _size - current));
    }

    /*!
     * \brief Returns the current label batch
     * \return a a batch of label.
     */
    auto label_batch() const {
        std::unique_lock<std::mutex> ulock(main_lock);

        const auto batch = current / batch_size;
        const auto b     = batch % big_batch_size;

        if (status[b]) {
            return etl::slice(label_cache(b), 0, std::min(batch_size, _size - current));
        }

        ready_condition.wait(ulock, [this, b] {
            return status[b];
        });

        return etl::slice(label_cache(b), 0, std::min(batch_size, _size - current));
    }

    /*!
     * \brief Returns the number of dimensions of the input.
     * \return The number of dimensions of the input.
     */
    static constexpr size_t dimensions() {
        return etl::dimensions<big_data_cache_type>() - 2;
    }
};

/*!
 * \brief Display the given generator on the given stream
 * \param os The output stream
 * \param generator The generator to display
 * \return os
 */
template <typename Iterator, typename LIterator, typename Desc>
std::ostream& operator<<(std::ostream& os, outmemory_data_generator<Iterator, LIterator, Desc>& generator) {
    return generator.display(os);
}

/*!
 * \brief Descriptor for a outmemory_data_generator
 */
template <typename... Parameters>
struct outmemory_data_generator_desc {
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
     * \brief Indicates if the generator is threaded
     */
    static constexpr bool Threaded = parameters::template contains<threaded>();

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
     * \brief The binarization threshold
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
                batch_size_id, big_batch_size_id, horizontal_mirroring_id, vertical_mirroring_id, random_crop_id,
                elastic_distortion_id, categorical_id, noise_id, threaded_id, nop_id, normalize_pre_id, binarize_pre_id, scale_pre_id, autoencoder_id>,
            Parameters...>,
        "Invalid parameters type for rbm_desc");

    /*!
     * The generator type
     */
    template <typename Iterator, typename LIterator>
    using generator_t = outmemory_data_generator<Iterator, LIterator, outmemory_data_generator_desc<Parameters...>>;
};

/*!
 * \brief Make an out of memory data generator from iterators
 */
template <typename Iterator, typename LIterator, typename... Parameters>
auto make_generator(Iterator first, Iterator last, LIterator lfirst, LIterator llast, size_t size, size_t n_classes, const outmemory_data_generator_desc<Parameters...>& /*desc*/) {
    using generator_t = typename outmemory_data_generator_desc<Parameters...>::template generator_t<Iterator, LIterator>;
    return std::make_unique<generator_t>(first, last, lfirst, llast, n_classes, size);
}

/*!
 * \brief Make an out of memory data generator from containers
 */
template <typename Container, typename LContainer, typename... Parameters>
auto make_generator(const Container& container, const LContainer& lcontainer, size_t size, size_t n_classes, const outmemory_data_generator_desc<Parameters...>& /*desc*/) {
    using generator_t = typename outmemory_data_generator_desc<Parameters...>::template generator_t<typename Container::const_iterator, typename LContainer::const_iterator>;
    return std::make_unique<generator_t>(container.begin(), container.end(), lcontainer.begin(), lcontainer.end(), n_classes, size);
}

} //end of dll namespace
