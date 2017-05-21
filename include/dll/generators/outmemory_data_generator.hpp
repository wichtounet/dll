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
 * \brief a out-of-memory data generator
 */
template<typename Iterator, typename LIterator, typename Desc, typename Enable = void>
struct outmemory_data_generator;

/*!
 * \copydoc outmemory_data_generator
 */
template<typename Iterator, typename LIterator, typename Desc>
struct outmemory_data_generator <Iterator, LIterator, Desc, std::enable_if_t<!is_augmented<Desc>::value && !is_threaded<Desc>::value>> {
    using desc = Desc;
    using weight = typename desc::weight;
    using data_cache_helper_t = cache_helper<Desc, Iterator>;
    using label_cache_helper_t = label_cache_helper<Desc, LIterator>;

    using big_data_cache_type  = typename data_cache_helper_t::big_cache_type;
    using big_label_cache_type = typename label_cache_helper_t::big_cache_type;

    static constexpr bool dll_generator = true;

    static constexpr size_t batch_size = desc::BatchSize;
    static constexpr size_t big_batch_size = desc::BigBatchSize;

    big_data_cache_type batch_cache;
    big_label_cache_type label_cache;

    size_t current = 0;
    size_t current_real = 0;
    size_t current_b = 0;

    const size_t _size;
    Iterator orig_it;
    LIterator orig_lit;
    Iterator it;
    LIterator lit;

    outmemory_data_generator(Iterator first, Iterator last, LIterator lfirst, LIterator llast, size_t n_classes, size_t size) : _size(size), orig_it(first), orig_lit(lfirst) {
        data_cache_helper_t::init_big(big_batch_size, batch_size, first, batch_cache);
        label_cache_helper_t::init_big(big_batch_size, batch_size, n_classes, lfirst, label_cache);

        reset();

        cpp_unused(last);
        cpp_unused(llast);
    }

    void fetch_next(){
        current_b = 0;

        for(size_t b = 0; b < big_batch_size && current_real < _size; ++b){
            for(size_t i = 0; i < batch_size && current_real < _size; ){
                batch_cache(b)(i) = *it;
                label_cache_helper_t::set(i, lit, label_cache(b));

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
    void reset(){
        current = 0;
        current_real = 0;

        it = orig_it;
        lit = orig_lit;

        fetch_next();
    }

    /*
     * \brief Reset the generator and shuffle the order of samples
     */
    void reset_shuffle(){
        cpp_unreachable("Impossible to shuffle out-of-memory data set");
    }

    /*!
     * \brief Shuffle the order of the samples.
     *
     * This should only be done when the generator is at the beginning.
     */
    void shuffle(){
        cpp_unreachable("Impossible to shuffle out-of-memory data set");
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
        return _size;
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
        return _size;
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
        ++current_b;

        if(current_b == big_batch_size){
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
template<typename Iterator, typename LIterator, typename Desc>
struct outmemory_data_generator <Iterator, LIterator, Desc, std::enable_if_t<is_augmented<Desc>::value || is_threaded<Desc>::value>> {
    using desc = Desc;
    using weight = typename desc::weight;
    using data_cache_helper_t = cache_helper<desc, Iterator>;
    using label_cache_helper_t = label_cache_helper<desc, LIterator>;

    using big_data_cache_type  = typename data_cache_helper_t::big_cache_type;
    using big_label_cache_type = typename label_cache_helper_t::big_cache_type;

    static constexpr bool dll_generator = true;

    static constexpr size_t batch_size = desc::BatchSize;
    static constexpr size_t big_batch_size = desc::BigBatchSize;

    big_data_cache_type batch_cache;
    big_label_cache_type label_cache;

    size_t current = 0;
    size_t current_read = 0;

    mutable volatile bool status[big_batch_size];
    mutable volatile size_t indices[big_batch_size];

    mutable std::mutex main_lock;
    mutable std::condition_variable condition;
    mutable std::condition_variable ready_condition;

    volatile bool stop_flag = false;

    std::thread main_thread;
    bool threaded = false;

    const size_t _size;
    Iterator orig_it;
    LIterator orig_lit;
    Iterator it;
    LIterator lit;

    random_cropper<Desc> cropper;
    random_mirrorer<Desc> mirrorer;
    elastic_distorter<Desc> distorter;
    random_noise<Desc> noiser;

    outmemory_data_generator(Iterator first, Iterator last, LIterator lfirst, LIterator llast, size_t n_classes, size_t size) : _size(size), orig_it(first), orig_lit(lfirst), cropper(*first), mirrorer(*first), distorter(*first), noiser(*first) {
        data_cache_helper_t::init_big(big_batch_size, batch_size, first, batch_cache);
        label_cache_helper_t::init_big(big_batch_size, batch_size, n_classes, lfirst, label_cache);

        cpp_unused(last);
        cpp_unused(llast);

        current_read = 0;
        it = orig_it;
        lit = orig_lit;

        main_thread = std::thread([this]{
            while(true){
                // The index of the batch inside the batch cache
                size_t index = 0;

                {
                    std::unique_lock<std::mutex> ulock(main_lock);

                    bool found = false;

                    // Try to find a read batch first
                    for(size_t b = 0; b < big_batch_size; ++b){
                        if(!status[b] && indices[b] * batch_size < _size){
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
                                if(!status[b] && indices[b] * batch_size < _size){
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

                for(size_t i = 0; i < batch_size && current_read < _size; ++i){
                    // Random crop the image
                    cropper.transform_first(batch_cache(index)(i), *it);

                    //// Mirror the image
                    mirrorer.transform(batch_cache(index)(i));

                    // Distort the image
                    distorter.transform(batch_cache(index)(i));

                    // Noise the image
                    noiser.transform(batch_cache(index)(i));

                    label_cache_helper_t::set(i, lit, label_cache(index));

                    ++it;
                    ++lit;
                    ++current_read;
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

    ~outmemory_data_generator(){
        cpp::with_lock(main_lock, [this] { stop_flag = true; });

        condition.notify_all();

        main_thread.join();
    }

    void reset_generation(){
        std::unique_lock<std::mutex> ulock(main_lock);

        current_read = 0;
        it = orig_it;
        lit = orig_lit;

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
        cpp_unreachable("Out-of-memory generator cannot be shuffled");
    }

    /*!
     * \brief Shuffle the order of the samples.
     *
     * This should only be done when the generator is at the beginning.
     */
    void shuffle(){
        cpp_unreachable("Out-of-memory generator cannot be shuffled");
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
        return _size;
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
        return cropper.scaling() * mirrorer.scaling() * noiser.scaling() * distorter.scaling() * size();
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
        const auto b = batch % big_batch_size;

        if(status[b]){
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
     * \brief Indicates if the generator is threaded
     */
    static constexpr bool Threaded = parameters::template contains<threaded>();

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
     * The type used to store the weights
     */
    using weight = typename detail::get_type<weight_type<float>, Parameters...>::value;

    static_assert(BatchSize > 0, "The batch size must be larger than one");

    //Make sure only valid types are passed to the configuration list
    static_assert(
        detail::is_valid<cpp::type_list<batch_size_id, big_batch_size_id, horizontal_mirroring_id, vertical_mirroring_id, random_crop_id, elastic_distortion_id, categorical_id, noise_id, threaded_id, nop_id>,
                         Parameters...>::value,
        "Invalid parameters type for rbm_desc");

    /*!
     * The generator type
     */
    template<typename Iterator, typename LIterator>
    using generator_t = outmemory_data_generator<Iterator, LIterator, outmemory_data_generator_desc<Parameters...>>;
};

template<typename Iterator, typename LIterator, typename... Parameters>
auto make_generator(Iterator first, Iterator last, LIterator lfirst, LIterator llast, size_t size, size_t n_classes, const outmemory_data_generator_desc<Parameters...>& /*desc*/){
    using generator_t = typename outmemory_data_generator_desc<Parameters...>::template generator_t<Iterator, LIterator>;
    return std::make_unique<generator_t>(first, last, lfirst, llast, n_classes, size);
}

template<typename Container, typename LContainer, typename... Parameters>
auto make_generator(const Container& container, const LContainer& lcontainer, size_t size, size_t n_classes, const outmemory_data_generator_desc<Parameters...>& /*desc*/){
    using generator_t = typename outmemory_data_generator_desc<Parameters...>::template generator_t<typename Container::const_iterator, typename LContainer::const_iterator>;
    return std::make_unique<generator_t>(container.begin(), container.end(), lcontainer.begin(), lcontainer.end(), n_classes, size);
}

} //end of dll namespace
