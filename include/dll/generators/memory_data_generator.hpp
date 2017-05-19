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

template<typename Desc>
struct is_augmented {
    static constexpr bool value = (Desc::random_crop_x > 0 && Desc::random_crop_y > 0) || Desc::HorizontalMirroring || Desc::VerticalMirroring || Desc::Noise || Desc::ElasticDistortion;
};

template<typename Iterator, typename LIterator, typename Desc, typename Enable = void>
struct memory_data_generator;

template<typename Iterator, typename LIterator, typename Desc>
struct memory_data_generator <Iterator, LIterator, Desc, std::enable_if_t<!is_augmented<Desc>::value>> {
    using desc = Desc;
    using weight = typename desc::weight;
    using data_cache_helper_t = cache_helper<Desc, Iterator>;
    using label_cache_helper_t = label_cache_helper<Desc, LIterator>;

    using data_cache_type = typename data_cache_helper_t::cache_type;
    using label_cache_type = typename label_cache_helper_t::cache_type;

    static constexpr bool dll_generator = true;

    static constexpr size_t batch_size = desc::BatchSize;

    data_cache_type input_cache;
    label_cache_type label_cache;

    size_t current = 0;

    memory_data_generator(Iterator first, Iterator last, LIterator lfirst, LIterator llast, size_t n_classes){
        const size_t n = std::distance(first, last);

        data_cache_helper_t::init(n, first, input_cache);
        label_cache_helper_t::init(n, n_classes, lfirst, label_cache);

        size_t i = 0;
        while(first != last){
            input_cache(i) = *first;

            label_cache_helper_t::set(i, lfirst, label_cache);

            ++i;
            ++first;
            ++lfirst;
        }

        cpp_unused(llast);
    }

    void reset(){
        current = 0;
    }

    void reset_shuffle(){
        current = 0;
        shuffle();
    }

    void shuffle(){
        cpp_assert(!current, "Shuffle should only be performed on start of generation");

        etl::parallel_shuffle(input_cache, label_cache);
    }

    size_t current_batch() const {
        return current / batch_size;
    }

    size_t size() const {
        return etl::dim<0>(input_cache);
    }

    size_t augmented_size() const {
        return etl::dim<0>(input_cache);
    }

    size_t batches() const {
        return size() / batch_size + (size() % batch_size == 0 ? 0 : 1);
    }

    bool has_next_batch() const {
        return current < size() - 1;
    }

    void next_batch() {
        current += batch_size;
    }

    auto data_batch() const {
        return etl::slice(input_cache, current, std::min(current + batch_size, size()));
    }

    auto label_batch() const {
        return etl::slice(label_cache, current, std::min(current + batch_size, size()));
    }

    static constexpr size_t dimensions() {
        return etl::dimensions<data_cache_type>() - 1;
    }
};

template<typename Iterator, typename LIterator, typename Desc>
struct memory_data_generator <Iterator, LIterator, Desc, std::enable_if_t<is_augmented<Desc>::value>> {
    using desc = Desc;
    using weight = typename desc::weight;
    using data_cache_helper_t = cache_helper<desc, Iterator>;
    using label_cache_helper_t = label_cache_helper<desc, LIterator>;

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
    bool threaded = false;

    memory_data_generator(Iterator first, Iterator last, LIterator lfirst, LIterator llast, size_t n_classes) : cropper(*first), mirrorer(*first), distorter(*first), noiser(*first) {
        const size_t n = std::distance(first, last);

        data_cache_helper_t::init(n, first, input_cache);
        data_cache_helper_t::init_big(big_batch_size, batch_size, first, batch_cache);

        label_cache_helper_t::init(n, n_classes, lfirst, label_cache);

        size_t i = 0;
        while(first != last){
            input_cache(i) = *first;

            label_cache_helper_t::set(i, lfirst, label_cache);

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

                for(size_t i = 0; i < batch_size; ++i){
                    // Random crop the image
                    cropper.transform_first(batch_cache(index)(i), input_cache(input_n + i));

                    //// Mirror the image
                    mirrorer.transform(batch_cache(index)(i));

                    // Distort the image
                    distorter.transform(batch_cache(index)(i));

                    // Noise the image
                    noiser.transform(batch_cache(index)(i));
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

    memory_data_generator(const memory_data_generator& rhs) = delete;
    memory_data_generator& operator=(const memory_data_generator& rhs) = delete;

    memory_data_generator(memory_data_generator&& rhs) = delete;
    memory_data_generator& operator=(memory_data_generator&& rhs) = delete;

    ~memory_data_generator(){
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

    void reset(){
        current = 0;
        reset_generation();
    }

    void reset_shuffle(){
        current = 0;
        shuffle();
        reset_generation();
    }

    void shuffle(){
        cpp_assert(!current, "Shuffle should only be performed on start of generation");

        etl::parallel_shuffle(input_cache, label_cache);
    }

    size_t current_batch() const {
        return current / batch_size;
    }

    size_t size() const {
        return etl::dim<0>(input_cache);
    }

    size_t augmented_size() const {
        return cropper.scaling() * mirrorer.scaling() * noiser.scaling() * distorter.scaling() * etl::dim<0>(input_cache);
    }

    size_t batches() const {
        return size() / batch_size + (size() % batch_size == 0 ? 0 : 1);
    }

    bool has_next_batch() const {
        return current < size() - 1;
    }

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

    auto label_batch() const {
        return etl::slice(label_cache, current, std::min(current + batch_size, size()));
    }

    static constexpr size_t dimensions() {
        return etl::dimensions<data_cache_type>() - 1;
    }
};

/*!
 * \brief Descriptor for a memory_data_generator
 */
template <typename... Parameters>
struct memory_data_generator_desc {
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
     * The type used to store the weights
     */
    using weight = typename detail::get_type<weight_type<float>, Parameters...>::value;

    static_assert(BatchSize > 0, "The batch size must be larger than one");

    //Make sure only valid types are passed to the configuration list
    static_assert(
        detail::is_valid<cpp::type_list<batch_size_id, big_batch_size_id, horizontal_mirroring_id, vertical_mirroring_id, random_crop_id, elastic_distortion_id, categorical_id, noise_id, nop_id>,
                         Parameters...>::value,
        "Invalid parameters type for rbm_desc");

    /*!
     * The generator type
     */
    template<typename Iterator, typename LIterator>
    using generator_t = memory_data_generator<Iterator, LIterator, memory_data_generator_desc<Parameters...>>;
};

template<typename Iterator, typename LIterator, typename... Parameters>
auto make_generator(Iterator first, Iterator last, LIterator lfirst, LIterator llast, size_t n_classes, const memory_data_generator_desc<Parameters...>& /*desc*/){
    using generator_t = typename memory_data_generator_desc<Parameters...>::template generator_t<Iterator, LIterator>;
    return std::make_unique<generator_t>(first, last, lfirst, llast, n_classes);
}

template<typename Container, typename LContainer, typename... Parameters>
auto make_generator(const Container& container, const LContainer& lcontainer, size_t n_classes, const memory_data_generator_desc<Parameters...>& /*desc*/){
    using generator_t = typename memory_data_generator_desc<Parameters...>::template generator_t<typename Container::const_iterator, typename LContainer::const_iterator>;
    return std::make_unique<generator_t>(container.begin(), container.end(), lcontainer.begin(), lcontainer.end(), n_classes);
}

} //end of dll namespace
