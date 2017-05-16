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

template<typename Desc, typename Iterator, typename Enable = void>
struct cache_helper;

template<typename Desc, typename Iterator>
struct cache_helper<Desc, Iterator, std::enable_if_t<etl::is_1d<typename Iterator::value_type>::value>> {
    using T = typename Desc::weight;

    using cache_type = etl::dyn_matrix<T, 2>;
    using big_cache_type = etl::dyn_matrix<T, 3>;

    static void init(size_t n, Iterator& it, cache_type& cache){
        auto one = *it;
        cache = cache_type(n, etl::dim<0>(one));
    }

    static void init_big(size_t big, size_t n, Iterator& it, big_cache_type& cache){
        auto one = *it;
        cache = big_cache_type(big, n, etl::dim<0>(one));
    }
};

template<typename Desc, typename Iterator>
struct cache_helper<Desc, Iterator, std::enable_if_t<etl::is_3d<typename Iterator::value_type>::value>> {
    using T = typename Desc::weight;

    using cache_type = etl::dyn_matrix<T, 4>;
    using big_cache_type = etl::dyn_matrix<T, 5>;

    static void init(size_t n, Iterator& it, cache_type& cache){
        auto one = *it;
        cache = cache_type(n, etl::dim<0>(one), etl::dim<1>(one), etl::dim<2>(one));
    }

    static void init_big(size_t big, size_t n, Iterator& it, big_cache_type& cache){
        auto one = *it;

        if(Desc::random_crop_x && Desc::random_crop_y){
            cache = big_cache_type(big, n, etl::dim<0>(one), Desc::random_crop_y, Desc::random_crop_x);
        } else {
            cache = big_cache_type(big, n, etl::dim<0>(one), etl::dim<1>(one), etl::dim<2>(one));
        }
    }
};

template<typename Desc>
struct is_augmented {
    static constexpr bool value = Desc::random_crop_x > 0 && Desc::random_crop_y > 0;
};

template<typename Iterator, typename LIterator, typename Desc, typename Enable = void>
struct memory_data_generator;

template<typename Iterator, typename LIterator, typename Desc>
struct memory_data_generator <Iterator, LIterator, Desc, std::enable_if_t<!is_augmented<Desc>::value>> {
    using desc = Desc;
    using weight = typename desc::weight;
    using cache_helper_t = cache_helper<Desc, Iterator>;

    using cache_type = typename cache_helper_t::cache_type;
    using label_type = etl::dyn_matrix<weight, 2>;

    static constexpr size_t batch_size = desc::BatchSize;

    cache_type input_cache;
    label_type labels;

    size_t current = 0;

    memory_data_generator(Iterator first, Iterator last, LIterator lfirst, LIterator llast, size_t n_classes){
        const size_t n = std::distance(first, last);

        cache_helper_t::init(n, first, input_cache);
        labels = label_type(n, n_classes);

        labels = weight(0);

        size_t i = 0;
        while(first != last){
            input_cache(i) = *first;

            labels(i, *lfirst) = weight(1);

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

        etl::parallel_shuffle(input_cache, labels);
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
        return etl::slice(labels, current, std::min(current + batch_size, size()));
    }

    static constexpr size_t dimensions() {
        return etl::dimensions<cache_type>() - 1;
    }
};

template<typename Desc, typename Enable = void>
struct random_cropper;

template<typename Desc>
struct random_cropper <Desc, std::enable_if_t<Desc::random_crop_x && Desc::random_crop_y>> {
    using weight = typename Desc::weight;

    static constexpr size_t random_crop_x = Desc::random_crop_x;
    static constexpr size_t random_crop_y = Desc::random_crop_y;

    size_t x = 0;
    size_t y = 0;

    std::random_device rd;
    std::default_random_engine engine;

    std::uniform_int_distribution<size_t> dist_x;
    std::uniform_int_distribution<size_t> dist_y;

    template<typename T>
    random_cropper(const T& image) : engine(rd()) {
        static_assert(etl::dimensions<T>() == 3, "random_cropper can only be used with 3D images");

        y = etl::dim<1>(image);
        x = etl::dim<2>(image);

        dist_x = std::uniform_int_distribution<size_t>{0, x - random_crop_x};
        dist_y = std::uniform_int_distribution<size_t>{0, y - random_crop_y};
    }

    size_t scaling() const {
        return (x - random_crop_x) * (y - random_crop_y);
    }

    template<typename T>
    auto crop(const T& image){
        etl::dyn_matrix<weight, 3> cropped(etl::dim<0>(image), random_crop_y, random_crop_x);

        const size_t y_offset = dist_y(engine);
        const size_t x_offset = dist_x(engine);

        for (size_t c = 0; c < etl::dim<0>(image); ++c) {
            for (size_t y = 0; y < random_crop_y; ++y) {
                for (size_t x = 0; x < random_crop_x; ++x) {
                    cropped(c, y, x) = image(c, y_offset + y, x_offset + x);
                }
            }
        }

        return cropped;
    }
};

template<typename Desc>
struct random_cropper <Desc, std::enable_if_t<!Desc::random_crop_x || !Desc::random_crop_y>> {
    template<typename T>
    random_cropper(const T& image){
        cpp_unused(image);
    }

    static constexpr size_t scaling() {
        return 1;
    }

    template<typename T>
    static auto crop(T&& image) {
        return std::forward<T>(image);
    }
};

template<typename Iterator, typename LIterator, typename Desc>
struct memory_data_generator <Iterator, LIterator, Desc, std::enable_if_t<is_augmented<Desc>::value>> {
    using desc = Desc;
    using weight = typename desc::weight;
    using cache_helper_t = cache_helper<desc, Iterator>;

    using cache_type = typename cache_helper_t::cache_type;
    using big_cache_type = typename cache_helper_t::big_cache_type;
    using label_type = etl::dyn_matrix<weight, 2>;

    static constexpr size_t batch_size = desc::BatchSize;
    static constexpr size_t big_batch_size = desc::BigBatchSize;

    cache_type input_cache;
    big_cache_type batch_cache;
    label_type labels;

    random_cropper<Desc> cropper;

    size_t current = 0;

    mutable volatile bool status[big_batch_size];
    mutable volatile size_t indices[big_batch_size];

    mutable std::mutex main_lock;
    mutable std::condition_variable condition;
    mutable std::condition_variable ready_condition;

    volatile bool stop_flag = false;

    std::thread main_thread;
    bool threaded = false;

    memory_data_generator(Iterator first, Iterator last, LIterator lfirst, LIterator llast, size_t n_classes) : cropper(*first) {
        const size_t n = std::distance(first, last);

        cache_helper_t::init(n, first, input_cache);
        cache_helper_t::init_big(big_batch_size, batch_size, first, batch_cache);

        labels = label_type(n, n_classes);

        labels = weight(0);

        size_t i = 0;
        while(first != last){
            input_cache(i) = *first;

            labels(i, *lfirst) = weight(1);

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
                        condition.wait(ulock, [this] {
                            if(stop_flag){
                                return true;
                            }

                            for(size_t b = 0; b < big_batch_size; ++b){
                                if(!status[b] && indices[b] * batch_size < size()){
                                    return true;
                                }
                            }

                            return false;
                        });

                        // If there is no more thread for the thread, exit
                        if (stop_flag) {
                            return;
                        }

                        // Pick a batch to fill
                        for(size_t b = 0; b < big_batch_size; ++b){
                            if(!status[b] && indices[b] * batch_size < size()){
                                index = b;
                                break;
                            }
                        }
                    }
                }

                // Get the batch that needs to be read
                const size_t batch = indices[index];

                // Get the index from where to read inside the input cache
                const size_t input_n = batch * batch_size;

                for(size_t i = 0; i < batch_size; ++i){
                    batch_cache(index)(i) = cropper.crop(input_cache(input_n + i));
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

        etl::parallel_shuffle(input_cache, labels);
    }

    size_t current_batch() const {
        return current / batch_size;
    }

    size_t size() const {
        return etl::dim<0>(input_cache);
    }

    size_t augmented_size() const {
        return cropper.scaling() * etl::dim<0>(input_cache);
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
        return etl::slice(labels, current, std::min(current + batch_size, size()));
    }

    static constexpr size_t dimensions() {
        return etl::dimensions<cache_type>() - 1;
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
     * \brief The random cropping X
     */
    static constexpr size_t random_crop_x = detail::get_value_1<random_crop<0,0>, Parameters...>::value;

    /*!
     * \brief The random cropping Y
     */
    static constexpr size_t random_crop_y = detail::get_value_2<random_crop<0,0>, Parameters...>::value;

    /*!
     * The type used to store the weights
     */
    using weight = typename detail::get_type<weight_type<float>, Parameters...>::value;

    static_assert(BatchSize > 0, "The batch size must be larger than one");

    //Make sure only valid types are passed to the configuration list
    static_assert(
        detail::is_valid<cpp::type_list<batch_size_id, big_batch_size_id, random_crop_id, nop_id>,
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
