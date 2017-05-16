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
    static constexpr bool value = (Desc::random_crop_x > 0 && Desc::random_crop_y > 0) || Desc::HorizontalMirroring || Desc::VerticalMirroring || Desc::ElasticDistortion;
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

    template<typename O, typename T>
    void transform_first(O&& target, const T& image){
        const size_t y_offset = dist_y(engine);
        const size_t x_offset = dist_x(engine);

        for (size_t c = 0; c < etl::dim<0>(image); ++c) {
            for (size_t y = 0; y < random_crop_y; ++y) {
                for (size_t x = 0; x < random_crop_x; ++x) {
                    target(c, y, x) = image(c, y_offset + y, x_offset + x);
                }
            }
        }
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

    template<typename O, typename T>
    void transform_first(O&& target, const T& image){
        target = image;
    }
};

template<typename Desc, typename Enable = void>
struct random_mirrorer;

template<typename Desc>
struct random_mirrorer <Desc, std::enable_if_t<Desc::HorizontalMirroring || Desc::VerticalMirroring>> {
    static constexpr bool horizontal = Desc::HorizontalMirroring;
    static constexpr bool vertical = Desc::VerticalMirroring;

    std::random_device rd;
    std::default_random_engine engine;

    std::uniform_int_distribution<size_t> dist;

    template<typename T>
    random_mirrorer(const T& image) : engine(rd()) {
        static_assert(etl::dimensions<T>() == 3, "random_mirrorer can only be used with 3D images");

        if(horizontal && vertical){
            dist = std::uniform_int_distribution<size_t>{0, 3};
        } else {
            dist = std::uniform_int_distribution<size_t>{0, 2};
        }

        cpp_unused(image);
    }

    size_t scaling() const {
        if(horizontal && vertical){
            return 3;
        } else {
            return 2;
        }
    }

    template<typename O>
    void transform(O&& target){
        auto choice = dist(engine);

        if(horizontal && vertical && choice == 1){
            for(size_t c= 0; c < etl::dim<0>(target); ++c){
                target(c) = vflip(target(c));
            }
        } else if(horizontal && vertical && choice == 2){
            for(size_t c= 0; c < etl::dim<0>(target); ++c){
                target(c) = hflip(target(c));
            }
        }

        if(horizontal && choice == 1){
            for(size_t c= 0; c < etl::dim<0>(target); ++c){
                target(c) = hflip(target(c));
            }
        }

        if(vertical && choice == 1){
            for(size_t c= 0; c < etl::dim<0>(target); ++c){
                target(c) = vflip(target(c));
            }
        }
    }
};

template<typename Desc>
struct random_mirrorer <Desc, std::enable_if_t<!Desc::HorizontalMirroring && !Desc::VerticalMirroring>> {
    template<typename T>
    random_mirrorer(const T& image){
        cpp_unused(image);
    }

    static constexpr size_t scaling() {
        return 1;
    }

    template<typename O>
    static void transform(O&& target){
        cpp_unused(target);
    }
};

template<typename Desc, typename Enable = void>
struct elastic_distorter;

// TODO This needs to be made MUCH faster

template<typename Desc>
struct elastic_distorter <Desc, std::enable_if_t<Desc::ElasticDistortion>> {
    using weight = typename Desc::weight;

    static constexpr size_t K     = Desc::ElasticDistortion;
    static constexpr size_t mid   = K / 2;
    static constexpr double sigma = 0.8 + 0.3 * ((K - 1) * 0.5 - 1);

    etl::fast_dyn_matrix<weight, K, K> kernel;

    static_assert(K % 2 == 1, "The kernel size must be odd");

    template<typename T>
    elastic_distorter(const T& image) {
        static_assert(etl::dimensions<T>() == 3, "elastic_distorter can only be used with 3D images");

        cpp_unused(image);

        // Precompute the gaussian kernel

        auto gaussian = [](double x, double y) {
            auto Z = 2.0 * M_PI * sigma * sigma;
            return (1.0 / Z) * std::exp(-((x * x + y * y) / (2.0 * sigma * sigma)));
        };

        for (size_t i = 0; i < K; ++i) {
            for (size_t j = 0; j < K; ++j) {
                kernel(i, j) = gaussian(double(i) - mid, double(j) - mid);
            }
        }
    }

    size_t scaling() const {
        return 10;
    }

    template<typename O>
    void transform(O&& target){
        const size_t width  = etl::dim<1>(target);
        const size_t height = etl::dim<2>(target);

        // 0. Generate random displacement fields

        etl::dyn_matrix<weight> d_x(width, height);
        etl::dyn_matrix<weight> d_y(width, height);

        d_x = etl::uniform_generator(-1.0, 1.0);
        d_y = etl::uniform_generator(-1.0, 1.0);

        // 1. Gaussian blur the displacement fields

        etl::dyn_matrix<weight> d_x_blur(width, height);
        etl::dyn_matrix<weight> d_y_blur(width, height);

        gaussian_blur(d_x, d_x_blur);
        gaussian_blur(d_y, d_y_blur);

        // 2. Normalize and scale the displacement field

        d_x_blur *= (weight(8) / sum(d_x_blur));
        d_y_blur *= (weight(8) / sum(d_y_blur));

        // 3. Apply the displacement field (using bilinear interpolation)

        auto safe = [&](size_t channel, weight x, weight y) {
            if (x < 0 || y < 0 || x > width - 1 || y > height - 1) {
                return target(channel, 0UL, 0UL);
            } else {
                return target(channel, size_t(x), size_t(y));
            }
        };

        for (size_t channel = 0; channel < etl::dim<0>(target); ++channel) {
            for (int x = 0; x < int(width); ++x) {
                for (int y = 0; y < int(height); ++y) {
                    auto dx = d_x_blur(x, y);
                    auto dy = d_y_blur(x, y);

                    weight px = x + dx;
                    weight py = y + dy;

                    weight a = safe(channel, std::floor(px), std::floor(py));
                    weight b = safe(channel, std::ceil(px), std::floor(py));
                    weight c = safe(channel, std::ceil(px), std::ceil(py));
                    weight d = safe(channel, std::floor(px), std::ceil(py));

                    auto e = a * (1.0 - (px - std::floor(px))) + d * (px - std::floor(px));
                    auto f = b * (1.0 - (px - std::floor(px))) + c * (px - std::floor(px));

                    auto value = e * (1.0 - (py - std::floor(py))) + f * (py - std::floor(py));

                    target(channel, x, y) = value;
                }
            }
        }
    }

    void gaussian_blur(const etl::dyn_matrix<weight>& d, etl::dyn_matrix<weight>& d_blur){
        const size_t width = etl::dim<0>(d);
        const size_t height = etl::dim<1>(d);

        for (size_t j = 0; j < width; ++j) {
            for (size_t k = 0; k < height; ++k) {
                weight sum(0.0);

                for (size_t p = 0; p < K; ++p) {
                    if (long(j) + p - mid >= 0 && long(j) + p - mid < width) {
                        for (size_t q = 0; q < K; ++q) {
                            if (long(k) + q - mid >= 0 && long(k) + q - mid < height) {
                                sum += kernel(p, q) * d(j + p - mid, k + q - mid);
                            }
                        }
                    }
                }

                d_blur(j, k) = d(j, k) - (sum / (K * K));
            }
        }
    }
};

template<typename Desc>
struct elastic_distorter <Desc, std::enable_if_t<!Desc::ElasticDistortion>> {
    template<typename T>
    elastic_distorter(const T& image){
        cpp_unused(image);
    }

    static constexpr size_t scaling() {
        return 1;
    }

    template<typename O>
    static void transform(O&& target){
        cpp_unused(target);
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
    random_mirrorer<Desc> mirrorer;
    elastic_distorter<Desc> distorter;

    size_t current = 0;

    mutable volatile bool status[big_batch_size];
    mutable volatile size_t indices[big_batch_size];

    mutable std::mutex main_lock;
    mutable std::condition_variable condition;
    mutable std::condition_variable ready_condition;

    volatile bool stop_flag = false;

    std::thread main_thread;
    bool threaded = false;

    memory_data_generator(Iterator first, Iterator last, LIterator lfirst, LIterator llast, size_t n_classes) : cropper(*first), mirrorer(*first), distorter(*first) {
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
        return cropper.scaling() * mirrorer.scaling() * etl::dim<0>(input_cache);
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
     * The type used to store the weights
     */
    using weight = typename detail::get_type<weight_type<float>, Parameters...>::value;

    static_assert(BatchSize > 0, "The batch size must be larger than one");

    //Make sure only valid types are passed to the configuration list
    static_assert(
        detail::is_valid<cpp::type_list<batch_size_id, big_batch_size_id, horizontal_mirroring_id, vertical_mirroring_id, random_crop_id, elastic_distortion_id, nop_id>,
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
