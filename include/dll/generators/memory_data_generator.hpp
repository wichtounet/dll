//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace dll {

template<typename T, typename Iterator, typename Enable = void>
struct cache_helper;

template<typename T, typename Iterator>
struct cache_helper<T, Iterator, std::enable_if_t<etl::is_1d<typename Iterator::value_type>::value>> {
    using cache_type = etl::dyn_matrix<T, 2>;

    static void init(size_t n, Iterator& it, cache_type& cache){
        auto one = *it;
        cache = cache_type(n, etl::dim<0>(one));
    }
};

template<typename T, typename Iterator>
struct cache_helper<T, Iterator, std::enable_if_t<etl::is_3d<typename Iterator::value_type>::value>> {
    using cache_type = etl::dyn_matrix<T, 4>;

    static void init(size_t n, Iterator& it, cache_type& cache){
        auto one = *it;
        cache = cache_type(n, etl::dim<0>(one), etl::dim<1>(one), etl::dim<2>(one));
    }
};

template<typename Iterator, typename LIterator, typename Desc>
struct memory_data_generator {
    using desc = Desc;
    using weight = typename desc::weight;

    using cache_type = typename cache_helper<weight, Iterator>::cache_type;
    using label_type = etl::dyn_matrix<weight, 2>;

    static constexpr size_t batch_size = desc::BatchSize;

    cache_type cache;
    label_type labels;

    size_t current = 0;

    memory_data_generator(Iterator first, Iterator last, LIterator lfirst, LIterator llast, size_t n_classes){
        const size_t n = std::distance(first, last);

        cache_helper<weight, Iterator>::init(n, first, cache);
        labels = label_type(n, n_classes);

        labels = weight(0);

        size_t i = 0;
        while(first != last){
            cache(i) = *first;

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

        etl::parallel_shuffle(cache, labels);
    }

    size_t current_batch() const {
        return current / batch_size;
    }

    size_t size() const {
        return etl::dim<0>(cache);
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
        return etl::slice(cache, current, std::min(current + batch_size, size()));
    }

    auto label_batch() const {
        return etl::slice(labels, current, std::min(current + batch_size, size()));
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
     * The type used to store the weights
     */
    using weight = typename detail::get_type<weight_type<float>, Parameters...>::value;

    static_assert(BatchSize > 0, "The batch size must be larger than one");

    //Make sure only valid types are passed to the configuration list
    static_assert(
        detail::is_valid<cpp::type_list<batch_size_id, nop_id>,
                         Parameters...>::value,
        "Invalid parameters type for rbm_desc");

    /*!
     * The generator type
     */
    template<typename Iterator, typename LIterator>
    using generator_t = memory_data_generator<Iterator, LIterator, memory_data_generator_desc<Parameters...>>;
};

template<typename Iterator, typename LIterator, typename... Parameters>
typename memory_data_generator_desc<Parameters...>::template generator_t<Iterator, LIterator>
make_generator(Iterator first, Iterator last, LIterator lfirst, LIterator llast, size_t n_classes, const memory_data_generator_desc<Parameters...>& /*desc*/){
    return {first, last, lfirst, llast, n_classes};
}

template<typename Container, typename LContainer, typename... Parameters>
auto make_generator(const Container& container, const LContainer& lcontainer, size_t n_classes, const memory_data_generator_desc<Parameters...>& desc){
    return make_generator(container.begin(), container.end(), lcontainer.begin(), lcontainer.end(), n_classes, desc);
}

} //end of dll namespace
