//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_INPUT_CONVERTER_HPP
#define DLL_INPUT_CONVERTER_HPP

namespace dll {

//Standard implementation
//Use convert_input on the layer to convert the input

template<typename DBN, std::size_t I, typename Iterator, typename Enable = void>
struct input_converter {
    using layer_t = typename DBN::template rbm_type<I>;
    using container = decltype(std::declval<layer_t>().convert_input(std::declval<Iterator>(), std::declval<Iterator>()));

    container c;

    input_converter(DBN& dbn, const Iterator& first, const Iterator& last) : c(dbn.template layer<I>().convert_input(first, last)) {
        //Nothing else to init
    }

    auto begin(){
        return c.begin();
    }

    auto end(){
        return c.end();
    }
};

//Fast specialization when the type is already of the correct type
//Simply returns the input without conversion

template<typename DBN, std::size_t I, typename Iterator>
struct input_converter <DBN, I, Iterator, std::enable_if_t<
            std::is_same<typename DBN::template rbm_type<I>::input_one_t, typename Iterator::value_type>::value
        &&  !layer_traits<typename DBN::template rbm_type<I>>::is_multiplex_layer()>> {
    Iterator& first;
    Iterator& last;

    input_converter(DBN& /*dbn*/, Iterator& first, Iterator& last) : first(first), last(last) {
        //Nothing else to init
    }

    Iterator& begin(){
        return first;
    }

    Iterator& end(){
        return last;
    }
};

//Specialization for transform layer
//Use the next layer to perform the conversion

template<typename DBN, std::size_t I, typename Iterator>
struct input_converter <DBN, I, Iterator, std::enable_if_t<layer_traits<typename DBN::template rbm_type<I>>::is_transform_layer()>> :
        input_converter<DBN, I + 1, Iterator> {
    using input_converter<DBN, I + 1, Iterator>::input_converter;
};

//Specialization for multiplex layer
//Fails if the type is not the same

template<typename DBN, std::size_t I, typename Iterator>
struct input_converter <DBN, I, Iterator, std::enable_if_t<layer_traits<typename DBN::template rbm_type<I>>::is_multiplex_layer()>> {
    static_assert(
        std::is_same<typename DBN::template rbm_type<I>::input_one_t, typename Iterator::value_type>::value,
        "Multiplex layer cannot perform input conversion on their own");

    Iterator& first;
    Iterator& last;

    input_converter(DBN& /*dbn*/, Iterator& first, Iterator& last) : first(first), last(last) {
        //Nothing else to init
    }

    Iterator& begin(){
        return first;
    }

    Iterator& end(){
        return last;
    }
};

//Standard implementation
//Use convert_sample  on the layer to convert the input

template<typename DBN, std::size_t I, typename Sample, typename Enable = void>
struct sample_converter {
    using layer_t = typename DBN::template rbm_type<I>;
    using result = decltype(std::declval<layer_t>().convert_sample(std::declval<Sample>()));

    result r;

    sample_converter(const DBN& dbn, const Sample& sample) : r(dbn.template layer<I>().convert_sample(sample)) {
        //Nothing else to init
    }

    decltype(auto) get(){
        return r;
    }
};

//Fast specialization when the type is already of the correct type
//Simply returns the input without conversion

template<typename DBN, std::size_t I, typename Sample>
struct sample_converter <DBN, I, Sample, std::enable_if_t<
            std::is_same<typename DBN::template rbm_type<I>::input_one_t, Sample>::value
        &&  !layer_traits<typename DBN::template rbm_type<I>>::is_multiplex_layer()>> {
    const Sample& r;

    sample_converter(const DBN& /*dbn*/, const Sample& sample) : r(sample) {
        //Nothing else to init
    }

    decltype(auto) get(){
        return r;
    }
};

//Specialization for transform layer
//Use the next layer to perform the conversion

template<typename DBN, std::size_t I, typename Sample>
struct sample_converter <DBN, I, Sample, std::enable_if_t<layer_traits<typename DBN::template rbm_type<I>>::is_transform_layer()>> :
        sample_converter<DBN, I + 1, Sample> {
    using sample_converter<DBN, I + 1, Sample>::sample_converter;
};

//Specialization for multiplex layer
//Fails if the type is not the same

template<typename DBN, std::size_t I, typename Sample>
struct sample_converter <DBN, I, Sample, std::enable_if_t<layer_traits<typename DBN::template rbm_type<I>>::is_multiplex_layer()>> {
    static_assert(
        std::is_same<typename DBN::template rbm_type<I>::input_one_t, Sample>::value,
        "Multiplex layer cannot perform input conversion on their own");

    const Sample& r;

    sample_converter(const DBN& /*dbn*/, const Sample& sample) : r(sample) {
        //Nothing else to init
    }

    decltype(auto) get(){
        return r;
    }
};

} //end of namespace dll

#endif
