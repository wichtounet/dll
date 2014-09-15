//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_CONV_DBN_HPP
#define DLL_CONV_DBN_HPP

#include <tuple>

#include "etl/dyn_vector.hpp"
#include "etl/dyn_matrix.hpp"

#include "conv_rbm.hpp"
#include "tuple_utils.hpp"
#include "dbn_trainer.hpp"
#include "conjugate_gradient.hpp"

namespace dll {

/*!
 * \brief A Deep Belief Network implementation
 */
template<typename Desc>
struct conv_dbn {
    using desc = Desc;

    using tuple_type = typename desc::layers::tuple_type;
    tuple_type tuples;

    static constexpr const std::size_t layers = desc::layers::layers;

    template <std::size_t N>
    using rbm_type = typename std::tuple_element<N, tuple_type>::type;

    using weight = typename rbm_type<0>::weight;

    //No arguments by default
    conv_dbn(){};

    //No copying
    conv_dbn(const conv_dbn& dbn) = delete;
    conv_dbn& operator=(const conv_dbn& dbn) = delete;

    //No moving
    conv_dbn(conv_dbn&& dbn) = delete;
    conv_dbn& operator=(conv_dbn&& dbn) = delete;

    void display() const {
        std::size_t parameters = 0;

        detail::for_each(tuples, [&parameters](auto& rbm){
            typedef typename std::remove_reference<decltype(rbm)>::type rbm_t;

            constexpr const auto NV = rbm_t::NV;
            constexpr const auto NH = rbm_t::NH;

            std::cout << "RBM: " << NV << "x" << NV << "->" << NH << "x" << NH << std::endl;
        });
    }

    void store(std::ostream& os) const {
        detail::for_each(tuples, [&os](auto& rbm){
            rbm.store(os);
        });
    }

    void load(std::istream& is){
        detail::for_each(tuples, [&is](auto& rbm){
            rbm.load(is);
        });
    }

    template<std::size_t N>
    auto layer() -> typename std::add_lvalue_reference<rbm_type<N>>::type {
        return std::get<N>(tuples);
    }

    template<std::size_t N>
    constexpr auto layer() const -> typename std::add_lvalue_reference<typename std::add_const<rbm_type<N>>::type>::type {
        return std::get<N>(tuples);
    }

    template<std::size_t N>
    static constexpr std::size_t nv(){
        return rbm_type<N>::NV;
    }

    template<std::size_t N>
    static constexpr std::size_t nh(){
        return rbm_type<N>::NH;
    }

    /*{{{ Pretrain */

    /*!
     * \brief Pretrain the network by training all layers in an unsupervised
     * manner.
     */
    template<typename Samples>
    void pretrain(const Samples& training_data, std::size_t max_epochs){
        using visible_t = std::vector<etl::dyn_vector<typename Samples::value_type::value_type>>;
        using hidden_t = std::vector<etl::dyn_vector<etl::dyn_matrix<weight>>>;

        //Convert data to an useful form
        visible_t data;
        data.reserve(training_data.size());

        for(auto& sample : training_data){
            data.emplace_back(sample);
        }

        hidden_t next_a;
        hidden_t next_s;
        visible_t next;

        auto input = std::ref(data);

        detail::for_each_i(tuples, [&input, &next, &next_a, &next_s, max_epochs](std::size_t I, auto& rbm){
            typedef typename std::remove_reference<decltype(rbm)>::type rbm_t;
            constexpr const auto NV = rbm_t::NV;
            constexpr const auto NH = rbm_t::NH;
            constexpr const auto K = rbm_t::K;

            auto input_size = static_cast<const visible_t&>(input).size();

            //TODO Train every layers but the one with EXP hidden unit

            //Train each layer but the last one
            if(I <= layers - 2){
                std::cout << "DBN: Train layer " << I << " (" << NV << "x" << NV << "->" << NH << "x" << "NH" << ") with " << input_size << " entries" << std::endl;

                rbm.train(static_cast<const visible_t&>(input), max_epochs);

                //Get the activation probabilities for the next level
                if(I < layers - 2){
                    next_a.clear();
                    next_a.reserve(input_size);
                    next_s.clear();
                    next_s.reserve(input_size);

                    //TODO Review that
                    for(std::size_t i = 0; i < input_size; ++i){
                        next_a.emplace_back(K, etl::dyn_matrix<weight>(NH, NH));
                        next_s.emplace_back(K, etl::dyn_matrix<weight>(NH, NH));
                    }

                    for(std::size_t i = 0; i < input_size; ++i){
                        rbm.v1 = static_cast<const visible_t&>(input)[i];
                        rbm.activate_hidden(next_a[i], next_s[i], rbm.v1, rbm.v1);
                    }

                    next.clear();
                    next.reserve(input_size);

                    //TODO Check the order of the output

                    for(std::size_t i = 0; i < input_size; ++i){
                        next.emplace_back(NH * NH * K);

                        for(std::size_t j = 0; j < NH; ++j){
                            for(std::size_t k = 0; k < NH; ++k){
                                for(std::size_t l = 0; l < K; ++l){
                                    next[i][j * NH * NH + k * NH + l] = next_a[i](k)(j,k);
                                }
                            }
                        }
                    }

                    input = std::ref(next);
                }
            }
        });
    }

    /*}}}*/

    /*{{{ Predict */

    //template<typename Sample, typename Output>
    //void predict_weights(const Sample& item_data, Output& result){
        //etl::dyn_vector<typename Sample::value_type> item(item_data);

        //auto input = std::cref(item);

        //detail::for_each_i(tuples, [&item, &input, &result](std::size_t I, auto& rbm){
            //if(I != layers - 1){
                //typedef typename std::remove_reference<decltype(rbm)>::type rbm_t;
                //constexpr const auto num_hidden = rbm_t::num_hidden;

                //static etl::dyn_vector<weight> next(num_hidden);
                //static etl::dyn_vector<weight> next_s(num_hidden);

                //rbm.activate_hidden(next, next_s, static_cast<const Sample&>(input), static_cast<const Sample&>(input));

                //input = std::cref(next);
            //}
        //});

        //constexpr const auto num_hidden = rbm_type<layers - 1>::num_hidden;

        //static etl::dyn_vector<weight> next_s(num_hidden);

        //layer<layers - 1>().activate_hidden(result, next_s, static_cast<const Sample&>(input), static_cast<const Sample&>(input));
    //}

    //template<typename Sample>
    //etl::dyn_vector<weight> predict_weights(const Sample& item_data){
        //etl::dyn_vector<weight> result(num_hidden<layers - 1>());

        //etl::dyn_vector<typename Sample::value_type> item(item_data);

        //auto input = std::cref(item);

        //detail::for_each_i(tuples, [&item, &input, &result](std::size_t I, auto& rbm){
            //typedef typename std::remove_reference<decltype(rbm)>::type rbm_t;
            //constexpr const auto num_hidden = rbm_t::num_hidden;

            //static etl::dyn_vector<weight> next(num_hidden);
            //static etl::dyn_vector<weight> next_s(num_hidden);
            //auto& output = (I == layers - 1) ? result : next;

            //rbm.activate_hidden(output, next_s, static_cast<const Sample&>(input), static_cast<const Sample&>(input));

            //input = std::cref(next);
        //});

        //return result;
    //}

    //template<typename Weights>
    //size_t predict_final(const Weights& result){
        //size_t label = 0;
        //weight max = 0;
        //for(size_t l = 0; l < result.size(); ++l){
            //auto value = result[l];

            //if(value > max){
                //max = value;
                //label = l;
            //}
        //}

        //return label;
    //}

    //template<typename Sample>
    //size_t predict(const Sample& item){
        //auto result = predict_weights(item);
        //return predict_final(result);;
    //}

    /*}}}*/
};

} //end of namespace dll

#endif
