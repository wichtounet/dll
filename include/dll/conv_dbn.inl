//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_CONV_DBN_INL
#define DLL_CONV_DBN_INL

#include <tuple>

#include "etl/dyn_vector.hpp"
#include "etl/dyn_matrix.hpp"

#include "conv_rbm.hpp"
#include "tuple_utils.hpp"
#include "dbn_trainer.hpp"
#include "conjugate_gradient.hpp"
#include "dbn_common.hpp"

//SVM Support is optional cause it requires libsvm

#ifdef DLL_SVM_SUPPORT
#include "nice_svm.hpp"
#endif

namespace dll {

/*!
 * \brief A Deep Belief Network implementation
 */
template<typename Desc>
struct conv_dbn {
    using desc = Desc;
    using this_type = conv_dbn<desc>;

    using tuple_type = typename desc::layers::tuple_type;
    tuple_type tuples;

    static constexpr const std::size_t layers = desc::layers::layers;

    template <std::size_t N>
    using rbm_type = typename std::tuple_element<N, tuple_type>::type;

    using weight = typename rbm_type<0>::weight;

#ifdef DLL_SVM_SUPPORT
    svm::model svm_model;               ///< The learned model
    svm::problem problem;               ///< libsvm is stupid, therefore, you cannot destroy the problem if you want to use the model...
    bool svm_loaded = false;            ///< Indicates if a SVM model has been loaded (and therefore must be saved)
#endif //DLL_SVM_SUPPORT

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
            constexpr const auto K = rbm_t::K;

            printf("RBM: %lux%lu -> %lux%lu (%lu)\n", NV, NV, NH, NH, K);
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
    static constexpr std::size_t rbm_nv(){
        return rbm_type<N>::NV;
    }

    template<std::size_t N>
    static constexpr std::size_t rbm_nh(){
        return rbm_type<N>::NH;
    }

    template<std::size_t N>
    static constexpr std::size_t rbm_k(){
        return rbm_type<N>::K;
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

        using watcher_t = typename desc::template watcher_t<this_type>;

        watcher_t watcher;

        watcher.pretraining_begin(*this);

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

        detail::for_each_i(tuples, [&watcher, this, &input, &next, &next_a, &next_s, max_epochs](std::size_t I, auto& rbm){
            typedef typename std::remove_reference<decltype(rbm)>::type rbm_t;
            constexpr const auto NH = rbm_t::NH;
            constexpr const auto K = rbm_t::K;

            auto input_size = static_cast<const visible_t&>(input).size();

            watcher.template pretrain_layer<rbm_t>(*this, I, input_size);

            rbm.template train<
                    visible_t,
                    !watcher_t::ignore_sub,                                 //Enable the RBM Watcher or not
                    typename dbn_detail::rbm_watcher_t<watcher_t>::type>    //Replace the RBM watcher if not void
                (static_cast<const visible_t&>(input), max_epochs);

            //Get the activation probabilities for the next level
            if(I < layers - 1){
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
        });

        watcher.pretraining_end(*this);
    }

    /*}}}*/

    /*{{{ Predict */

    template<typename Sample, typename Output>
    void activation_probabilities(const Sample& item_data, Output& result){
        using visible_t = etl::dyn_vector<typename Sample::value_type>;
        using hidden_t = etl::dyn_vector<etl::dyn_matrix<weight>>;

        visible_t item(item_data);

        auto input = std::cref(item);

        detail::for_each_i(tuples, [&item, &input, &result](std::size_t I, auto& rbm){
            if(I != layers - 1){
                typedef typename std::remove_reference<decltype(rbm)>::type rbm_t;
                constexpr const auto NH = rbm_t::NH;
                constexpr const auto K = rbm_t::K;

                static visible_t next(K * NH * NH);
                static hidden_t next_a(K, etl::dyn_matrix<weight>(NH, NH));
                static hidden_t next_s(K, etl::dyn_matrix<weight>(NH, NH));

                rbm.v1 = static_cast<const Sample&>(input);
                rbm.activate_hidden(next_a, next_s, rbm.v1, rbm.v1);

                //TODO Check the order of the output

                for(std::size_t j = 0; j < NH; ++j){
                    for(std::size_t k = 0; k < NH; ++k){
                        for(std::size_t l = 0; l < K; ++l){
                            next[j * NH * NH + k * NH + l] = next_a(k)(j,k);
                        }
                    }
                }

                input = std::cref(next);
            }
        });

        constexpr const auto K = rbm_k<layers - 1>();
        constexpr const auto NH = rbm_nh<layers - 1>();

        static hidden_t next_a(K, etl::dyn_matrix<weight>(NH, NH));
        static hidden_t next_s(K, etl::dyn_matrix<weight>(NH, NH));

        auto& last_rbm = layer<layers - 1>();

        last_rbm.v1 = static_cast<const Sample&>(input);
        last_rbm.activate_hidden(next_a, next_s, last_rbm.v1, last_rbm.v1);

        //TODO Check the order of the output

        for(std::size_t j = 0; j < NH; ++j){
            for(std::size_t k = 0; k < NH; ++k){
                for(std::size_t l = 0; l < K; ++l){
                    result[j * NH * NH + k * NH + l] = next_a(k)(j,k);
                }
            }
        }
    }

    template<typename Sample>
    etl::dyn_vector<weight> activation_probabilities(const Sample& item_data){
        etl::dyn_vector<weight> result(rbm_nh<layers - 1>() * rbm_nh<layers - 1>() * rbm_k<layers - 1>());

        activation_probabilities(item_data, result);

        return result;
    }

    template<typename Weights>
    size_t predict_label(const Weights& result){
        size_t label = 0;
        weight max = 0;
        for(size_t l = 0; l < result.size(); ++l){
            auto value = result[l];

            if(value > max){
                max = value;
                label = l;
            }
        }

        return label;
    }

    template<typename Sample>
    size_t predict(const Sample& item){
        auto result = activation_probabilities(item);
        return predict_label(result);;
    }

    /*}}}*/

#ifdef DLL_SVM_SUPPORT

    /*{{{ SVM Training and prediction */



    /*}}}*/

#endif //DLL_SVM_SUPPORT

};

} //end of namespace dll

#endif
