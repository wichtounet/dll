//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_CONV_DBN_INL
#define DLL_CONV_DBN_INL

#include <tuple>

#include "cpp_utils/tuple_utils.hpp"

#include "etl/etl.hpp"

#include "io.hpp"
#include "dbn_trainer.hpp"
#include "dbn_common.hpp"
#include "svm_common.hpp"

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

        cpp::for_each(tuples, [&parameters](auto& rbm){
            typedef typename std::remove_reference<decltype(rbm)>::type rbm_t;

            constexpr const auto NV = rbm_t::NV;
            constexpr const auto NH = rbm_t::NH;
            constexpr const auto K = rbm_t::K;

            printf("RBM: %lux%lu -> %lux%lu (%lu)\n", NV, NV, NH, NH, K);
        });
    }

    void store(std::ostream& os) const {
        cpp::for_each(tuples, [&os](auto& rbm){
            rbm.store(os);
        });

#ifdef DLL_SVM_SUPPORT
        svm_store(*this, os);
#endif //DLL_SVM_SUPPORT
    }

    void load(std::istream& is){
        cpp::for_each(tuples, [&is](auto& rbm){
            rbm.load(is);
        });

#ifdef DLL_SVM_SUPPORT
        svm_load(*this, is);
#endif //DLL_SVM_SUPPORT
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
    static constexpr std::size_t rbm_k(){
        return rbm_type<N>::K;
    }

    template<std::size_t N>
    static constexpr std::size_t rbm_nh(){
        return rbm_type<N>::NH;
    }

    template<typename RBM, cpp::enable_if_u<rbm_traits<RBM>::has_probabilistic_max_pooling()> = cpp::detail::dummy>
    static constexpr std::size_t rbm_t_no(){
        return RBM::NP;
    }

    template<typename RBM, cpp::disable_if_u<rbm_traits<RBM>::has_probabilistic_max_pooling()> = cpp::detail::dummy>
    static constexpr std::size_t rbm_t_no(){
        return RBM::NH;
    }

    template<std::size_t N>
    static constexpr std::size_t rbm_no(){
        return rbm_t_no<rbm_type<N>>();
    }

    template<std::size_t N>
    static constexpr std::size_t rbm_input(){
        return rbm_traits<rbm_type<N>>::input_size();
    }

    template<std::size_t N>
    static constexpr std::size_t rbm_output(){
        return rbm_traits<rbm_type<N>>::output_size();
    }

    static constexpr std::size_t input_size(){
        return rbm_output<0>();
    }

    static constexpr std::size_t output_size(){
        return rbm_output<layers - 1>();
    }

    static std::size_t full_output_size(){
        std::size_t output;
        for_each_type<tuple_type>([&output](auto* rbm){
            output += std::decay_t<std::remove_pointer_t<decltype(rbm)>>::output_size();
        });
        return output;
    }

    /*{{{ Pretrain */

    template<typename RBM, typename Input, typename Next, cpp::enable_if_u<rbm_traits<RBM>::has_probabilistic_max_pooling()> = cpp::detail::dummy>
    static void propagate(RBM& rbm, Input& input, Next& next_a, Next& next_s){
        rbm.v1 = input;
        rbm.activate_pooling(next_a, next_s, rbm.v1, rbm.v1);
    }

    template<typename RBM, typename Input, typename Next, cpp::disable_if_u<rbm_traits<RBM>::has_probabilistic_max_pooling()> = cpp::detail::dummy>
    static void propagate(RBM& rbm, Input& input, Next& next_a, Next& next_s){
        rbm.v1 = input;
        rbm.activate_hidden(next_a, next_s, rbm.v1, rbm.v1);
    }

    /*!
     * \brief Pretrain the network by training all layers in an unsupervised
     * manner.
     */
    template<typename Samples>
    void pretrain(const Samples& training_data, std::size_t max_epochs){
        using visible_t = std::vector<etl::dyn_matrix<weight, 3>>;
        using hidden_t = std::vector<etl::dyn_matrix<weight, 3>>;

        using watcher_t = typename desc::template watcher_t<this_type>;

        watcher_t watcher;

        watcher.pretraining_begin(*this);

        //I don't know why it is necesary to copy them here
        constexpr const auto NC = rbm_type<0>::NC;
        constexpr const auto NV = rbm_type<0>::NV;

        //Convert data to an useful form
        visible_t data;
        data.reserve(training_data.size());

        for(auto& sample : training_data){
            data.emplace_back(NC, NV, NV);
            data.back() = sample;
        }

        hidden_t next_a;
        hidden_t next_s;

        auto input = std::ref(data);

        cpp::for_each_i(tuples, [&watcher, this, &input, &next_a, &next_s, max_epochs](std::size_t I, auto& rbm){
            typedef typename std::remove_reference<decltype(rbm)>::type rbm_t;
            constexpr const auto K = rbm_t::K;
            constexpr const auto NO = rbm_t_no<rbm_t>();

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

                for(std::size_t i = 0; i < input_size; ++i){
                    next_a.emplace_back(K, NO, NO);
                    next_s.emplace_back(K, NO, NO);
                }

                for(std::size_t i = 0; i < input_size; ++i){
                    propagate(rbm, static_cast<const visible_t&>(input)[i], next_a[i], next_s[i]);
                }

                input = std::ref(next_a);
            }
        });

        watcher.pretraining_end(*this);
    }

    /*}}}*/

    /*{{{ Predict */

    template<typename Sample, typename Output>
    void activation_probabilities(const Sample& item_data, Output& result){
        using visible_t = etl::dyn_matrix<weight, 3>;
        using hidden_t = etl::dyn_matrix<weight, 3>;

        visible_t item(rbm_type<0>::NC, rbm_type<0>::NV, rbm_type<0>::NV);

        item = item_data;

        auto input = std::cref(item);

        cpp::for_each_i(tuples, [&item, &input, &result](std::size_t I, auto& rbm){
            if(I != layers - 1){
                typedef typename std::remove_reference<decltype(rbm)>::type rbm_t;
                constexpr const auto K = rbm_t::K;
                constexpr const auto NO = rbm_t_no<rbm_t>();

                static hidden_t next_a(K, NO, NO);
                static hidden_t next_s(K, NO, NO);

                propagate(rbm, static_cast<const visible_t&>(input), next_a, next_s);

                input = std::cref(next_a);
            }
        });

        constexpr const auto K = rbm_k<layers - 1>();
        constexpr const auto NO = rbm_no<layers - 1>();

        static hidden_t next_a(K, NO, NO);
        static hidden_t next_s(K, NO, NO);

        auto& last_rbm = layer<layers - 1>();

        propagate(last_rbm, static_cast<const visible_t&>(input), next_a, next_s);

        result = next_a;
    }

    template<typename Sample>
    etl::dyn_vector<weight> activation_probabilities(const Sample& item_data){
        etl::dyn_vector<weight> result(rbm_output<layers - 1>());

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

    template<typename Samples, typename Labels>
    bool svm_train(const Samples& training_data, const Labels& labels, const svm_parameter& parameters = default_svm_parameters()){
        return dll::svm_train(*this, training_data, labels, parameters);
    }

    template<typename Iterator, typename LIterator>
    bool svm_train(Iterator&& first, Iterator&& last, LIterator&& lfirst, LIterator&& llast, const svm_parameter& parameters = default_svm_parameters()){
        return dll::svm_train(*this,
            std::forward<Iterator>(first), std::forward<Iterator>(last),
            std::forward<LIterator>(lfirst), std::forward<LIterator>(llast),
            parameters);
    }

    template<typename Samples, typename Labels>
    bool svm_grid_search(const Samples& training_data, const Labels& labels, std::size_t n_fold = 5, const svm::rbf_grid& g = svm::rbf_grid()){
        return dll::svm_grid_search(*this, training_data, labels, n_fold, g);
    }

    template<typename Iterator, typename LIterator>
    bool svm_grid_search(Iterator&& first, Iterator&& last, LIterator&& lfirst, LIterator&& llast, std::size_t n_fold = 5, const svm::rbf_grid& g = svm::rbf_grid()){
        return dll::svm_grid_search(*this,
            std::forward<Iterator>(first), std::forward<Iterator>(last),
            std::forward<LIterator>(lfirst), std::forward<LIterator>(llast),
            n_fold, g);
    }

    template<typename Sample>
    double svm_predict(const Sample& sample){
        return dll::svm_predict(*this, sample);
    }

    /*}}}*/

#endif //DLL_SVM_SUPPORT

};

} //end of namespace dll

#endif
