//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_DBN_INL
#define DLL_DBN_INL

#include <tuple>

#include "cpp_utils/tuple_utils.hpp"

#include "unit_type.hpp"
#include "dbn_trainer.hpp"
#include "conjugate_gradient.hpp"
#include "dbn_common.hpp"
#include "svm_common.hpp"

namespace dll {

/*!
 * \brief A Deep Belief Network implementation
 */
template<typename Desc>
struct dbn final {
    using desc = Desc;
    using this_type = dbn<desc>;

    using tuple_type = typename desc::layers::tuple_type;
    tuple_type tuples;

    static constexpr const std::size_t layers = desc::layers::layers;

    template <std::size_t N>
    using rbm_type = typename std::tuple_element<N, tuple_type>::type;

    //TODO Could be good to ensure that either a) all rbm have the same weight b) use the correct type for each rbm
    using weight = typename rbm_type<0>::weight;

    using watcher_t = typename desc::template watcher_t<this_type>;

    weight learning_rate = 0.77;

    weight initial_momentum = 0.5;      ///< The initial momentum
    weight final_momentum = 0.9;        ///< The final momentum applied after *final_momentum_epoch* epoch
    weight final_momentum_epoch = 6;    ///< The epoch at which momentum change

    weight weight_cost = 0.0002;        ///< The weight cost for weight decay

    weight momentum = 0;                ///< The current momentum

#ifdef DLL_SVM_SUPPORT
    svm::model svm_model;               ///< The learned model
    svm::problem problem;               ///< libsvm is stupid, therefore, you cannot destroy the problem if you want to use the model...
    bool svm_loaded = false;            ///< Indicates if a SVM model has been loaded (and therefore must be saved)
#endif //DLL_SVM_SUPPORT

    //No arguments by default
    dbn(){};

    //No copying
    dbn(const dbn& dbn) = delete;
    dbn& operator=(const dbn& dbn) = delete;

    //No moving
    dbn(dbn&& dbn) = delete;
    dbn& operator=(dbn&& dbn) = delete;

    void display() const {
        std::size_t parameters = 0;

        std::cout << "DBN with " << layers << " layers" << std::endl;

        cpp::for_each(tuples, [&parameters](auto& rbm){
            parameters += rbm.parameters();
            rbm.display();
        });

        std::cout << "Total parameters: " << parameters << std::endl;
    }

    void store(const std::string& file) const {
        std::ofstream os(file, std::ofstream::binary);
        store(os);
    }

    void load(const std::string& file){
        std::ifstream is(file, std::ifstream::binary);
        load(is);
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
    static constexpr std::size_t layer_input_size(){
        return rbm_traits<rbm_type<N>>::input_size();
    }

    template<std::size_t N>
    static constexpr std::size_t layer_output_size(){
        return rbm_traits<rbm_type<N>>::output_size();
    }

    static constexpr std::size_t input_size(){
        return layer_input_size<0>();
    }

    static constexpr std::size_t output_size(){
        return layer_output_size<layers - 1>();
    }

    static std::size_t full_output_size(){
        std::size_t output = 0;
        for_each_type<tuple_type>([&output](auto* rbm){
            output += std::decay_t<std::remove_pointer_t<decltype(rbm)>>::output_size();
        });
        return output;
    }

    /*{{{ Pretrain */

    /*!
     * \brief Pretrain the network by training all layers in an unsupervised
     * manner.
     */
    template<typename Samples>
    void pretrain(Samples& training_data, std::size_t max_epochs){
        pretrain(training_data.begin(), training_data.end(), max_epochs);
    }

    /*!
     * \brief Pretrain the network by training all layers in an unsupervised
     * manner.
     */
    template<typename Iterator>
    void pretrain(Iterator&& first, Iterator&& last, std::size_t max_epochs){
        using input_t = typename rbm_type<0>::input_t;
        using output_t = typename rbm_type<0>::output_t;

        watcher_t watcher;

        watcher.pretraining_begin(*this);

        //Convert data to an useful form
        auto data = rbm_type<0>::convert_input(std::forward<Iterator>(first), std::forward<Iterator>(last));

        output_t next_a;
        output_t next_s;

        auto input_ref = std::ref(data);

        cpp::for_each_i(tuples, [&watcher, this, &input_ref, &next_a, &next_s, max_epochs](std::size_t I, auto& rbm){
            using rbm_t = typename std::remove_reference<decltype(rbm)>::type;

            decltype(auto) input = static_cast<input_t&>(input_ref);

            watcher.template pretrain_layer<rbm_t>(*this, I, input.size());

            rbm.template train<
                    input_t,
                    !watcher_t::ignore_sub,                  //Enable the RBM Watcher or not
                    dbn_detail::rbm_watcher_t<watcher_t>>    //Replace the RBM watcher if not void
                (input, max_epochs);

            //Get the activation probabilities for the next level
            if(I < layers - 1){
                rbm.prepare_output(next_a, input.size());
                rbm.prepare_output(next_s, input.size());

                rbm.activate_many(input, next_a, next_s);

                input_ref = std::ref(next_a);
            }
        });

        watcher.pretraining_end(*this);
    }

    /*}}}*/

    /*{{{ With labels */

    //Note: dyn_vector cannot be replaced with fast_vector, because labels is runtime

    template<typename Samples, typename Labels>
    void train_with_labels(Samples& training_data, const Labels& training_labels, std::size_t labels, std::size_t max_epochs){
        cpp_assert(training_data.size() == training_labels.size(), "There must be the same number of values than labels");
        cpp_assert(layer_input_size<layers - 1>() == layer_output_size<layers - 2>() + labels, "There is no room for the labels units");

        train_with_labels(training_data.begin(), training_data.end(), training_labels.begin(), training_labels.end(), labels, max_epochs);
    }

    template<typename Iterator, typename LabelIterator>
    void train_with_labels(Iterator first, Iterator last, LabelIterator lfirst, LabelIterator llast, std::size_t labels, std::size_t max_epochs){
        cpp_assert(std::distance(first, last) == std::distance(lfirst, llast), "There must be the same number of values than labels");
        cpp_assert(layer_input_size<layers - 1>() == layer_output_size<layers - 2>() + labels, "There is no room for the labels units");

        using training_t = std::vector<etl::dyn_vector<weight>>;

        //Convert data to an useful form
        training_t data;
        data.reserve(std::distance(first, last));

        std::for_each(first, last, [&data](auto& sample){
            data.emplace_back(sample);
        });

        auto input = std::ref(data);

        cpp::for_each_i(tuples, [&input, llast, lfirst, labels, max_epochs](size_t I, auto& rbm){
            using rbm_t = typename std::remove_reference<decltype(rbm)>::type;

            constexpr const auto output_size = rbm_traits<rbm_t>::output_size();

            static training_t next;

            next.reserve(static_cast<training_t&>(input).size());

            rbm.train(static_cast<training_t&>(input), max_epochs);

            if(I < layers - 1){
                auto append_labels = (I + 1 == layers - 1);

                for(auto& training_item : static_cast<training_t&>(input)){
                    etl::dyn_vector<weight> next_item_a(output_size + (append_labels ? labels : 0));
                    etl::dyn_vector<weight> next_item_s(output_size + (append_labels ? labels : 0));
                    rbm.activate_hidden(next_item_a, next_item_s, training_item, training_item);
                    next.emplace_back(std::move(next_item_a));
                }

                //If the next layers is the last layer
                if(append_labels){
                    auto it = lfirst;
                    auto end = llast;

                    std::size_t i = 0;
                    while(it != end){
                        auto label = *it;

                        for(size_t l = 0; l < labels; ++l){
                            next[i][output_size + l] = label == l ? 1.0 : 0.0;
                        }

                        ++i;
                        ++it;
                    }
                }
            }

            input = std::ref(next);
        });
    }

    template<typename TrainingItem>
    size_t predict_labels(const TrainingItem& item_data, std::size_t labels){
        cpp_assert(layer_input_size<layers - 1>() == layer_output_size<layers - 2>() + labels, "There is no room for the labels units");

        using training_t = etl::dyn_vector<weight>;

        training_t item(item_data);

        etl::dyn_vector<weight> output_a(layer_input_size<layers - 1>());
        etl::dyn_vector<weight> output_s(layer_input_size<layers - 1>());

        auto input_ref = std::cref(item);

        cpp::for_each_i(tuples, [labels,&input_ref,&output_a,&output_s](size_t I, auto& rbm){
            using rbm_t = typename std::remove_reference<decltype(rbm)>::type;

            constexpr const auto output_size = rbm_traits<rbm_t>::output_size();

            auto& input = static_cast<const training_t&>(input_ref);

            if(I == layers - 1){
                static etl::dyn_vector<weight> h1_a(output_size);
                static etl::dyn_vector<weight> h1_s(output_size);

                rbm.activate_hidden(h1_a, h1_s, input, input);
                rbm.activate_visible(h1_a, h1_s, output_a, output_s);
            } else {
                static etl::dyn_vector<weight> next_a(output_size);
                static etl::dyn_vector<weight> next_s(output_size);

                rbm.activate_hidden(next_a, next_s, input, input);

                //If the next layers is the last layer
                if(I + 1 == layers - 1){
                    static etl::dyn_vector<weight> big_next_a(output_size + labels);

                    std::copy(next_a.begin(), next_a.end(), big_next_a.begin());
                    std::fill(big_next_a.begin() + output_size, big_next_a.end(), 0.1);

                    input_ref = std::cref(big_next_a);
                } else {
                    input_ref = std::cref(next_a);
                }

            }
        });

        size_t label = 0;
        weight max = 0;
        for(size_t l = 0; l < labels; ++l){
            auto value = output_a[layer_input_size<layers - 1>() - labels + l];

            if(value > max){
                max = value;
                label = l;
            }
        }

        return label;
    }

    /*}}}*/

    /*{{{ Predict */

    template<std::size_t I, typename Input, typename Result>
    std::enable_if_t<(I<layers)> activation_probabilities(Input& input, Result& result){
        auto& rbm = layer<I>();

        auto next_s = rbm.prepare_one_output();

        if(I < layers - 1){
            auto next_a = rbm.prepare_one_output();
            rbm.activate_one(input, next_a, next_s);
            activation_probabilities<I+1>(next_a, result);
        } else {
            rbm.activate_one(input, result, next_s);
        }
    }

    //Stop template recursion
    template<std::size_t I, typename Input, typename Result>
    std::enable_if_t<(I==layers)> activation_probabilities(Input&, Result&){}

    template<typename Sample, typename Output>
    void activation_probabilities(const Sample& item_data, Output& result){
        auto data = rbm_type<0>::convert_sample(item_data);

        activation_probabilities<0>(data, result);
    }

    template<typename Sample>
    auto activation_probabilities(const Sample& item_data){
        auto result = layer<layers - 1>().prepare_one_output();

        activation_probabilities(item_data, result);

        return result;
    }

    template<std::size_t I, typename Input, typename Result>
    std::enable_if_t<(I<layers)> full_activation_probabilities(Input& input, std::size_t& i, Result& result){
        auto& rbm = layer<I>();

        auto next_s = rbm.prepare_one_output();
        auto next_a = rbm.prepare_one_output();

        rbm.activate_one(input, next_a, next_s);

        for(auto& value : next_a){
            result[i++] = value;
        }

        full_activation_probabilities<I+1>(next_a, i, result);
    }

    //Stop template recursion
    template<std::size_t I, typename Input, typename Result>
    std::enable_if_t<(I==layers)> full_activation_probabilities(Input&, std::size_t&, Result&){}

    template<typename Sample, typename Output>
    void full_activation_probabilities(const Sample& item_data, Output& result){
        auto data = rbm_type<0>::convert_sample(item_data);

        std::size_t i = 0;

        full_activation_probabilities<0>(data, i, result);
    }

    template<typename Sample>
    etl::dyn_vector<weight> full_activation_probabilities(const Sample& item_data){
        etl::dyn_vector<weight> result(full_output_size());

        full_activation_probabilities(item_data, result);

        return result;
    }

    template<typename Sample, typename DBN = this_type, cpp::enable_if_u<dbn_traits<DBN>::concatenate()> = cpp::detail::dummy>
    auto get_final_activation_probabilities(Sample& sample){
        return full_activation_probabilities(sample);
    }

    template<typename Sample, typename DBN = this_type, cpp::disable_if_u<dbn_traits<DBN>::concatenate()> = cpp::detail::dummy>
    auto get_final_activation_probabilities(Sample& sample){
        return activation_probabilities(sample);
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

    /*{{{ Fine-tuning */

    template<typename Samples, typename Labels>
    weight fine_tune(const Samples& training_data, Labels& labels, size_t max_epochs, size_t batch_size){
        return fine_tune(training_data.begin(), training_data.end(), labels.begin(), labels.end(), max_epochs, batch_size);
    }

    template<typename Iterator, typename LIterator>
    weight fine_tune(Iterator&& first, Iterator&& last, LIterator&& lfirst, LIterator&& llast, size_t max_epochs, size_t batch_size){
        dll::dbn_trainer<this_type> trainer;
        return trainer.train(*this,
            std::forward<Iterator>(first), std::forward<Iterator>(last),
            std::forward<LIterator>(lfirst), std::forward<LIterator>(llast),
            max_epochs, batch_size);
    }

    /*}}}*/

#ifdef DLL_SVM_SUPPORT

    /*{{{ SVM Training and prediction */

    using svm_samples_t = std::conditional_t<
        dbn_traits<this_type>::concatenate(),
        std::vector<etl::dyn_vector<weight>>,     //In full mode, use a simple 1D vector
        typename rbm_type<layers - 1>::output_t>; //In normal mode, use the output of the last layer

    template<typename DBN = this_type, typename Result, typename Sample, cpp::enable_if_u<dbn_traits<DBN>::concatenate()> = cpp::detail::dummy>
    void add_activation_probabilities(Result& result, Sample& sample){
        result.emplace_back(full_output_size());
        full_activation_probabilities(sample, result.back());
    }

    template<typename DBN = this_type, typename Result, typename Sample, cpp::disable_if_u<dbn_traits<DBN>::concatenate()> = cpp::detail::dummy>
    void add_activation_probabilities(Result& result, Sample& sample){
        result.push_back(layer<layers - 1>().prepare_one_output());
        activation_probabilities(sample, result.back());
    }

    template<typename Samples, typename Labels>
    void make_problem(const Samples& training_data, const Labels& labels, bool scale = false){
        svm_samples_t svm_samples;

        //Get all the activation probabilities
        for(auto& sample : training_data){
            add_activation_probabilities(svm_samples, sample);
        }

        //static_cast ensure using the correct overload
        problem = svm::make_problem(labels, static_cast<const svm_samples_t&>(svm_samples), scale);
    }

    template<typename Iterator, typename LIterator>
    void make_problem(Iterator first, Iterator last, LIterator&& lfirst, LIterator&& llast, bool scale = false){
        svm_samples_t svm_samples;

        //Get all the activation probabilities
        std::for_each(first, last, [this, &svm_samples](auto& sample){
            this->add_activation_probabilities(svm_samples, sample);
        });

        //static_cast ensure using the correct overload
        problem = svm::make_problem(
            std::forward<LIterator>(lfirst), std::forward<LIterator>(llast),
            svm_samples.begin(), svm_samples.end(),
            scale);
    }

    template<typename Samples, typename Labels>
    bool svm_train(const Samples& training_data, const Labels& labels, const svm_parameter& parameters = default_svm_parameters()){
        cpp::stop_watch<std::chrono::seconds> watch;

        make_problem(training_data, labels, dbn_traits<this_type>::scale());

        //Make libsvm quiet
        svm::make_quiet();

        //Make sure parameters are not messed up
        if(!svm::check(problem, parameters)){
            return false;
        }

        //Train the SVM
        svm_model = svm::train(problem, parameters);

        svm_loaded = true;

        std::cout << "SVM training took " << watch.elapsed() << "s" << std::endl;

        return true;
    }

    template<typename Iterator, typename LIterator>
    bool svm_train(Iterator&& first, Iterator&& last, LIterator&& lfirst, LIterator&& llast, const svm_parameter& parameters = default_svm_parameters()){
        cpp::stop_watch<std::chrono::seconds> watch;

        make_problem(
            std::forward<Iterator>(first), std::forward<Iterator>(last),
            std::forward<LIterator>(lfirst), std::forward<LIterator>(llast),
            dbn_traits<this_type>::scale());

        //Make libsvm quiet
        svm::make_quiet();

        //Make sure parameters are not messed up
        if(!svm::check(problem, parameters)){
            return false;
        }

        //Train the SVM
        svm_model = svm::train(problem, parameters);

        svm_loaded = true;

        std::cout << "SVM training took " << watch.elapsed() << "s" << std::endl;

        return true;
    }

    template<typename Samples, typename Labels>
    bool svm_grid_search(const Samples& training_data, const Labels& labels, std::size_t n_fold = 5, const svm::rbf_grid& g = svm::rbf_grid()){
        make_problem(training_data, labels, dbn_traits<this_type>::scale());

        //Make libsvm quiet
        svm::make_quiet();

        auto parameters = default_svm_parameters();

        //Make sure parameters are not messed up
        if(!svm::check(problem, parameters)){
            return false;
        }

        //Perform a grid-search
        svm::rbf_grid_search(problem, parameters, n_fold, g);

        return true;
    }

    template<typename It, typename LIt>
    bool svm_grid_search(It&& first, It&& last, LIt&& lfirst, LIt&& llast, std::size_t n_fold = 5, const svm::rbf_grid& g = svm::rbf_grid()){
        make_problem(
            std::forward<It>(first), std::forward<It>(last),
            std::forward<LIt>(lfirst), std::forward<LIt>(llast),
            dbn_traits<this_type>::scale());

        //Make libsvm quiet
        svm::make_quiet();

        auto parameters = default_svm_parameters();

        //Make sure parameters are not messed up
        if(!svm::check(problem, parameters)){
            return false;
        }

        //Perform a grid-search
        svm::rbf_grid_search(problem, parameters, n_fold, g);

        return true;
    }

    template<typename Sample>
    double svm_predict(const Sample& sample){
        auto features = get_final_activation_probabilities(sample);
        return svm::predict(svm_model, features);
    }

    /*}}}*/

#endif //DLL_SVM_SUPPORT

};

} //end of namespace dll

#endif
