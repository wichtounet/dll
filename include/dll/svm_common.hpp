//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_SVM_COMMON
#define DLL_SVM_COMMON

//SVM Support is optional cause it requires libsvm

#ifdef DLL_SVM_SUPPORT

#include <fstream>

#include "cpp_utils/io.hpp"
#include "nice_svm.hpp"

namespace dll {

inline svm_parameter default_svm_parameters() {
    auto parameters = svm::default_parameters();

    parameters.svm_type    = C_SVC;
    parameters.kernel_type = RBF;
    parameters.probability = 1;
    parameters.C           = 2.8;
    parameters.gamma       = 0.0073;

    return parameters;
}

template <typename DBN>
void svm_store(const DBN& dbn, std::ostream& os) {
    if (dbn.svm_loaded) {
        cpp::binary_write(os, true);

        svm::save(dbn.svm_model, "..tmp.svm");

        std::ifstream svm_is("..tmp.svm", std::ios::binary);

        char buffer[1024];

        while (true) {
            svm_is.read(buffer, 1024);

            if (svm_is.gcount() == 0) {
                break;
            }

            os.write(buffer, svm_is.gcount());
        }
    } else {
        cpp::binary_write(os, false);
    }
}

template <typename DBN>
void svm_load(DBN& dbn, std::istream& is) {
    dbn.svm_loaded = false;

    if (is.good()) {
        bool svm;
        cpp::binary_load(is, svm);

        if (svm) {
            std::ofstream svm_os("..tmp.svm", std::ios::binary);

            char buffer[1024];

            while (true) {
                is.read(buffer, 1024);

                if (is.gcount() == 0) {
                    break;
                }

                svm_os.write(buffer, is.gcount());
            }

            svm_os.close();

            dbn.svm_model = svm::load("..tmp.svm");

            dbn.svm_loaded = true;
        }
    }
}

template <typename DBN, typename Result, typename Sample>
void add_activation_probabilities(DBN& dbn, Result& result, Sample& sample) {
    if constexpr (dbn_traits<std::decay_t<DBN>>::concatenate()) {
        result.emplace_back(dbn_full_output_size(dbn));
        dbn.smart_full_activation_probabilities(sample, result.back());
    } else {
        result.emplace_back(dbn_output_size(dbn));
        dbn.activation_probabilities(sample, result.back());
    }
}

template <typename DBN, typename Sample>
etl::dyn_vector<typename DBN::weight> get_activation_probabilities(DBN& dbn, Sample& sample) {
    if constexpr (dbn_traits<std::decay_t<DBN>>::concatenate()) {
        return dbn.smart_full_activation_probabilities(sample);
    } else {
        return dbn.activation_probabilities(sample);
    }
}

template <typename DBN, typename Samples, typename Labels>
void make_problem(DBN& dbn, const Samples& training_data, const Labels& labels, bool scale = false) {
    using svm_samples_t = std::vector<etl::dyn_vector<typename DBN::weight>>;
    svm_samples_t svm_samples;

    //Get all the activation probabilities
    for (auto& sample : training_data) {
        add_activation_probabilities(dbn, svm_samples, sample);
    }

    //static_cast ensure using the correct overload
    dbn.problem = svm::make_problem(labels, static_cast<const svm_samples_t&>(svm_samples), scale);
}

template <typename DBN, typename Iterator, typename LIterator>
void make_problem(DBN& dbn, Iterator first, Iterator last, LIterator&& lfirst, LIterator&& llast, bool scale = false) {
    std::vector<etl::dyn_vector<typename DBN::weight>> svm_samples;

    //Get all the activation probabilities
    std::for_each(first, last, [&dbn, &svm_samples](auto& sample) {
        add_activation_probabilities(dbn, svm_samples, sample);
    });

    //static_cast ensure using the correct overload
    dbn.problem = svm::make_problem(
        std::forward<LIterator>(lfirst), std::forward<LIterator>(llast),
        svm_samples.begin(), svm_samples.end(),
        scale);
}

template <typename DBN, typename Samples, typename Labels>
bool svm_train(DBN& dbn, const Samples& training_data, const Labels& labels, const svm_parameter& parameters) {
    cpp::stop_watch<std::chrono::seconds> watch;

    make_problem(dbn, training_data, labels, dbn_traits<DBN>::scale());

    //Make libsvm quiet
    svm::make_quiet();

    //Make sure parameters are not messed up
    if (!svm::check(dbn.problem, parameters)) {
        return false;
    }

    //Train the SVM
    dbn.svm_model = svm::train(dbn.problem, parameters);

    dbn.svm_loaded = true;

    std::cout << "SVM training took " << watch.elapsed() << "s" << std::endl;

    return true;
}

template <typename DBN, typename Iterator, typename LIterator>
bool svm_train(DBN& dbn, Iterator&& first, Iterator&& last, LIterator&& lfirst, LIterator&& llast, const svm_parameter& parameters) {
    cpp::stop_watch<std::chrono::seconds> watch;

    make_problem(dbn,
                 std::forward<Iterator>(first), std::forward<Iterator>(last),
                 std::forward<LIterator>(lfirst), std::forward<LIterator>(llast),
                 dbn_traits<DBN>::scale());

    //Make libsvm quiet
    svm::make_quiet();

    //Make sure parameters are not messed up
    if (!svm::check(dbn.problem, parameters)) {
        return false;
    }

    //Train the SVM
    dbn.svm_model = svm::train(dbn.problem, parameters);

    dbn.svm_loaded = true;

    std::cout << "SVM training took " << watch.elapsed() << "s" << std::endl;

    return true;
}

template <typename DBN, typename Samples, typename Labels>
bool svm_grid_search(DBN& dbn, const Samples& training_data, const Labels& labels, size_t n_fold = 5, const svm::rbf_grid& g = svm::rbf_grid()) {
    make_problem(dbn, training_data, labels, dbn_traits<DBN>::scale());

    //Make libsvm quiet
    svm::make_quiet();

    auto parameters = default_svm_parameters();

    //Make sure parameters are not messed up
    if (!svm::check(dbn.problem, parameters)) {
        return false;
    }

    //Perform a grid-search
    svm::rbf_grid_search(dbn.problem, parameters, n_fold, g);

    return true;
}

template <typename DBN, typename Iterator, typename LIterator>
bool svm_grid_search(DBN& dbn, Iterator&& first, Iterator&& last, LIterator&& lfirst, LIterator&& llast, size_t n_fold = 5, const svm::rbf_grid& g = svm::rbf_grid()) {
    make_problem(dbn,
                 std::forward<Iterator>(first), std::forward<Iterator>(last),
                 std::forward<LIterator>(lfirst), std::forward<LIterator>(llast),
                 dbn_traits<DBN>::scale());

    //Make libsvm quiet
    svm::make_quiet();

    auto parameters = default_svm_parameters();

    //Make sure parameters are not messed up
    if (!svm::check(dbn.problem, parameters)) {
        return false;
    }

    //Perform a grid-search
    svm::rbf_grid_search(dbn.problem, parameters, n_fold, g);

    return true;
}

template <typename DBN, typename Sample>
double svm_predict(DBN& dbn, const Sample& sample) {
    auto features = get_activation_probabilities(dbn, sample);
    return svm::predict(dbn.svm_model, features);
}

} // end of namespace dll

#endif //DLL_SVM_SUPPORT

#endif //DLL_SVM_COMMON
