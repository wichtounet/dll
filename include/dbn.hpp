//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DBN_DBN_HPP
#define DBN_DBN_HPP

#include <tuple>

#include "rbm.hpp"
#include "vector.hpp"
#include "utils.hpp"

template< bool B, class T = void >
using enable_if_t = typename std::enable_if<B, T>::type;

namespace dbn {

template<typename Input, typename Target>
struct gradient_context {
    size_t max_iterations;
    size_t epoch;
    batch<Input> inputs;
    batch<Target> targets;
    size_t start_layer;

    gradient_context(batch<Input> i, batch<Target> t, size_t e)
        : max_iterations(3), epoch(e), inputs(i), targets(t), start_layer(0)
    {
        //Nothing else to init
    }
};

template<typename... Layers>
struct dbn {
private:
    typedef std::tuple<rbm<Layers>...> tuple_type;
    tuple_type tuples;

    template <std::size_t N>
    using rbm_type = typename std::tuple_element<N, tuple_type>::type;

    static constexpr const std::size_t layers = sizeof...(Layers);

    typedef typename rbm_type<0>::weight weight;

public:
    template<std::size_t N>
    auto layer() -> typename std::add_lvalue_reference<rbm_type<N>>::type {
        return std::get<N>(tuples);
    }

    template<std::size_t N>
    constexpr auto layer() const -> typename std::add_lvalue_reference<typename std::add_const<rbm_type<N>>::type>::type {
        return std::get<N>(tuples);
    }

    template<std::size_t N>
    constexpr std::size_t num_visible() const {
        return rbm_type<N>::num_visible;
    }

    template<std::size_t N>
    constexpr std::size_t num_hidden() const {
        return rbm_type<N>::num_hidden;
    }

    template<std::size_t I, typename TrainingItems, typename LabelItems>
    inline enable_if_t<(I == layers - 1), void>
    train_rbm_layers(TrainingItems& training_data, std::size_t max_epochs, const LabelItems&, std::size_t){
        std::cout << "Train layer " << I << std::endl;

        std::get<I>(tuples).train(training_data, max_epochs);
    }

    template<std::size_t I, typename TrainingItems, typename LabelItems>
    inline enable_if_t<(I < layers - 1), void>
    train_rbm_layers(TrainingItems& training_data, std::size_t max_epochs, const LabelItems& training_labels = {}, std::size_t labels = 0){
        std::cout << "Train layer " << I << std::endl;

        auto& rbm = layer<I>();

        rbm.train(training_data, max_epochs);

        auto append_labels = I + 1 == layers - 1 && !training_labels.empty();

        std::vector<vector<double>> next;
        next.reserve(training_data.size());

        for(auto& training_item : training_data){
            vector<double> next_item(num_hidden<I>() + (append_labels ? labels : 0));
            rbm.activate_hidden(next_item, training_item);
            next.emplace_back(std::move(next_item));
        }

        //If the next layers is the last layer
        if(append_labels){
            for(size_t i = 0; i < training_labels.size(); ++i){
                auto label = training_labels[i];

                for(size_t l = 0; l < labels; ++l){
                    if(label == l){
                        next[i][num_hidden<I>() + l] = 1;
                    } else {
                        next[i][num_hidden<I>() + l] = 0;
                    }
                }
            }
        }

        train_rbm_layers<I + 1>(next, max_epochs, training_labels, labels);
    }

    template<typename TrainingItem>
    void pretrain(std::vector<TrainingItem>& training_data, std::size_t max_epochs){
        train_rbm_layers<0, decltype(training_data), std::vector<uint8_t>>(training_data, max_epochs);
    }

    template<typename TrainingItem, typename Label>
    void pretrain_with_labels(std::vector<TrainingItem>& training_data, const std::vector<Label>& training_labels, std::size_t labels, std::size_t max_epochs){
        dbn_assert(training_data.size() == training_labels.size(), "There must be the same number of values than labels");
        dbn_assert(num_visible<layers - 1>() == num_hidden<layers - 2>() + labels, "There is no room for the labels units");

        train_rbm_layers<0>(training_data, max_epochs, training_labels, labels);
    }

    template<std::size_t I, typename TrainingItem, typename Output>
    inline enable_if_t<(I == layers - 1), void>
    activate_layers(const TrainingItem& input, size_t, Output& output){
        auto& rbm = layer<I>();

        static vector<double> h1(num_hidden<I>());
        static vector<double> hs(num_hidden<I>());

        rbm.activate_hidden(h1, input);
        rbm.activate_visible(rbm_type<I>::bernoulli(h1, hs), output);
    }

    template<std::size_t I, typename TrainingItem, typename Output>
    inline enable_if_t<(I < layers - 1), void>
    activate_layers(const TrainingItem& input, std::size_t labels, Output& output){
        auto& rbm = layer<I>();

        static vector<double> next(num_visible<I+1>());

        rbm.activate_hidden(next, input);

        //If the next layers is the last layer
        if(I + 1 == layers - 1){
            for(size_t l = 0; l < labels; ++l){
                next[num_hidden<I>() + l] = 0.1;
            }
        }

        activate_layers<I + 1>(next, labels, output);
    }

    template<typename TrainingItem>
    size_t predict(TrainingItem& item, std::size_t labels){
        dbn_assert(num_visible<layers - 1>() == num_hidden<layers - 2>() + labels, "There is no room for the labels units");

        static vector<double> output(num_visible<layers - 1>());

        activate_layers<0>(item, labels, output);

        size_t label = 0;
        double max = 0;
        for(size_t l = 0; l < labels; ++l){
            auto value = output[num_visible<layers - 1>() - labels + l];

            if(value > max){
                max = value;
                label = l;
            }
        }

        return label;
    }

    template<std::size_t I, typename TrainingItem, typename Output>
    inline enable_if_t<(I == layers - 1), void>
    deep_activate_layers(const TrainingItem& input, size_t, Output& output, std::size_t sampling){
        auto& rbm = layer<I>();

        static vector<double> v1(num_visible<I>());
        static vector<double> v2(num_visible<I>());

        for(size_t i = 0; i < input.size(); ++i){
            v1(i) = input[i];
        }

        static vector<double> h1(num_hidden<I>());
        static vector<double> h2(num_hidden<I>());
        static vector<double> hs(num_hidden<I>());

        for(size_t i = 0; i< sampling; ++i){
            rbm.activate_hidden(h1, v1);
            rbm.activate_visible(rbm_type<I>::bernoulli(h1, hs), v1);

            //TODO Perhaps we should apply a new bernoulli on v1 ?
        }

        rbm.activate_hidden(h1, input);
        rbm.activate_visible(rbm_type<I>::bernoulli(h1, hs), output);
    }

    template<std::size_t I, typename TrainingItem, typename Output>
    inline enable_if_t<(I < layers - 1), void>
    deep_activate_layers(const TrainingItem& input, std::size_t labels, Output& output, std::size_t sampling){
        auto& rbm = layer<I>();

        static vector<double> next(num_visible<I+1>());

        rbm.activate_hidden(next, input);

        //If the next layers is the last layer
        if(I + 1 == layers - 1){
            for(size_t l = 0; l < labels; ++l){
                next[num_hidden<I>() + l] = 0.1;
            }
        }

        deep_activate_layers<I + 1>(next, labels, output, sampling);
    }

    template<typename TrainingItem>
    size_t deep_predict(TrainingItem& item, std::size_t labels, std::size_t sampling){
        dbn_assert(num_visible<layers - 1>() == num_hidden<layers - 2>() + labels, "There is no room for the labels units");

        vector<double> output(num_visible<layers - 1>());
        deep_activate_layers<0>(item, labels, output, sampling);

        size_t label = 0;
        double max = 0;
        for(size_t l = 0; l < labels; ++l){
            auto value = output[num_visible<layers - 1>() - labels + l];

            if(value > max){
                max = value;
                label = l;
            }
        }

        return label;
    }

    /* Gradient */

    struct clear_weigths_incs {
        template<typename T>
        void operator()(T& a) const {
            a.gr_weights_incs = 0.0;
            a.gr_b_incs = 0.0;
        }
    };

    struct resize_probs {
        size_t size;

        resize_probs(size_t s) : size(s){}

        template<typename T>
        void operator()(T& a) const {
            a.gr_probs.resize(size);
        }
    };

    template<std::size_t I, bool Temp, typename TrainingItem>
    inline enable_if_t<(I == layers - 1), void>
    gr_activate_layers(const TrainingItem&, size_t sample){
        auto& rbm = layer<I>();
        auto& output = rbm.gr_probs[sample];

        rbm.template gr_activate_hidden<Temp>(output, layer<I-1>().gr_probs[sample]);
    }

    template<std::size_t I, bool Temp, typename TrainingItem>
    inline enable_if_t<(I > 0 && I < layers - 1), void>
    gr_activate_layers(const TrainingItem& input, size_t sample){
        auto& rbm = layer<I>();

        auto& output = rbm.gr_probs[sample];

        rbm.template gr_activate_hidden<Temp>(output, layer<I-1>().gr_probs[sample]);
        gr_activate_layers<I + 1, Temp>(input, sample);
    }

    template<std::size_t I, bool Temp, typename TrainingItem>
    inline enable_if_t<(I == 0), void>
    gr_activate_layers(const TrainingItem& input, size_t sample){
        auto& rbm = layer<I>();

        auto& output = rbm.gr_probs[sample];

        rbm.template gr_activate_hidden<Temp>(output, input);
        gr_activate_layers<I + 1, Temp>(input, sample);
    }

    template<bool Temp, typename R1, typename R2, typename D>
    void update_diffs(R1& r1, R2& r2,std::vector<D>& diffs, size_t n_samples){
        auto n_visible = R2::num_visible;
        auto n_hidden = R2::num_hidden;

        for(size_t sample = 0;  sample < n_samples; ++sample){
            D diff(n_visible);

            for(size_t i = 0; i < n_visible; ++i){
                float s = 0.0;
                for(size_t j = 0; j < n_hidden; ++j){
                    s+= diffs[sample][j] * (Temp ? r2.gr_weights_tmp(i, j) : r2.gr_weights(i, j));
                }
                s *= r1.gr_probs[sample][i] * (1.0 - r1.gr_probs[sample][i]);
                diff[i] = s;
            }
            diffs[sample].swap(diff);
        }
    }

    template<bool Temp, typename R, typename D, typename V>
    void update_incs(R& rbm, std::vector<D>& diffs, size_t n_samples, const V& visibles){
        auto n_visible = R::num_visible;
        auto n_hidden = R::num_hidden;

        for(size_t sample = 0;  sample < n_samples; ++sample){
            auto& v = visibles[sample];
            auto& d = diffs[sample];

            for(size_t i = 0; i < n_visible; ++i){
                for(size_t j = 0; j < n_hidden; ++j){
                    rbm.gr_weights_incs(i, j) += v[i] * d[j];
                }
            }

            for(size_t j = 0; j < n_hidden; ++j){
                rbm.gr_b_incs(j) += d[j];
            }
        }
    }

    template<std::size_t I, bool Temp, typename Input, typename D>
    inline enable_if_t<(I == layers - 1), void>
    gradient_descent(Input& inputs, std::vector<D>& diffs, size_t n_samples){
        update_incs<Temp>(layer<I>(), diffs, n_samples, layer<I-1>().gr_probs);

        gradient_descent<I - 1, Temp>(inputs, diffs, n_samples);
    }

    template<std::size_t I, bool Temp, typename Input, typename D>
    inline enable_if_t<(I > 0 && I != layers - 1), void>
    gradient_descent(Input& inputs, std::vector<D>& diffs, size_t n_samples){
        update_diffs<Temp>(layer<I>(), layer<I+1>(), diffs, n_samples);

        update_incs<Temp>(layer<I>(), diffs, n_samples, layer<I-1>().gr_probs);

        gradient_descent<I -1, Temp>(inputs, diffs, n_samples);
    }

    template<std::size_t I, bool Temp, typename Input, typename D>
    inline enable_if_t<(I == 0), void>
    gradient_descent(Input& inputs, std::vector<D>& diffs, size_t n_samples){
        update_diffs<Temp>(layer<I>(), layer<I+1>(), diffs, n_samples);

        update_incs<Temp>(layer<I>(), diffs, n_samples, inputs);
    }

    template<bool Temp, typename Input, typename Target>
    void gradient(const gradient_context<Input, Target>& context, weight& cost){
        auto n_samples = context.inputs.size();
        auto n_hidden = num_hidden<layers - 1>();

        for_each(tuples, clear_weigths_incs());
        std::vector<std::vector<double>> diffs(n_samples);

        cost = 0.0;
        double error = 0.0;

        for(size_t sample = 0; sample < n_samples; ++sample){
            auto& input = context.inputs[sample];

            gr_activate_layers<0, Temp>(input, sample);

            auto& result = layer<layers - 1>().gr_probs[sample];
            auto& diff = diffs[sample];
            diff.resize(n_hidden);

            double scale = std::accumulate(result.begin(), result.end(), 0.0);
            result *= (1.0 / scale);

            auto& target = context.targets[sample];
            for(size_t i = 0; i < n_hidden; ++i){
                diff[i] = (result[i] - target[i]);
                cost += target[i] * log(result[i]);
                error += diff[i] * diff[i];
            }
        }

        cost = -cost;

        gradient_descent<layers - 1, Temp>(context.inputs, diffs, n_samples);

        std::cout << "evaluating: cost:" << cost << " error: " << (error / n_samples) << std::endl;
    }

    struct gr_init_weights {
        template<typename T>
        void operator()(T& a) const {
            a.gr_weights = a.w;
            a.gr_b = a.b;
        }
    };

    struct gr_copy_init {
        template<typename T>
        void operator()(T& a) const {
            a.gr_weights_s = a.gr_weights_df0 = a.gr_weights_incs;
            a.gr_b_s = a.gr_b_df0 = a.gr_b_incs;

            a.gr_b_s *= -1.0;
            a.gr_weights_s *= -1.0;
        }
    };

    struct gr_copy_best {
        template<typename T>
        void operator()(T& a) const {
            a.gr_weights_best = a.gr_weights;
            a.gr_weights_best_incs = a.gr_weights_incs;

            a.gr_b_best = a.gr_b;
            a.gr_b_best_incs = a.gr_b_incs;

            a.gr_weights_df3 = 0.0;
            a.gr_b_df3 = 0.0;
        }
    };

    struct gr_df0_to_df3 {
        template<typename T>
        void operator()(T& a) const {
            a.gr_weights_df3 = a.gr_weights_df0;
            a.gr_b_df3 = a.gr_b_df0;
        }
    };

    struct gr_minus_df0_to_s {
        template<typename T>
        void operator()(T& a) const {
            a.gr_weights_s = a.gr_weights_df0 * -1.0;
            a.gr_b_s = a.gr_b_df0 * -1.0;
        }
    };

    struct gr_gs_minus_df3 {
        double g;

        gr_gs_minus_df3(double g1) : g(g1){}

        template<typename T>
        void operator()(T& a) const {
            a.gr_weights_s = (a.gr_weights_s * g) + (a.gr_weights_df3 * -1.0);
            a.gr_b_s = (a.gr_b_s * g) + (a.gr_b_df3 * -1.0);
        }
    };

    struct gr_df3_to_df0 {
        template<typename T>
        void operator()(T& a) const {
            a.gr_weights_df0 = a.gr_weights_df3;
            a.gr_b_df0 = a.gr_b_df3;
        }
    };

    struct gr_w_incs_to_df3 {
        template<typename T>
        void operator()(T& a) const {
            a.gr_weights_df3 = a.gr_weights_incs;
            a.gr_b_df3 = a.gr_b_incs;
        }
    };

    struct gr_tmp_to_best {
        template<typename T>
        void operator()(T& a) const {
            a.gr_weights_best = a.gr_weights_tmp;
            a.gr_b_best = a.gr_b_tmp;

            a.gr_weights_best_incs = a.gr_weights_incs;
            a.gr_b_best_incs = a.gr_b_incs;
        }
    };

    struct gr_apply_weights {
        template<typename T>
        void operator()(T& a) const {
            a.w = a.gr_weights_best;
            a.b = a.gr_b_best;
        }
    };

    struct gr_save_tmp {
        double x3 = 0.0;

        gr_save_tmp(double x) : x3(x){}

        template<typename T>
        void operator()(T& a) const {
            a.gr_weights_tmp = a.gr_weights + a.gr_weights_s * x3;
            a.gr_b_tmp = a.gr_b + a.gr_b_s * x3;
        }
    };

    struct gr_add_sx3_to_weights {
        double x3 = 0.0;

        gr_add_sx3_to_weights(double x) : x3(x) {}

        template<typename T>
        void operator()(T& a) const {
            a.gr_weights += a.gr_weights_s * x3;
            a.gr_b += a.gr_b_s * x3;
        }
    };

    struct gr_check_finite {
        bool finite = true;

        template<typename T>
        void operator()(T& a){
            if(!finite){
                return;
            }

            for(auto value : a.gr_weights_incs){
                if(!std::isfinite(value)){
                    finite = false;
                    return;
                }
            }

            for(auto value : a.gr_b_incs){
                if(!std::isfinite(value)){
                    finite = false;
                    break;
                }
            }
        }
    };

    bool is_finite(){
        gr_check_finite check;
        for_each(tuples, check);
        return check.finite;
    }

    template<typename C1, typename C2>
    static double dot(const C1& c1, const C2& c2){
        double d = 0.0;
        for(size_t i = 0; i < c1.size(); ++i){
            d += c1[i] * c2[i];
        }
        return d;
    }

    struct gr_s_dot_s {
        double acc = 0.0;

        template<typename T>
        void operator()(T& a){
            acc += dot(a.gr_weights_s, a.gr_weights_s) + dot(a.gr_b_s, a.gr_b_s);
        }
    };

    struct gr_df3_dot_s {
        double acc = 0.0;

        template<typename T>
        void operator()(T& a){
            acc += dot(a.gr_weights_df3, a.gr_weights_s) + dot(a.gr_b_df3, a.gr_b_s);
        }
    };

    struct gr_df3_dot_df3 {
        double acc = 0.0;

        template<typename T>
        void operator()(T& a){
            acc += dot(a.gr_weights_df3, a.gr_weights_df3) + dot(a.gr_b_df3, a.gr_b_df3);
        }
    };

    struct gr_df0_dot_df0 {
        double acc = 0.0;

        template<typename T>
        void operator()(T& a){
            acc += dot(a.gr_weights_df0, a.gr_weights_df0) + dot(a.gr_b_df0, a.gr_b_df0);
        }
    };

    struct gr_df0_dot_df3 {
        double acc = 0.0;

        template<typename T>
        void operator()(T& a){
            acc += dot(a.gr_weights_df0, a.gr_weights_df3) + dot(a.gr_b_df0, a.gr_b_df3);
        }
    };

    double s_dot_s(){
        gr_s_dot_s f;
        for_each(tuples, f);
        return f.acc;
    }

    double df3_dot_s(){
        gr_df3_dot_s f;
        for_each(tuples, f);
        return f.acc;
    }

    double df3_dot_df3(){
        gr_df3_dot_df3 f;
        for_each(tuples, f);
        return f.acc;
    }

    double df0_dot_df0(){
        gr_df0_dot_df0 f;
        for_each(tuples, f);
        return f.acc;
    }

    double df0_dot_df3(){
        gr_df0_dot_df3 f;
        for_each(tuples, f);
        return f.acc;
    }

    template<typename Input, typename Target>
    void minimize(const gradient_context<Input, Target>& context){
        constexpr const double INT = 0.1;
        constexpr const double EXT = 3.0;
        constexpr const double SIG = 0.1;
        constexpr const double RHO = SIG / 2.0;
        constexpr const double RATIO = 10.0;

        for_each(tuples, gr_init_weights());

        auto max_iteration = context.max_iterations;

        double cost = 0.0;
        gradient<false>(context, cost);

        for_each(tuples, gr_copy_init());

        auto d0 = s_dot_s();
        auto f0 = cost;
        double d3 = 0.0;
        double x3 = 1.0 / (1 - d0);

        bool failed = false;
        for(size_t i = 0; i < max_iteration; ++i){
            auto best_cost = f0;
            double f3 = 0.0;

            for_each(tuples, gr_copy_best());

            int64_t M = 20;
            double f1 = 0.0;
            double x1 = 0.0;
            double d1 = 0.0;
            double f2 = 0.0;
            double x2 = 0.0;
            double d2 = 0.0;

            while(true){
                x2 = 0.0;
                f2 = f0;
                d2 = d0;
                f3 = f0;

                for_each(tuples, gr_df0_to_df3());

                while(true){
                    if(M-- < 0){
                        break;
                    }

                    //tmp_weights = weights + s * x3;
                    for_each(tuples, gr_save_tmp(x3));

                    gradient<true>(context, cost);

                    f3 = cost;
                    for_each(tuples, gr_w_incs_to_df3());

                    if(std::isfinite(cost) && is_finite()){
                        if(f3 < best_cost){
                            best_cost = f3;
                            for_each(tuples, gr_tmp_to_best());
                        }
                        break;
                    }

                    x3 = (x2 + x3) / 2.0;
                }

                x3 = df3_dot_s();
                if(d3 > SIG * d0 || f3 > f0 + x3 * RHO * d0 || M <= 0){
                    break;
                }

                x1 = x2;
                f1 = f2;
                d1 = d2;
                x2 = x3;
                f2 = f3;
                d2 = d3;

                //Cubic extrapolation
                auto dx = x2 - x1;
                auto A = 6.0 * (f1 - f2) + 3.0 * (d2 + d1) * dx;
                auto B = 3.0 * (f2 - f1) - (2.0 * d1 + d2) * dx;
                x3 = x1 - d1 * dx * dx / (B + sqrt(B * B - A * d1 * dx));

                auto upper = x2 * EXT;
                auto lower = x2 + INT * dx;
                if(!std::isfinite(x3) || x3 < 0 || x3 > upper){
                    x3 = upper;
                } else if(x3 < lower){
                    x3 = lower;
                }
            }

            //Interpolation
            double f4 = 0.0;
            double x4 = 0.0;
            double d4 = 0.0;

            while((std::abs(d3) > -SIG * d0 || f3 > f0 + x3 * RHO * d0) && M > 0){
                if(d3 > 0 || f3 > f0 + x3 * RHO * d0){
                    x4 = x3;
                    f4 = f3;
                    d4 = d3;
                } else {
                    x2 = x3;
                    f2 = f3;
                    d2 = d3;
                }

                auto dx = x4 - x2;
                if(f4 > f0){
                    x3 = x2 - (0.5 * d2 * dx * dx) / (f4 - f2 - d2 * dx); //Quadratic interpolation
                } else {
                    auto A = 6.0 * (f2 - f4) / dx + 3.0 * (d4 + d2);
                    auto B = 3.0 * (f4 - f2) - (2.0 * d2 + d4) * dx;
                    x3 = x2 + (sqrt(B * B - A * d2 * dx * dx) - B) / A;
                }

                if(!std::isfinite(x3)){
                    x3 = (x2 + x4) / 2.0;
                }

                x3 = std::max(std::min(x3, x4 - INT * (x4 - x2)), x2 + INT * (x4 -x2));

                //tmp_weights = weights + s * x3;
                for_each(tuples, gr_save_tmp(x3));

                gradient<true>(context, cost);

                f3 = cost;
                for_each(tuples, gr_w_incs_to_df3());

                if(f3 < best_cost){
                    best_cost = f3;
                    for_each(tuples, gr_tmp_to_best());
                }

                --M;

                d3 = df3_dot_s();
            }

            if(std::abs(d3) < -SIG * d0 && f3 < f0 + x3 * RHO * d0){
                for_each(tuples, gr_add_sx3_to_weights(x3));

                f0 = f3;

                auto g = (df3_dot_df3() - df0_dot_df3()) / df0_dot_df0();
                for_each(tuples, gr_gs_minus_df3(g));

                d3 = d0;
                d0 = df3_dot_s();
                for_each(tuples, gr_df3_to_df0());

                if(d0 > 0){
                    for_each(tuples, gr_minus_df0_to_s());
                    d0 = -df0_dot_df0();
                }

                x3 = x3 * std::min(RATIO, double(d3 / (d0 - 1e-37)));
                failed = false;
                std::cout << "Found iteration i" <<i << ", cost =" << f3 << std::endl;
            } else {
                std::cout << "x3 = " << x3 << " failed" << std::endl;

                if(failed){
                    break;
                }

                for_each(tuples, gr_minus_df0_to_s());
                d0 = -s_dot_s();

                x3 = 1.0 / (1.0 - d0);

                failed = true;
            }
        }

        std::cout << "Apply new weights to RBMs" << std::endl;

        for_each(tuples, gr_apply_weights());
    }

    template<typename TrainingItem, typename Label>
    void fine_tune(std::vector<TrainingItem>& training_data, std::vector<Label>& labels, size_t epochs, size_t batch_size = rbm_type<0>::BatchSize){
        auto batches = training_data.size() / batch_size;
        batches = 100;

        for_each(tuples, resize_probs(batch_size));

        for(size_t epoch = 0; epoch < epochs; ++epoch){
            for(size_t i = 0; i < batches; ++i){
                auto start = i * batch_size;
                auto end = (i+1) * batch_size;

                gradient_context<TrainingItem, Label> context(
                    batch<TrainingItem>(training_data.begin() + start, training_data.begin() + end),
                    batch<Label>(labels.begin() + start, labels.begin() + end),
                    epoch);

                minimize(context);
            }
        }
    }
};

} //end of namespace dbn

#endif
