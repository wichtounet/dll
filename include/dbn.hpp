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
    //No arguments by default
    dbn(){};

    //No copying
    dbn(const dbn& dbn) = delete;
    dbn& operator=(const dbn& dbn) = delete;

    //No moving
    dbn(dbn&& dbn) = delete;
    dbn& operator=(dbn&& dbn) = delete;

    template<std::size_t N>
    auto layer() -> typename std::add_lvalue_reference<rbm_type<N>>::type {
        return std::get<N>(tuples);
    }

    template<std::size_t N>
    constexpr auto layer() const -> typename std::add_lvalue_reference<typename std::add_const<rbm_type<N>>::type>::type {
        return std::get<N>(tuples);
    }

    template<std::size_t N>
    static constexpr std::size_t num_visible(){
        return rbm_type<N>::num_visible;
    }

    template<std::size_t N>
    static constexpr std::size_t num_hidden(){
        return rbm_type<N>::num_hidden;
    }

    /*{{{ Pretrain */

    template<std::size_t I, typename TrainingItems>
    inline enable_if_t<(I == layers - 2), void>
    pretrain_rbm_layers(TrainingItems& training_data, std::size_t max_epochs){
        std::cout << "Train layer " << I << std::endl;

        layer<I>().train(training_data, max_epochs);
    }

    template<std::size_t I, typename TrainingItems>
    inline enable_if_t<(I < layers - 2), void>
    pretrain_rbm_layers(TrainingItems& training_data, std::size_t max_epochs){
        static_assert(num_hidden<I>() == num_visible<I+1>(), "Layers should have a common unit size");

        std::cout << "Train layer " << I << std::endl;

        auto& rbm = layer<I>();

        rbm.train(training_data, max_epochs);

        std::vector<fast_vector<weight, num_hidden<I>()>> next(training_data.size());

        for(size_t i = 0; i < training_data.size(); ++i){
            rbm.activate_hidden(next[i], training_data[i]);
        }

        pretrain_rbm_layers<I + 1>(next, max_epochs);
    }

    template<typename TrainingItem>
    void pretrain(std::vector<TrainingItem>& training_data, std::size_t max_epochs){
        pretrain_rbm_layers<0>(training_data, max_epochs);
    }

    /*}}}*/

    /*{{{ Train with labels */

    template<std::size_t I, typename TrainingItems, typename LabelItems>
    inline enable_if_t<(I == layers - 1), void>
    train_rbm_layers_labels(TrainingItems& training_data, std::size_t max_epochs, const LabelItems&, std::size_t){
        std::cout << "Train layer " << I << " with labels " << std::endl;

        std::get<I>(tuples).train(training_data, max_epochs);
    }

    template<std::size_t I, typename TrainingItems, typename LabelItems>
    inline enable_if_t<(I < layers - 1), void>
    train_rbm_layers_labels(TrainingItems& training_data, std::size_t max_epochs, const LabelItems& training_labels, std::size_t labels){
        std::cout << "Train layer " << I << " with labels " << std::endl;

        auto& rbm = layer<I>();

        rbm.train(training_data, max_epochs);

        auto append_labels = (I + 1 == layers - 1);

        std::vector<vector<weight>> next;
        next.reserve(training_data.size());

        for(auto& training_item : training_data){
            vector<weight> next_item(num_hidden<I>() + (append_labels ? labels : 0));
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

        train_rbm_layers_labels<I + 1>(next, max_epochs, training_labels, labels);
    }

    template<typename TrainingItem, typename Label>
    void train_with_labels(std::vector<TrainingItem>& training_data, const std::vector<Label>& training_labels, std::size_t labels, std::size_t max_epochs){
        dbn_assert(training_data.size() == training_labels.size(), "There must be the same number of values than labels");
        dbn_assert(num_visible<layers - 1>() == num_hidden<layers - 2>() + labels, "There is no room for the labels units");

        train_rbm_layers_labels<0>(training_data, max_epochs, training_labels, labels);
    }

    /*}}}*/

    /*{{{ Predict */

    size_t predict(vector<weight>& item){
        vector<weight> result(num_hidden<layers - 1>());

        auto input = std::ref(item);

        //TODO That can probably be solved in a more elegant way
        for_each_i(tuples, [&item, &input, &result](std::size_t I, auto& rbm){
            typedef typename std::remove_reference<decltype(rbm)>::type rbm_t;

            static vector<weight> next(rbm_t::num_hidden);
            auto& output = (I == layers - 1) ? result : next;

            rbm.activate_hidden(output, static_cast<vector<weight>&>(input));

            input = std::ref(next);
        });

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

    /*}}}*/

    /*{{{ Predict with labels */

    template<std::size_t I, typename TrainingItem, typename Output>
    inline enable_if_t<(I == layers - 1), void>
    labels_activate(const TrainingItem& input, size_t, Output& output){
        auto& rbm = layer<I>();

        static vector<weight> h1(num_hidden<I>());
        static vector<weight> hs(num_hidden<I>());

        rbm.activate_hidden(h1, input);
        rbm.activate_visible(rbm_type<I>::bernoulli(h1, hs), output);
    }

    template<std::size_t I, typename TrainingItem, typename Output>
    inline enable_if_t<(I < layers - 1), void>
    labels_activate(const TrainingItem& input, std::size_t labels, Output& output){
        auto& rbm = layer<I>();

        static vector<weight> next(num_visible<I+1>());

        rbm.activate_hidden(next, input);

        //If the next layers is the last layer
        if(I + 1 == layers - 1){
            for(size_t l = 0; l < labels; ++l){
                next[num_hidden<I>() + l] = 0.1;
            }
        }

        labels_activate<I + 1>(next, labels, output);
    }

    template<typename TrainingItem>
    size_t predict_labels(TrainingItem& item, std::size_t labels){
        dbn_assert(num_visible<layers - 1>() == num_hidden<layers - 2>() + labels, "There is no room for the labels units");

        static vector<weight> output(num_visible<layers - 1>());

        labels_activate<0>(item, labels, output);

        size_t label = 0;
        weight max = 0;
        for(size_t l = 0; l < labels; ++l){
            auto value = output[num_visible<layers - 1>() - labels + l];

            if(value > max){
                max = value;
                label = l;
            }
        }

        return label;
    }

    /*}}}*/

    /*{{{ Deep predict with labels */

    template<std::size_t I, typename TrainingItem, typename Output>
    inline enable_if_t<(I == layers - 1), void>
    deep_activate_labels(const TrainingItem& input, size_t, Output& output, std::size_t sampling){
        auto& rbm = layer<I>();

        static vector<weight> v1(num_visible<I>());
        static vector<weight> v2(num_visible<I>());

        for(size_t i = 0; i < input.size(); ++i){
            v1(i) = input[i];
        }

        static vector<weight> h1(num_hidden<I>());
        static vector<weight> h2(num_hidden<I>());
        static vector<weight> hs(num_hidden<I>());

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
    deep_activate_labels(const TrainingItem& input, std::size_t labels, Output& output, std::size_t sampling){
        auto& rbm = layer<I>();

        static vector<weight> next(num_visible<I+1>());

        rbm.activate_hidden(next, input);

        //If the next layers is the last layer
        if(I + 1 == layers - 1){
            for(size_t l = 0; l < labels; ++l){
                next[num_hidden<I>() + l] = 0.1;
            }
        }

        deep_activate_labels<I + 1>(next, labels, output, sampling);
    }

    template<typename TrainingItem>
    size_t deep_predict_labels(TrainingItem& item, std::size_t labels, std::size_t sampling){
        dbn_assert(num_visible<layers - 1>() == num_hidden<layers - 2>() + labels, "There is no room for the labels units");

        vector<weight> output(num_visible<layers - 1>());
        deep_activate_labels<0>(item, labels, output, sampling);

        size_t label = 0;
        weight max = 0;
        for(size_t l = 0; l < labels; ++l){
            auto value = output[num_visible<layers - 1>() - labels + l];

            if(value > max){
                max = value;
                label = l;
            }
        }

        return label;
    }

    /*}}}*/

    /* Gradient */

    template<bool Temp, typename R1, typename R2, typename D>
    void update_diffs(R1& r1, R2& r2, std::vector<D>& diffs, size_t n_samples){
        auto n_visible = R2::num_visible;
        auto n_hidden = R2::num_hidden;

        for(size_t sample = 0;  sample < n_samples; ++sample){
            D diff(n_visible);

            for(size_t i = 0; i < n_visible; ++i){
                double s = 0.0;
                for(size_t j = 0; j < n_hidden; ++j){
                    s += diffs[sample][j] * (Temp ? r2.gr_w_tmp(i, j) : r2.gr_w(i, j));
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
                    rbm.gr_w_incs(i, j) += v[i] * d[j];
                }
            }

            for(size_t j = 0; j < n_hidden; ++j){
                rbm.gr_b_incs(j) += d[j];
            }
        }
    }

    template<std::size_t I, bool Temp, typename Input, typename D>
    inline enable_if_t<(I == layers - 1), void>
    gradient_descent(const Input& inputs, std::vector<D>& diffs, size_t n_samples){
        update_incs<Temp>(layer<I>(), diffs, n_samples, layer<I-1>().gr_probs);

        gradient_descent<I - 1, Temp>(inputs, diffs, n_samples);
    }

    template<std::size_t I, bool Temp, typename Input, typename D>
    inline enable_if_t<(I > 0 && I != layers - 1), void>
    gradient_descent(const Input& inputs, std::vector<D>& diffs, size_t n_samples){
        update_diffs<Temp>(layer<I>(), layer<I+1>(), diffs, n_samples);

        update_incs<Temp>(layer<I>(), diffs, n_samples, layer<I-1>().gr_probs);

        gradient_descent<I -1, Temp>(inputs, diffs, n_samples);
    }

    template<std::size_t I, bool Temp, typename Input, typename D>
    inline enable_if_t<(I == 0), void>
    gradient_descent(const Input& inputs, std::vector<D>& diffs, size_t n_samples){
        update_diffs<Temp>(layer<I>(), layer<I+1>(), diffs, n_samples);

        update_incs<Temp>(layer<I>(), diffs, n_samples, inputs);
    }

    template<bool Temp, typename Input, typename Target>
    void gradient(const gradient_context<Input, Target>& context, weight& cost){
        auto n_hidden = num_hidden<layers - 1>();
        auto n_samples = context.inputs.size();

        std::vector<std::vector<weight>> diffs(n_samples);

        for_each(tuples, [](auto& rbm){
            rbm.gr_w_incs = 0.0;
            rbm.gr_b_incs = 0.0;
        });

        cost = 0.0;
        weight error = 0.0;

        for(size_t sample = 0; sample < n_samples; ++sample){
            auto& input = context.inputs[sample];
            auto output = std::ref(layer<0>().gr_probs[sample]);
            auto& target = context.targets[sample];

            for_each_i(tuples, [&input,&output,sample](std::size_t I, auto& rbm){
                if(I == 0){
                    rbm.template gr_activate_hidden<Temp>(static_cast<vector<weight>&>(output), input);
                } else {
                    auto& next_output = rbm.gr_probs[sample];
                    rbm.template gr_activate_hidden<Temp>(next_output, static_cast<vector<weight>&>(output));
                    output = std::ref(rbm.gr_probs[sample]);
                }
            });

            auto& diff = diffs[sample];
            diff.resize(n_hidden);

            auto& result = layer<layers - 1>().gr_probs[sample];
            weight scale = std::accumulate(result.begin(), result.end(), 0.0);

            for(auto& r : result){
                r *= (1.0 / scale);
            }

            for(size_t i = 0; i < n_hidden; ++i){
                diff[i] = result[i] - target[i];
                cost += target[i] * log(result[i]);
                error += diff[i] * diff[i];
            }
        }

        cost = -cost;

        gradient_descent<layers - 1, Temp>(context.inputs, diffs, n_samples);

        //std::cout << "evaluating(" << Temp << "): cost:" << cost << " error: " << (error / n_samples) << std::endl;
    }

    bool is_finite(){
        bool finite = true;
        for_each(tuples, [&finite](auto& a){
            if(!finite){
                return;
            }

            for(auto value : a.gr_w_incs){
                if(!std::isfinite(value)){
                    finite = false;
                    return;
                }
            }

            for(auto value : a.gr_b_incs){
                if(!std::isfinite(value)){
                    finite = false;
                    return;
                }
            }
        });
        return finite;
    }

    template<typename C1, typename C2>
    static weight dot(const C1& c1, const C2& c2){
        weight d = 0.0;
        for(size_t i = 0; i < c1.size(); ++i){
            d += c1[i] * c2[i];
        }
        return d;
    }

    weight s_dot_s(){
        weight acc = 0.0;
        for_each(tuples, [&acc](auto& rbm){
            acc += dot(rbm.gr_w_s, rbm.gr_w_s) + dot(rbm.gr_b_s, rbm.gr_b_s);
        });
        return acc;
    }

    weight df3_dot_s(){
        weight acc = 0.0;
        for_each(tuples, [&acc](auto& rbm){
            acc += dot(rbm.gr_w_df3, rbm.gr_w_s) + dot(rbm.gr_b_df3, rbm.gr_b_s);
        });
        return acc;
    }

    weight df3_dot_df3(){
        weight acc = 0.0;
        for_each(tuples, [&acc](auto& rbm){
            acc += dot(rbm.gr_w_df3, rbm.gr_w_df3) + dot(rbm.gr_b_df3, rbm.gr_b_df3);
        });
        return acc;
    }

    weight df0_dot_df0(){
        weight acc = 0.0;
        for_each(tuples, [&acc](auto& rbm){
            acc += dot(rbm.gr_w_df0, rbm.gr_w_df0) + dot(rbm.gr_b_df0, rbm.gr_b_df0);
        });
        return acc;
    }

    weight df0_dot_df3(){
        weight acc = 0.0;
        for_each(tuples, [&acc](auto& rbm){
            acc += dot(rbm.gr_w_df0, rbm.gr_w_df3) + dot(rbm.gr_b_df0, rbm.gr_b_df3);
        });
        return acc;
    }

    struct int_t {
        weight f;
        weight d;
        weight x;
    };

    template<typename Input, typename Target>
    void minimize(const gradient_context<Input, Target>& context){
        constexpr const weight INT = 0.1;
        constexpr const weight EXT = 3.0;
        constexpr const weight SIG = 0.1;
        constexpr const weight RHO = SIG / 2.0;
        constexpr const weight RATIO = 10.0;

        auto max_iteration = context.max_iterations;

        weight cost = 0.0;
        gradient<false>(context, cost);

        for_each(tuples, [](auto& rbm){
            rbm.gr_w_df0 = rbm.gr_w_incs;
            rbm.gr_b_df0 = rbm.gr_b_incs;

            rbm.gr_w_s = rbm.gr_w_df0 * -1.0;
            rbm.gr_b_s = rbm.gr_b_df0 * -1.0;
        });

        int_t i0 = {cost, s_dot_s(), 0.0};
        int_t i3 = {0.0, 0.0, 1.0 / (1 - i0.d)};

        bool failed = false;
        for(size_t i = 0; i < max_iteration; ++i){
            auto best_cost = i0.f;
            i3.f = 0.0;

            for_each(tuples, [](auto& rbm){
                rbm.gr_w_best = rbm.gr_w;
                rbm.gr_b_best = rbm.gr_b;

                rbm.gr_w_best_incs = rbm.gr_w_incs;
                rbm.gr_b_best_incs = rbm.gr_b_incs;

                rbm.gr_w_df3 = 0.0;
                rbm.gr_b_df3 = 0.0;
            });

            int64_t M = 20;
            int_t i1 = {0.0, 0.0, 0.0};
            int_t i2 = {0.0, 0.0, 0.0};

            while(true){
                i2.x = 0.0;
                i2.f = i0.f;
                i2.d = i0.d;
                i3.f = i0.f;

                for_each(tuples, [](auto& rbm){
                    rbm.gr_w_df3 = rbm.gr_w_df0;
                    rbm.gr_b_df3 = rbm.gr_b_df0;
                });

                while(true){
                    if(M-- < 0){
                        break;
                    }

                    for_each(tuples, [&i3](auto& rbm){
                        rbm.gr_w_tmp = rbm.gr_w + rbm.gr_w_s * i3.x;
                        rbm.gr_b_tmp = rbm.gr_b + rbm.gr_b_s * i3.x;
                    });

                    gradient<true>(context, cost);

                    i3.f = cost;
                    for_each(tuples, [](auto& rbm){
                        rbm.gr_w_df3 = rbm.gr_w_incs;
                        rbm.gr_b_df3 = rbm.gr_b_incs;
                    });

                    if(std::isfinite(cost) && is_finite()){
                        if(i3.f < best_cost){
                            best_cost = i3.f;
                            for_each(tuples, [](auto& rbm){
                                rbm.gr_w_best = rbm.gr_w_tmp;
                                rbm.gr_b_best = rbm.gr_b_tmp;

                                rbm.gr_w_best_incs = rbm.gr_w_incs;
                                rbm.gr_b_best_incs = rbm.gr_b_incs;
                            });
                        }
                        break;
                    }

                    i3.x = (i2.x + i3.x) / 2.0;
                }

                i3.d = df3_dot_s();
                if(i3.d > SIG * i0.d || i3.f > i0.f + i3.x * RHO * i0.d || M <= 0){
                    break;
                }

                i1 = i2;
                i2 = i3;

                //Cubic extrapolation
                auto dx = i2.x - i1.x;
                auto A = 6.0 * (i1.f - i2.f) + 3.0 * (i2.d + i1.d) * dx;
                auto B = 3.0 * (i2.f - i1.f) - (2.0 * i1.d + i2.d) * dx;
                i3.x = i1.x - i1.d * dx * dx / (B + sqrt(B * B - A * i1.d * dx));

                auto upper = i2.x * EXT;
                auto lower = i2.x + INT * dx;
                if(!std::isfinite(i3.x) || i3.x < 0 || i3.x > upper){
                    i3.x = upper;
                } else if(i3.x < lower){
                    i3.x = lower;
                }
            }

            //Interpolation
            int_t i4 = {0.0, 0.0, 0.0};

            while((std::abs(i3.d) > -SIG * i0.d || i3.f > i0.f + i3.x * RHO * i0.d) && M > 0){
                if(i3.d > 0 || i3.f > i0.f + i3.x * RHO * i0.d){
                    i4 = i3;
                } else {
                    i2 = i3;
                }

                auto dx = i4.x - i2.x;
                if(i4.f > i0.f){
                    i3.x = i2.x - (0.5 * i2.d * dx * dx) / (i4.f - i2.f - i2.d * dx); //Quadratic interpolation
                } else {
                    auto A = 6.0 * (i2.f - i4.f) / dx + 3.0 * (i4.d + i2.d);
                    auto B = 3.0 * (i4.f - i2.f) - (2.0 * i2.d + i4.d) * dx;
                    i3.x = i2.x + (sqrt(B * B - A * i2.d * dx * dx) - B) / A;
                }

                if(!std::isfinite(i3.x)){
                    i3.x = (i2.x + i4.x) / 2.0;
                }

                i3.x = std::max(std::min(i3.x, i4.x - INT * (i4.x - i2.x)), i2.x + INT * (i4.x -i2.x));

                for_each(tuples, [&i3](auto& rbm){
                    rbm.gr_w_tmp = rbm.gr_w + rbm.gr_w_s * i3.x;
                    rbm.gr_b_tmp = rbm.gr_b + rbm.gr_b_s * i3.x;
                });

                gradient<true>(context, cost);

                i3.f = cost;
                for_each(tuples, [](auto& rbm){
                    rbm.gr_w_df3 = rbm.gr_w_incs;
                    rbm.gr_b_df3 = rbm.gr_b_incs;
                });

                if(i3.f < best_cost){
                    best_cost = i3.f;
                    for_each(tuples, [](auto& rbm){
                        rbm.gr_w_best = rbm.gr_w_tmp;
                        rbm.gr_b_best = rbm.gr_b_tmp;

                        rbm.gr_w_best_incs = rbm.gr_w_incs;
                        rbm.gr_b_best_incs = rbm.gr_b_incs;
                    });
                }

                --M;

                i3.d = df3_dot_s();
            }

            if(std::abs(i3.d) < -SIG * i0.d && i3.f < i0.f + i3.x * RHO * i0.d){
                for_each(tuples, [&i3](auto& rbm){
                    rbm.gr_w += rbm.gr_w_s * i3.x;
                    rbm.gr_b += rbm.gr_b_s * i3.x;
                });

                i0.f = i3.f;

                auto g = (df3_dot_df3() - df0_dot_df3()) / df0_dot_df0();

                for_each(tuples, [g](auto& rbm){
                    rbm.gr_w_s = (rbm.gr_w_s * g) + (rbm.gr_w_df3 * -1.0);
                    rbm.gr_b_s = (rbm.gr_b_s * g) + (rbm.gr_b_df3 * -1.0);
                });

                i3.d = i0.d;
                i0.d = df3_dot_s();

                for_each(tuples, [](auto& rbm){
                    rbm.gr_w_df0 = rbm.gr_w_df3;
                    rbm.gr_b_df0 = rbm.gr_b_df3;
                });

                if(i0.d > 0){
                    for_each(tuples, [] (auto& rbm) {
                        rbm.gr_w_s = rbm.gr_w_df0 * -1.0;
                        rbm.gr_b_s = rbm.gr_b_df0 * -1.0;
                    });
                    i0.d = -df0_dot_df0();
                }

                i3.x = i3.x * std::min(RATIO, weight(i3.d / (i0.d - 1e-37)));
                failed = false;
            } else {
                if(failed){
                    break;
                }

                for_each(tuples, [] (auto& rbm) {
                    rbm.gr_w_s = rbm.gr_w_df0 * -1.0;
                    rbm.gr_b_s = rbm.gr_b_df0 * -1.0;
                });
                i0.d = -s_dot_s();

                i3.x = 1.0 / (1.0 - i0.d);

                failed = true;
            }
        }
    }

    template<typename TrainingItem, typename Label>
    void fine_tune(std::vector<TrainingItem>& training_data, std::vector<Label>& labels, size_t epochs, size_t batch_size = rbm_type<0>::BatchSize){
        auto batches = training_data.size() / batch_size;

        for_each(tuples, [batch_size](auto& rbm){
            for(size_t i = 0; i < batch_size; ++i){
                rbm.gr_probs.emplace_back(rbm.n_hiddens());
            }
        });

        for(size_t epoch = 0; epoch < epochs; ++epoch){
            for(size_t i = 0; i < batches; ++i){
                auto start = i * batch_size;
                auto end = start + batch_size;

                gradient_context<TrainingItem, Label> context(
                    batch<TrainingItem>(training_data.begin() + start, training_data.begin() + end),
                    batch<Label>(labels.begin() + start, labels.begin() + end),
                    epoch);

                minimize(context);

                std::cout << "epoch(" << epoch << ") batch:" << i << "/" << batches << std::endl;
            }
        }
    }
};

} //end of namespace dbn

#endif
