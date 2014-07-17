//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*! \file Conjugate Gradient (CG) descent Implementation */

#ifndef DBN_CONJUGATE_GRADIENT_HPP
#define DBN_CONJUGATE_GRADIENT_HPP

namespace dll {

template<typename Target>
struct gradient_context {
    size_t max_iterations;
    size_t epoch;
    batch<vector<double>> inputs;
    batch<Target> targets;
    size_t start_layer;

    gradient_context(batch<vector<double>> i, batch<Target> t, size_t e)
        : max_iterations(5), epoch(e), inputs(i), targets(t), start_layer(0)
    {
        //Nothing else to init
    }
};

template<typename DBN, bool Debug = false>
struct cg_trainer {
    using dbn_t = DBN;
    using weight = typename dbn_t::weight;

    static constexpr const std::size_t layers = dbn_t::layers;

    dbn_t& dbn;
    typename dbn_t::tuple_type& tuples;

    cg_trainer(dbn_t& dbn) : dbn(dbn), tuples(dbn.tuples) {}

    void init_training(std::size_t batch_size){
        detail::for_each(dbn.tuples, [batch_size](auto& rbm){
            typedef typename std::remove_reference<decltype(rbm)>::type rbm_t;
            constexpr const auto num_hidden = rbm_t::num_hidden;

            for(size_t i = 0; i < batch_size; ++i){
                rbm.gr_probs_a.emplace_back(num_hidden);
                rbm.gr_probs_s.emplace_back(num_hidden);
            }
        });
    }

    template<typename T, typename L>
    void train_batch(std::size_t epoch, const dll::batch<T>& data_batch, const dll::batch<L>& label_batch){
        gradient_context<L> context(data_batch, label_batch, epoch);

        minimize(context);
    }

    /* Gradient */

    template<bool Temp, typename R1, typename R2, typename D>
    static void update_diffs(R1& r1, R2& r2, std::vector<D>& diffs, size_t n_samples){
        constexpr auto n_visible = R2::num_visible;
        constexpr auto n_hidden = R2::num_hidden;

        for(size_t sample = 0;  sample < n_samples; ++sample){
            D diff(n_visible);

            for(size_t i = 0; i < n_visible; ++i){
                double s = 0.0;
                for(size_t j = 0; j < n_hidden; ++j){
                    s += diffs[sample][j] * (Temp ? r2.gr_w_tmp(i, j) : r2.gr_w(i, j));
                }

                if(R1::hidden_unit != unit_type::RELU){
                    s *= r1.gr_probs_a[sample][i] * (1.0 - r1.gr_probs_a[sample][i]);
                }

                diff[i] = s;
            }

            diffs[sample].swap(diff);
        }
    }

    template<bool Temp, typename R, typename D, typename V>
    static void update_incs(R& rbm, std::vector<D>& diffs, size_t n_samples, const V& visibles){
        constexpr auto n_visible = R::num_visible;
        constexpr auto n_hidden = R::num_hidden;

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

    template<bool Temp, typename Target>
    void gradient(const gradient_context<Target>& context, weight& cost){
        constexpr const auto n_hidden = dbn_t::template num_hidden<layers - 1>();
        auto n_samples = context.inputs.size();

        static std::vector<std::vector<weight>> diffs;
        diffs.resize(n_samples);

        detail::for_each(tuples, [](auto& rbm){
            rbm.gr_w_incs = 0.0;
            rbm.gr_b_incs = 0.0;
        });

        cost = 0.0;
        weight error = 0.0;

        for(size_t sample = 0; sample < n_samples; ++sample){
            auto& input = context.inputs[sample];
            auto output = std::ref(dbn.layer<0>().gr_probs_a[sample]);
            auto& target = context.targets[sample];

            detail::for_each_i(tuples, [&input,&output,sample](std::size_t I, auto& rbm){
                auto& output_ref = static_cast<vector<weight>&>(output);

                if(I == 0){
                    rbm.template gr_activate_hidden<Temp>(output_ref, rbm.gr_probs_s[sample], input, input);
                } else {
                    rbm.template gr_activate_hidden<Temp>(rbm.gr_probs_a[sample], rbm.gr_probs_s[sample], output_ref, output_ref);
                    output = std::ref(rbm.gr_probs_a[sample]);
                }
            });

            auto& diff = diffs[sample];
            diff.resize(n_hidden);

            auto& result = dbn.layer<layers - 1>().gr_probs_a[sample];
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

        //Get pointers to the different gr_probs
        std::array<std::vector<vector<weight>>*, layers> probs_refs;
        detail::for_each_i(tuples, [&probs_refs](std::size_t I, auto& rbm){
            probs_refs[I] = &rbm.gr_probs_a;
        });

        update_incs<Temp>(dbn.layer<layers-1>(), diffs, n_samples, dbn.layer<layers-2>().gr_probs_a);

        detail::for_each_rpair_i(tuples, [n_samples, &probs_refs](std::size_t I, auto& r1, auto& r2){
            update_diffs<Temp>(r1, r2, diffs, n_samples);

            if(I > 0){
                update_incs<Temp>(r1, diffs, n_samples, *probs_refs[I-1]);
            }
        });

        update_incs<Temp>(dbn.layer<0>(), diffs, n_samples, context.inputs);

        if(Debug){
            std::cout << "evaluating(" << Temp << "): cost:" << cost << " error: " << (error / n_samples) << std::endl;
        }
    }

    bool is_finite(){
        bool finite = true;

        detail::for_each(tuples, [&finite](auto& a){
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

    weight s_dot_s(){
        weight acc = 0.0;
        detail::for_each(tuples, [&acc](auto& rbm){
            acc += dot(rbm.gr_w_s, rbm.gr_w_s) + dot(rbm.gr_b_s, rbm.gr_b_s);
        });
        return acc;
    }

    weight df3_dot_s(){
        weight acc = 0.0;
        detail::for_each(tuples, [&acc](auto& rbm){
            acc += dot(rbm.gr_w_df3, rbm.gr_w_s) + dot(rbm.gr_b_df3, rbm.gr_b_s);
        });
        return acc;
    }

    weight df3_dot_df3(){
        weight acc = 0.0;
        detail::for_each(tuples, [&acc](auto& rbm){
            acc += dot(rbm.gr_w_df3, rbm.gr_w_df3) + dot(rbm.gr_b_df3, rbm.gr_b_df3);
        });
        return acc;
    }

    weight df0_dot_df0(){
        weight acc = 0.0;
        detail::for_each(tuples, [&acc](auto& rbm){
            acc += dot(rbm.gr_w_df0, rbm.gr_w_df0) + dot(rbm.gr_b_df0, rbm.gr_b_df0);
        });
        return acc;
    }

    weight df0_dot_df3(){
        weight acc = 0.0;
        detail::for_each(tuples, [&acc](auto& rbm){
            acc += dot(rbm.gr_w_df0, rbm.gr_w_df3) + dot(rbm.gr_b_df0, rbm.gr_b_df3);
        });
        return acc;
    }

    struct int_t {
        weight f;
        weight d;
        weight x;
    };

    template<typename Target>
    void minimize(const gradient_context<Target>& context){
        constexpr const weight INT = 0.1;       //Don't reevaluate within 0.1 of the limit of the current bracket
        constexpr const weight EXT = 3.0;       //Extrapolate maximum 3 times the current step-size
        constexpr const weight SIG = 0.1;       //Maximum allowed maximum ration between previous and new slopes
        constexpr const weight RHO = SIG / 2.0; //mimimum allowd fraction of the expected
        constexpr const weight RATIO = 10.0;    //Maximum allowed slope ratio
        constexpr const size_t MAX = 20;        //Maximum number of function evaluations per line search

        //Maximum number of try
        auto max_iteration = context.max_iterations;

        weight cost = 0.0;
        gradient<false>(context, cost);

        detail::for_each(tuples, [](auto& rbm){
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

            detail::for_each(tuples, [](auto& rbm){
                rbm.gr_w_best = rbm.gr_w;
                rbm.gr_b_best = rbm.gr_b;

                rbm.gr_w_best_incs = rbm.gr_w_incs;
                rbm.gr_b_best_incs = rbm.gr_b_incs;

                rbm.gr_w_df3 = 0.0;
                rbm.gr_b_df3 = 0.0;
            });

            int64_t M = MAX;

            int_t i1 = {0.0, 0.0, 0.0};
            int_t i2 = {0.0, 0.0, 0.0};

            while(true){
                i2.x = 0.0;
                i2.f = i0.f;
                i2.d = i0.d;
                i3.f = i0.f;

                detail::for_each(tuples, [](auto& rbm){
                    rbm.gr_w_df3 = rbm.gr_w_df0;
                    rbm.gr_b_df3 = rbm.gr_b_df0;
                });

                while(true){
                    if(M-- < 0){
                        break;
                    }

                    detail::for_each(tuples, [&i3](auto& rbm){
                        rbm.gr_w_tmp = rbm.gr_w + rbm.gr_w_s * i3.x;
                        rbm.gr_b_tmp = rbm.gr_b + rbm.gr_b_s * i3.x;
                    });

                    gradient<true>(context, cost);

                    i3.f = cost;
                    detail::for_each(tuples, [](auto& rbm){
                        rbm.gr_w_df3 = rbm.gr_w_incs;
                        rbm.gr_b_df3 = rbm.gr_b_incs;
                    });

                    if(std::isfinite(cost) && is_finite()){
                        if(i3.f < best_cost){
                            best_cost = i3.f;
                            detail::for_each(tuples, [](auto& rbm){
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

                detail::for_each(tuples, [&i3](auto& rbm){
                    rbm.gr_w_tmp = rbm.gr_w + rbm.gr_w_s * i3.x;
                    rbm.gr_b_tmp = rbm.gr_b + rbm.gr_b_s * i3.x;
                });

                gradient<true>(context, cost);

                i3.f = cost;
                detail::for_each(tuples, [](auto& rbm){
                    rbm.gr_w_df3 = rbm.gr_w_incs;
                    rbm.gr_b_df3 = rbm.gr_b_incs;
                });

                if(i3.f < best_cost){
                    best_cost = i3.f;
                    detail::for_each(tuples, [](auto& rbm){
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
                detail::for_each(tuples, [&i3](auto& rbm){
                    rbm.gr_w += rbm.gr_w_s * i3.x;
                    rbm.gr_b += rbm.gr_b_s * i3.x;
                });

                i0.f = i3.f;

                auto g = (df3_dot_df3() - df0_dot_df3()) / df0_dot_df0();

                detail::for_each(tuples, [g](auto& rbm){
                    rbm.gr_w_s = (rbm.gr_w_s * g) + (rbm.gr_w_df3 * -1.0);
                    rbm.gr_b_s = (rbm.gr_b_s * g) + (rbm.gr_b_df3 * -1.0);
                });

                i3.d = i0.d;
                i0.d = df3_dot_s();

                detail::for_each(tuples, [](auto& rbm){
                    rbm.gr_w_df0 = rbm.gr_w_df3;
                    rbm.gr_b_df0 = rbm.gr_b_df3;
                });

                if(i0.d > 0){
                    detail::for_each(tuples, [] (auto& rbm) {
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

                detail::for_each(tuples, [] (auto& rbm) {
                    rbm.gr_w_s = rbm.gr_w_df0 * -1.0;
                    rbm.gr_b_s = rbm.gr_b_df0 * -1.0;
                });
                i0.d = -s_dot_s();

                i3.x = 1.0 / (1.0 - i0.d);

                failed = true;
            }
        }
    }

};

} //end of dbn namespace

#endif