//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*! \file Conjugate Gradient (CG) descent Implementation */

#ifndef DLL_CONJUGATE_GRADIENT_HPP
#define DLL_CONJUGATE_GRADIENT_HPP

#include "context.hpp"
#include "batch.hpp"

namespace dll {

template<typename Sample, typename Label>
struct gradient_context {
    std::size_t max_iterations;
    std::size_t epoch;
    batch<Sample> inputs;
    batch<Label> targets;
    std::size_t start_layer;

    gradient_context(batch<Sample> i, batch<Label> t, std::size_t e)
        : max_iterations(5), epoch(e), inputs(i), targets(t), start_layer(0){
        //Nothing else to init
    }
};

template<typename RBM>
struct cg_context {
    using rbm_t = RBM;
    using weight = typename rbm_t::weight;

    static constexpr const std::size_t num_visible = rbm_t::num_visible;
    static constexpr const std::size_t num_hidden = rbm_t::num_hidden;

    etl::fast_matrix<weight, num_visible, num_hidden> gr_w_incs;
    etl::fast_vector<weight, num_hidden> gr_b_incs;

    etl::fast_matrix<weight, num_visible, num_hidden> gr_w_best;
    etl::fast_vector<weight, num_hidden> gr_b_best;

    etl::fast_matrix<weight, num_visible, num_hidden> gr_w_best_incs;
    etl::fast_vector<weight, num_hidden> gr_b_best_incs;

    etl::fast_matrix<weight, num_visible, num_hidden> gr_w_df0;
    etl::fast_vector<weight, num_hidden> gr_b_df0;

    etl::fast_matrix<weight, num_visible, num_hidden> gr_w_df3;
    etl::fast_vector<weight, num_hidden> gr_b_df3;

    etl::fast_matrix<weight, num_visible, num_hidden> gr_w_s;
    etl::fast_vector<weight, num_hidden> gr_b_s;

    etl::fast_matrix<weight, num_visible, num_hidden> gr_w_tmp;
    etl::fast_vector<weight, num_hidden> gr_b_tmp;

    std::vector<etl::dyn_vector<weight>> gr_probs_a;
    std::vector<etl::dyn_vector<weight>> gr_probs_s;
};

template<typename DBN, bool Debug = false>
struct cg_trainer {
    using dbn_t = DBN;
    using weight = typename dbn_t::weight;

    using this_type = cg_trainer<DBN, Debug>;

    using rbm_context_tuple_t = typename context_builder<cg_context, typename dbn_t::tuple_type>::type;

    static constexpr const std::size_t layers = dbn_t::layers;

    dbn_t& dbn;
    typename dbn_t::tuple_type& tuples;

    rbm_context_tuple_t rbm_contexts;

    cg_trainer(dbn_t& dbn) : dbn(dbn), tuples(dbn.tuples) {}

    void init_training(std::size_t batch_size){
        cpp::for_each(rbm_contexts, [batch_size](auto& ctx){
            typedef typename std::remove_reference<decltype(ctx)>::type ctx_t;
            constexpr const auto num_hidden = ctx_t::num_hidden;

            for(std::size_t i = 0; i < batch_size; ++i){
                ctx.gr_probs_a.emplace_back(num_hidden);
                ctx.gr_probs_s.emplace_back(num_hidden);
            }
        });
    }

    template<typename T, typename L>
    void train_batch(std::size_t epoch, const dll::batch<T>& data_batch, const dll::batch<L>& label_batch){
        gradient_context<T, L> context(data_batch, label_batch, epoch);

        minimize(context);
    }

    /* Gradient */

    template<bool Temp, typename R1, typename R2, typename C1, typename C2, typename D>
    static void update_diffs(R1&, R2& r2, C1& c1, C2& c2, std::vector<D>& diffs, std::size_t n_samples){
        constexpr auto n_visible = C2::num_visible;
        constexpr auto n_hidden = C2::num_hidden;

        for(std::size_t sample = 0;  sample < n_samples; ++sample){
            D diff(n_visible);

            for(std::size_t i = 0; i < n_visible; ++i){
                double s = 0.0;
                for(std::size_t j = 0; j < n_hidden; ++j){
                    s += diffs[sample][j] * (Temp ? c2.gr_w_tmp(i, j) : r2.w(i, j));
                }

                if(R1::hidden_unit != unit_type::RELU){
                    s *= c1.gr_probs_a[sample][i] * (1.0 - c1.gr_probs_a[sample][i]);
                }

                diff[i] = s;
            }

            diffs[sample].swap(diff);
        }
    }

    template<bool Temp, typename C, typename D, typename V>
    static void update_incs(C& ctx, std::vector<D>& diffs, const V& visibles){
        constexpr auto n_visible = C::num_visible;
        constexpr auto n_hidden = C::num_hidden;

        auto it = visibles.begin();
        auto end = visibles.end();
        std::size_t sample = 0;

        while(it != end){
            auto& v = *it;
            auto& d = diffs[sample];

            for(std::size_t i = 0; i < n_visible; ++i){
                for(std::size_t j = 0; j < n_hidden; ++j){
                    ctx.gr_w_incs(i, j) += v[i] * d[j];
                }
            }

            for(std::size_t j = 0; j < n_hidden; ++j){
                ctx.gr_b_incs(j) += d[j];
            }

            ++it;
            ++sample;
        }
    }

    template<bool Temp, typename Sample, typename Target>
    void gradient(const gradient_context<Sample, Target>& context, weight& cost){
        constexpr const auto n_hidden = dbn_t::template num_hidden<layers - 1>();
        auto n_samples = context.inputs.size();

        static std::vector<std::vector<weight>> diffs;
        diffs.resize(n_samples);

        cpp::for_each(rbm_contexts, [](auto& ctx){
            ctx.gr_w_incs = 0.0;
            ctx.gr_b_incs = 0.0;
        });

        cost = 0.0;
        weight error = 0.0;

        auto it = context.inputs.begin();
        auto end = context.inputs.end();
        auto tit = context.targets.begin();

        std::size_t sample = 0;

        while(it != end){
            auto& input = *it;
            auto output = std::ref(std::get<0>(rbm_contexts).gr_probs_a[sample]);
            auto& target = *tit;

            cpp::for_each_i(tuples, rbm_contexts, [&input,&output,sample](std::size_t I, auto& rbm, auto& ctx){
                auto& output_ref = static_cast<etl::dyn_vector<weight>&>(output);

                if(I == 0){
                    rbm.activate_hidden(output_ref, ctx.gr_probs_s[sample], input, input, Temp ? ctx.gr_b_tmp : rbm.b, Temp ? ctx.gr_w_tmp : rbm.w);
                } else {
                    rbm.activate_hidden(ctx.gr_probs_a[sample], ctx.gr_probs_s[sample], output_ref, output_ref, Temp ? ctx.gr_b_tmp : rbm.b, Temp ? ctx.gr_w_tmp : rbm.w);
                    output = std::ref(ctx.gr_probs_a[sample]);
                }
            });

            auto& diff = diffs[sample];
            diff.resize(n_hidden);

            auto& result = std::get<layers - 1>(rbm_contexts).gr_probs_a[sample];
            weight scale = std::accumulate(result.begin(), result.end(), 0.0);

            for(auto& r : result){
                r *= (1.0 / scale);
            }

            for(std::size_t i = 0; i < n_hidden; ++i){
                diff[i] = result[i] - target[i];
                cost += target[i] * log(result[i]);
                error += diff[i] * diff[i];
            }

            ++it;
            ++tit;
            ++sample;
        }

        cost = -cost;

        //Get pointers to the different gr_probs
        std::array<std::vector<etl::dyn_vector<weight>>*, layers> probs_refs;
        cpp::for_each_i(rbm_contexts, [&probs_refs](std::size_t I, auto& ctx){
            probs_refs[I] = &ctx.gr_probs_a;
        });

        update_incs<Temp>(std::get<layers-1>(rbm_contexts), diffs, std::get<layers-2>(rbm_contexts).gr_probs_a);

        std::vector<std::vector<weight>>& diffs_p = diffs;

        cpp::for_each_rpair_i(tuples, rbm_contexts, [&diffs_p, n_samples, &probs_refs](std::size_t I, auto& r1, auto& r2, auto& c1, auto& c2){
            this_type::update_diffs<Temp>(r1, r2, c1, c2, diffs_p, n_samples);

            if(I > 0){
                this_type::update_incs<Temp>(c1, diffs_p, *probs_refs[I-1]);
            }
        });

        update_incs<Temp>(std::get<0>(rbm_contexts), diffs, context.inputs);

        if(Debug){
            std::cout << "evaluating(" << Temp << "): cost:" << cost << " error: " << (error / n_samples) << std::endl;
        }
    }

    bool is_finite(){
        bool finite = true;

        cpp::for_each(rbm_contexts, [&finite](auto& a){
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
        cpp::for_each(rbm_contexts, [&acc](auto& ctx){
            acc += dot(ctx.gr_w_s, ctx.gr_w_s) + dot(ctx.gr_b_s, ctx.gr_b_s);
        });
        return acc;
    }

    weight df3_dot_s(){
        weight acc = 0.0;
        cpp::for_each(rbm_contexts, [&acc](auto& ctx){
            acc += dot(ctx.gr_w_df3, ctx.gr_w_s) + dot(ctx.gr_b_df3, ctx.gr_b_s);
        });
        return acc;
    }

    weight df3_dot_df3(){
        weight acc = 0.0;
        cpp::for_each(rbm_contexts, [&acc](auto& ctx){
            acc += dot(ctx.gr_w_df3, ctx.gr_w_df3) + dot(ctx.gr_b_df3, ctx.gr_b_df3);
        });
        return acc;
    }

    weight df0_dot_df0(){
        weight acc = 0.0;
        cpp::for_each(rbm_contexts, [&acc](auto& ctx){
            acc += dot(ctx.gr_w_df0, ctx.gr_w_df0) + dot(ctx.gr_b_df0, ctx.gr_b_df0);
        });
        return acc;
    }

    weight df0_dot_df3(){
        weight acc = 0.0;
        cpp::for_each(rbm_contexts, [&acc](auto& ctx){
            acc += dot(ctx.gr_w_df0, ctx.gr_w_df3) + dot(ctx.gr_b_df0, ctx.gr_b_df3);
        });
        return acc;
    }

    struct int_t {
        weight f;
        weight d;
        weight x;
    };

    template<typename Sample, typename Target>
    void minimize(const gradient_context<Sample, Target>& context){
        constexpr const weight INT = 0.1;       //Don't reevaluate within 0.1 of the limit of the current bracket
        constexpr const weight EXT = 3.0;       //Extrapolate maximum 3 times the current step-size
        constexpr const weight SIG = 0.1;       //Maximum allowed maximum ration between previous and new slopes
        constexpr const weight RHO = SIG / 2.0; //mimimum allowd fraction of the expected
        constexpr const weight RATIO = 10.0;    //Maximum allowed slope ratio
        constexpr const std::size_t MAX = 20;        //Maximum number of function evaluations per line search

        //Maximum number of try
        auto max_iteration = context.max_iterations;

        weight cost = 0.0;
        gradient<false>(context, cost);

        cpp::for_each(rbm_contexts, [](auto& ctx){
            ctx.gr_w_df0 = ctx.gr_w_incs;
            ctx.gr_b_df0 = ctx.gr_b_incs;

            ctx.gr_w_s = ctx.gr_w_df0 * -1.0;
            ctx.gr_b_s = ctx.gr_b_df0 * -1.0;
        });

        int_t i0 = {cost, s_dot_s(), 0.0};
        int_t i3 = {0.0, 0.0, 1.0 / (1 - i0.d)};

        bool failed = false;
        for(std::size_t i = 0; i < max_iteration; ++i){
            auto best_cost = i0.f;
            i3.f = 0.0;

            cpp::for_each(tuples, rbm_contexts, [](auto& rbm, auto& ctx){
                ctx.gr_w_best = rbm.w;
                ctx.gr_b_best = rbm.b;

                ctx.gr_w_best_incs = ctx.gr_w_incs;
                ctx.gr_b_best_incs = ctx.gr_b_incs;

                ctx.gr_w_df3 = 0.0;
                ctx.gr_b_df3 = 0.0;
            });

            int64_t M = MAX;

            int_t i1 = {0.0, 0.0, 0.0};
            int_t i2 = {0.0, 0.0, 0.0};

            while(true){
                i2.x = 0.0;
                i2.f = i0.f;
                i2.d = i0.d;
                i3.f = i0.f;

                cpp::for_each(rbm_contexts, [](auto& ctx){
                    ctx.gr_w_df3 = ctx.gr_w_df0;
                    ctx.gr_b_df3 = ctx.gr_b_df0;
                });

                while(true){
                    if(M-- < 0){
                        break;
                    }

                    cpp::for_each(tuples, rbm_contexts, [&i3](auto& rbm, auto& ctx){
                        ctx.gr_w_tmp = rbm.w + ctx.gr_w_s * i3.x;
                        ctx.gr_b_tmp = rbm.b + ctx.gr_b_s * i3.x;
                    });

                    gradient<true>(context, cost);

                    i3.f = cost;
                    cpp::for_each(rbm_contexts, [](auto& ctx){
                        ctx.gr_w_df3 = ctx.gr_w_incs;
                        ctx.gr_b_df3 = ctx.gr_b_incs;
                    });

                    if(std::isfinite(cost) && is_finite()){
                        if(i3.f < best_cost){
                            best_cost = i3.f;
                            cpp::for_each(rbm_contexts, [](auto& ctx){
                                ctx.gr_w_best = ctx.gr_w_tmp;
                                ctx.gr_b_best = ctx.gr_b_tmp;

                                ctx.gr_w_best_incs = ctx.gr_w_incs;
                                ctx.gr_b_best_incs = ctx.gr_b_incs;
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

                cpp::for_each(tuples, rbm_contexts, [&i3](auto& rbm, auto& ctx){
                    ctx.gr_w_tmp = rbm.w + ctx.gr_w_s * i3.x;
                    ctx.gr_b_tmp = rbm.b + ctx.gr_b_s * i3.x;
                });

                gradient<true>(context, cost);

                i3.f = cost;
                cpp::for_each(rbm_contexts, [](auto& ctx){
                    ctx.gr_w_df3 = ctx.gr_w_incs;
                    ctx.gr_b_df3 = ctx.gr_b_incs;
                });

                if(i3.f < best_cost){
                    best_cost = i3.f;
                    cpp::for_each(rbm_contexts, [](auto& ctx){
                        ctx.gr_w_best = ctx.gr_w_tmp;
                        ctx.gr_b_best = ctx.gr_b_tmp;

                        ctx.gr_w_best_incs = ctx.gr_w_incs;
                        ctx.gr_b_best_incs = ctx.gr_b_incs;
                    });
                }

                --M;

                i3.d = df3_dot_s();
            }

            if(std::abs(i3.d) < -SIG * i0.d && i3.f < i0.f + i3.x * RHO * i0.d){
                cpp::for_each(tuples, rbm_contexts, [&i3](auto& rbm, auto& ctx){
                    rbm.w += ctx.gr_w_s * i3.x;
                    rbm.b += ctx.gr_b_s * i3.x;
                });

                i0.f = i3.f;

                auto g = (df3_dot_df3() - df0_dot_df3()) / df0_dot_df0();

                cpp::for_each(rbm_contexts, [g](auto& ctx){
                    ctx.gr_w_s = (ctx.gr_w_s * g) + (ctx.gr_w_df3 * -1.0);
                    ctx.gr_b_s = (ctx.gr_b_s * g) + (ctx.gr_b_df3 * -1.0);
                });

                i3.d = i0.d;
                i0.d = df3_dot_s();

                cpp::for_each(rbm_contexts, [](auto& ctx){
                    ctx.gr_w_df0 = ctx.gr_w_df3;
                    ctx.gr_b_df0 = ctx.gr_b_df3;
                });

                if(i0.d > 0){
                    cpp::for_each(rbm_contexts, [] (auto& ctx) {
                        ctx.gr_w_s = ctx.gr_w_df0 * -1.0;
                        ctx.gr_b_s = ctx.gr_b_df0 * -1.0;
                    });
                    i0.d = -df0_dot_df0();
                }

                i3.x = i3.x * std::min(RATIO, weight(i3.d / (i0.d - 1e-37)));
                failed = false;
            } else {
                if(failed){
                    break;
                }

                cpp::for_each(rbm_contexts, [] (auto& ctx) {
                    ctx.gr_w_s = ctx.gr_w_df0 * -1.0;
                    ctx.gr_b_s = ctx.gr_b_df0 * -1.0;
                });
                i0.d = -s_dot_s();

                i3.x = 1.0 / (1.0 - i0.d);

                failed = true;
            }
        }
    }

    static std::string name(){
        return "Conjugate Gradient";
    }
};

} //end of dbn namespace

#endif