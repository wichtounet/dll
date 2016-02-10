//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "neural_base.hpp"

namespace dll {

template<typename Desc>
struct lcn_layer : neural_base<lcn_layer<Desc>> {
    using desc = Desc;

    lcn_layer() = default;

    static std::string to_short_string(){
        return "LCN";
    }

    static void display(){
        std::cout << to_short_string() << std::endl;
    }

    static cpp14_constexpr double gaussian(double x, double y, double sigma = 2.0){
        auto Z = 2.0 * M_PI * sigma * sigma;
        return  1.0 / Z * std::exp(-(x * x + y * y) / (2.0 * sigma * sigma));
    }

    template<typename W>
    static etl::fast_dyn_matrix<W, 9, 9> filter(){
        etl::fast_dyn_matrix<W, 9, 9> w;

        for(std::size_t i = 0; i < 9; ++i){
            for(std::size_t j = 0; j < 9; ++j){
                w(i, j) = gaussian(static_cast<double>(i) - 4.0, static_cast<double>(j) - 4.0);
            }
        }

        w /= etl::sum(w);

        return w;
    }

    template<typename Input, typename Output>
    static void activate_hidden(Output& y, const Input& x){
        using weight_t = etl::value_t<Input>;

        auto w = filter<weight_t>();

        auto v = etl::force_temporary(x(0));
        auto o = etl::force_temporary(x(0));

        for(std::size_t c = 0; c < etl::dim<0>(x); ++c){
            //1. For each pixel, remove mean of 9x9 neighborhood
            for(std::size_t j = 0; j < etl::dim<1>(x); ++j){
                for(std::size_t k = 0; k < etl::dim<2>(x); ++k){
                    weight_t sum(0.0);

                    for (std::size_t p = 0; p < 9; ++p) {
                        if (j + p >= 4 && j + p - 4 < etl::dim<1>(x)) {
                            for (std::size_t q = 0; q < 9; ++q) {
                                if (k + q >= 4 && k + q - 4 < etl::dim<2>(x)) {
                                    sum += w(p, q) * x(c, j + p - 4, k + q - 4);
                                }
                            }
                        }
                    }

                    v(j, k) = x(c, j, k) - sum;
                }
            }

            //2. Scale down norm of 9x9 patch if norm is bigger than 1

            for(std::size_t j = 0; j < etl::dim<1>(x); ++j){
                for(std::size_t k = 0; k < etl::dim<2>(x); ++k){
                    weight_t sum(0.0);

                    for (std::size_t p = 0; p < 9; ++p) {
                        if (j + p >= 4 && j + p - 4 < etl::dim<1>(x)) {
                            for (std::size_t q = 0; q < 9; ++q) {
                                if (k + q >= 4 && k + q - 4 < etl::dim<2>(x)) {
                                    sum += w(p, q) * x(c, j + p - 4, k + q - 4) * x(c, j + p - 4, k + q - 4);
                                }
                            }
                        }
                    }

                    o(j, k) = std::sqrt(sum);
                }
            }

            auto cst = etl::mean(o);
            y(c) = v / etl::max(o, cst);
        }
    }

    template<typename Input, typename Output>
    static void batch_activate_hidden(Output& output, const Input& input){
        for(std::size_t b = 0; b < etl::dim<0>(input); ++b){
            activate_hidden(output(b), input(b));
        }
    }

    template<typename I, typename O_A>
    static void activate_many(const I& input, O_A& h_a){
        for(std::size_t i = 0; i < input.size(); ++i){
            activate_one(input[i], h_a[i]);
        }
    }

    template<typename Input>
    static std::vector<Input> prepare_output(std::size_t samples){
        return std::vector<Input>(samples);
    }

    template<typename Input>
    static Input prepare_one_output(){
        return {};
    }
};

} //end of dll namespace
