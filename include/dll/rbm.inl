//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef DLL_RBM_INL
#define DLL_RBM_INL

#include "cpp_utils/assert.hpp"             //Assertions
#include "cpp_utils/stop_watch.hpp"         //Performance counter

#include "etl/etl.hpp"

#include "standard_rbm.hpp"

namespace dll {

/*!
 * \brief Standard version of Restricted Boltzmann Machine
 *
 * This follows the definition of a RBM by Geoffrey Hinton.
 */
template<typename Desc>
struct rbm final : public standard_rbm<rbm<Desc>, Desc> {
    using desc = Desc;
    using weight = typename desc::weight;
    using base_type = standard_rbm<rbm<Desc>, Desc>;

    static constexpr const std::size_t num_visible = desc::num_visible;
    static constexpr const std::size_t num_hidden = desc::num_hidden;

    static constexpr const unit_type visible_unit = desc::visible_unit;
    static constexpr const unit_type hidden_unit = desc::hidden_unit;

    //Weights and biases
    etl::fast_matrix<weight, num_visible, num_hidden> w;    //!< Weights
    etl::fast_vector<weight, num_hidden> b;                 //!< Hidden biases
    etl::fast_vector<weight, num_visible> c;                //!< Visible biases

    //Reconstruction data
    etl::fast_vector<weight, num_visible> v1; //!< State of the visible units

    etl::fast_vector<weight, num_hidden> h1_a; //!< Activation probabilities of hidden units after first CD-step
    etl::fast_vector<weight, num_hidden> h1_s; //!< Sampled value of hidden units after first CD-step

    etl::fast_vector<weight, num_visible> v2_a; //!< Activation probabilities of visible units after first CD-step
    etl::fast_vector<weight, num_visible> v2_s; //!< Sampled value of visible units after first CD-step

    etl::fast_vector<weight, num_hidden> h2_a; //!< Activation probabilities of hidden units after last CD-step
    etl::fast_vector<weight, num_hidden> h2_s; //!< Sampled value of hidden units after last CD-step

    //No copying
    rbm(const rbm& rbm) = delete;
    rbm& operator=(const rbm& rbm) = delete;

    //No moving
    rbm(rbm&& rbm) = delete;
    rbm& operator=(rbm&& rbm) = delete;

    /*!
     * \brief Initialize a RBM with basic weights.
     *
     * The weights are initialized from a normal distribution of
     * zero-mean and 0.1 variance.
     */
    rbm() : standard_rbm<rbm<Desc>, Desc>(), b(0.0), c(0.0) {
        //Initialize the weights with a zero-mean and unit variance Gaussian distribution
        w = etl::normal_generator<weight>() * 0.1;
    }

    static constexpr std::size_t input_size() noexcept {
        return num_visible;
    }

    static constexpr std::size_t output_size() noexcept {
        return num_hidden;
    }

    static constexpr std::size_t parameters() noexcept {
        return num_visible * num_hidden;
    }

    static std::string to_short_string(){
        return "RBM: " + std::to_string(num_visible) + "(" + to_string(visible_unit) + ") -> " + std::to_string(num_hidden) + "(" + to_string(hidden_unit) + ")";
    }

    void display() const {
        std::cout << to_short_string() << std::endl;
    }

    template<bool P = true, bool S = true, typename H1, typename H2, typename V>
    void activate_hidden(H1&& h_a, H2&& h_s, const V& v_a, const V& v_s) const {
        static etl::fast_matrix<weight, 1, num_hidden> t;
        base_type::template std_activate_hidden<P, S>(std::forward<H1>(h_a), std::forward<H2>(h_s), v_a, v_s, b, w, t);
    }

    template<bool P = true, bool S = true, typename H1, typename H2, typename V, typename T>
    void activate_hidden(H1&& h_a, H2&& h_s, const V& v_a, const V& v_s, T&& t) const {
        base_type::template std_activate_hidden<P, S>(std::forward<H1>(h_a), std::forward<H2>(h_s), v_a, v_s, b, w, std::forward<T>(t));
    }

    template<bool P = true, bool S = true, typename H1, typename H2, typename V, typename B, typename W>
    static void activate_hidden(H1&& h_a, H2&& h_s, const V& v_a, const V& v_s, const B& b, const W& w){
        static etl::fast_matrix<weight, 1, num_hidden> t;
        base_type::template std_activate_hidden<P, S>(std::forward<H1>(h_a), std::forward<H2>(h_s), v_a, v_s, b, w, t);
    }

    template<bool P = true, bool S = true, typename H, typename V>
    void activate_visible(const H& h_a, const H& h_s, V&& v_a, V&& v_s) const {
        static etl::fast_matrix<weight, num_visible, 1> t;
        base_type::template std_activate_visible<P, S>(h_a, h_s, std::forward<V>(v_a), std::forward<V>(v_s), c, w, t);
    }

    template<bool P = true, bool S = true, typename H, typename V, typename T>
    void activate_visible(const H& h_a, const H& h_s, V&& v_a, V&& v_s, T&& t) const {
        base_type::template std_activate_visible<P, S>(h_a, h_s, std::forward<V>(v_a), std::forward<V>(v_s), c, w, std::forward<T>(t));
    }

    template<typename Sample, typename Output>
    void activation_probabilities(const Sample& item_data, Output& result){
        etl::dyn_vector<weight> item(item_data);

        static etl::dyn_vector<weight> next_s(num_hidden);

        activate_hidden(result, next_s, item, item);
    }

    template<typename Sample>
    etl::dyn_vector<weight> activation_probabilities(const Sample& item_data){
        etl::dyn_vector<weight> result(output_size());

        activation_probabilities(item_data, result);

        return result;
    }

    //Utilities to be used by DBNs

    using input_one_t = etl::dyn_vector<weight>;
    using output_one_t = etl::dyn_vector<weight>;
    using input_t = std::vector<input_one_t>;
    using output_t = std::vector<output_one_t>;

    template<typename Iterator>
    static auto convert_input(Iterator&& first, Iterator&& last){
        input_t input;
        input.reserve(std::distance(std::forward<Iterator>(first), std::forward<Iterator>(last)));

        std::for_each(std::forward<Iterator>(first), std::forward<Iterator>(last), [&input](auto& sample){
            input.emplace_back(sample);
        });

        return input;
    }

    template<typename Sample>
    static input_one_t convert_sample(const Sample& sample){
        return {sample};
    }

    output_t prepare_output(std::size_t samples, bool is_last = false, std::size_t labels = 0){
        output_t output;
        output.reserve(samples);

        for(std::size_t i = 0; i < samples; ++i){
            output.emplace_back(output_size() + (is_last ? labels : 0));
        }

        return output;
    }

    static output_one_t prepare_one_output(){
        return output_one_t(output_size());
    }

    void activate_one(const input_one_t& input, output_one_t& h_a, output_one_t& h_s) const {
        activate_hidden(h_a, h_s, input, input);
    }

    void activate_many(const input_t& input, output_t& h_a, output_t& h_s) const {
        for(std::size_t i = 0; i < input.size(); ++i){
            activate_one(input[i], h_a[i], h_s[i]);
        }
    }
};

//Allow odr-use of the constexpr static members

template<typename Desc>
const std::size_t rbm<Desc>::num_visible;

template<typename Desc>
const std::size_t rbm<Desc>::num_hidden;

} //end of dll namespace

#endif
