//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace dll {

template <typename Augment>
struct augmenter;

template <std::size_t C>
struct augmenter <copy<C>> {
    template <typename Input, typename Output>
    static void apply(Output& result, const Input& input) {
        for(std::size_t c = 0; c < C; ++c){
            // Simply create a copy
            result.push_back(input);
        }
    }

    static void concat_name(std::string& name) {
        name += " copy<" + std::to_string(C) + ">";
    }
};

/*!
 * \brief Layer to perform data augmentation
 */
template <typename Desc>
struct augment_layer {
    using desc = Desc;

    augment_layer() = default;

    template <typename... Augmenter>
    static void concat_all_names(const cpp::type_list<Augmenter...>&, std::string& name) {
        int wormhole[] = {(augmenter<Augmenter>::concat_name(name), 0)...};
        cpp_unused(wormhole);
    }

    static std::string to_short_string() {
        std::string name = "Augment<";

        concat_all_names(typename desc::parameters(), name);

        name += " >";

        return name;
    }

    static void display() {
        std::cout << to_short_string() << std::endl;
    }

    template <typename... Augmenter, typename Input, typename Output>
    static void apply_all(const cpp::type_list<Augmenter...>&, Output& h_a, const Input& input) {
        int wormhole[] = {(augmenter<Augmenter>::apply(h_a, input), 0)...};
        cpp_unused(wormhole);
    }

    template <typename Input, typename Output>
    static void activate_hidden(Output& h_a, const Input& input) {
        h_a.clear();

        // The original is always kept
        h_a.push_back(input);

        apply_all(typename desc::parameters(), h_a, input);
    }

    template <typename Input, typename Output>
    static void activate_many(Output& h_a, const Input& input) {
        for (std::size_t i = 0; i < input.size(); ++i) {
            activate_one(input[i], h_a[i]);
        }
    }

    template <typename Input>
    static std::vector<std::vector<Input>> prepare_output(std::size_t samples) {
        return std::vector<std::vector<Input>>(samples);
    }

    template <typename Input>
    static std::vector<Input> prepare_one_output() {
        return std::vector<Input>();
    }
};

} //end of dll namespace
