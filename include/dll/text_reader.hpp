//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains reader functions to read from a "text" format
 */

#pragma once

#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <cstdint>
#include <memory>

#include <dirent.h>

#include "cpp_utils/tmp.hpp"
#include "etl/etl_light.hpp"

namespace dll {
namespace text {

template<typename Container, typename Functor>
void read_images(Container& images, const std::string& path, size_t limit, Functor func){
    using Image = typename Container::value_type;

    struct dirent* entry;
    auto dir = opendir(path.c_str());

    while ((entry = readdir(dir))) {
        std::string file_name(entry->d_name);

        if (file_name.size() <= 3 || file_name.find(".dat") != file_name.size() - 4) {
            continue;
        }

        int id = std::atoi(std::string(file_name.begin(), file_name.begin() + file_name.size() - 4).c_str());

        if(!limit || id - 1 < (int) limit){
            std::vector<double> temp;

            std::string full_path(path + "/" + file_name);

            std::ifstream file(full_path);

            size_t lines = 0;
            size_t columns = 0;

            // There is a bug in G++7.1 that causes this false positive
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
            std::string line;
            while (std::getline(file, line)) {
                std::istringstream ss(line);
                std::string value;

                while (std::getline(ss, value, ';')) {
                    auto v = std::atof(value.c_str());
                    temp.push_back(v);

                    if(lines == 0){
                        ++columns;
                    }
                }

                ++lines;
            }
#pragma GCC diagnostic pop

            if((int) images.size() < id){
                images.resize(id);
            }

            images[id - 1] = func(1, lines, columns);

            size_t i = 0;
            for (auto& value : temp) {
                images[id - 1][i++] = static_cast<typename Image::value_type>(value);
            }
        }
    }
}

template<template<typename...> typename  Container = std::vector, typename Label = uint8_t>
void read_labels(Container<Label>& labels, const std::string& path, size_t limit = 0){
    struct dirent* entry;
    auto dir = opendir(path.c_str());

    while ((entry = readdir(dir))) {
        std::string file_name(entry->d_name);

        if (file_name.size() <= 3 || file_name.find(".dat") != file_name.size() - 4) {
            continue;
        }

        int id = std::atoi(std::string(file_name.begin(), file_name.begin() + file_name.size() - 4).c_str());

        if((int) labels.size() < id){
            labels.resize(id);
        }

        std::string full_path(path + "/" + file_name);

        std::ifstream file(full_path);
        int value;
        file >> value;
        labels[id - 1] = static_cast<Label>(value);
    }

    if(limit && labels.size() > limit){
        labels.resize(limit);
    }
}

template<bool Three, typename Container, cpp_enable_iff(etl::all_fast<typename Container::value_type>)>
void read_images_direct(Container& images, const std::string& path, size_t limit){
    read_images(images, path, limit, [](size_t /*c*/, size_t /*h*/, size_t /*w*/){ return typename Container::value_type();});
}

template<bool Three, typename Container, cpp_enable_iff(Three && !etl::all_fast<typename Container::value_type>)>
void read_images_direct(Container& images, const std::string& path, size_t limit){
    read_images(images, path, limit, [](size_t c, size_t h, size_t w){ return typename Container::value_type(c, h, w);});
}

template<bool Three, typename Container, cpp_enable_iff(!Three && !etl::all_fast<typename Container::value_type>)>
void read_images_direct(Container& images, const std::string& path, size_t limit){
    read_images(images, path, limit, [](size_t c, size_t h, size_t w){ return typename Container::value_type(c * h * w);});
}

template<template<typename...> typename Container, typename Image, bool Three>
Container<Image> read_images(const std::string& path, size_t limit){
    Container<Image> images;
    read_images_direct<Three>(images, path, limit);
    return images;
}

template<template<typename...> typename Container = std::vector, typename Label = uint8_t>
Container<Label> read_labels(const std::string& path, size_t limit){
    Container<Label> labels;
    read_labels<Container, Label>(labels, path, limit);
    return labels;
}

} //end of namespace text
} //end of namespace dll
