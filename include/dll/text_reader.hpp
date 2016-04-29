//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains reader functions to read from a "text" format
 */

#pragma once

#include <fstream>
#include <iostream>
#include <vector>
#include <cstdint>
#include <memory>

#include <dirent.h>

namespace dll {
namespace text {

template<template<typename...> class Container = std::vector, typename Image, typename Functor>
void read_images(Container<Image>& images, const std::string& path, std::size_t limit, Functor func){
    struct dirent* entry;
    auto dir = opendir(path.c_str());

    while ((entry = readdir(dir))) {
        std::string file_name(entry->d_name);

        if (file_name.size() <= 3 || file_name.find(".dat") != file_name.size() - 4) {
            continue;
        }

        int id = std::atoi(std::string(file_name.begin(), file_name.begin() + file_name.size() - 4).c_str());

        if(id - 1 < (int) limit){
            if((int) images.size() < id){
                images.resize(id);
            }

            images[id - 1] = func();

            std::string full_path(path + "/" + file_name);

            std::ifstream file(full_path);

            std::size_t i = 0;

            std::string line;
            while (std::getline(file, line)) {
                std::istringstream ss(line);
                std::string value;

                while (std::getline(ss, value, ';')) {
                    auto v = std::atof(value.c_str());
                    images[id - 1][i++] = static_cast<typename Image::value_type>(v);
                }
            }
        }
    }
}

template<template<typename...> class  Container = std::vector, typename Label = uint8_t>
void read_labels(Container<Label>& labels, const std::string& path, std::size_t limit = 0){
    struct dirent* entry;
    auto dir = opendir(path.c_str());

    while ((entry = readdir(dir))) {
        std::string file_name(entry->d_name);

        if (file_name.size() <= 3 || file_name.find(".dat") != file_name.size() - 4) {
            continue;
        }

        int id = std::atoi(std::string(file_name.begin(), file_name.begin() + file_name.size() - 4).c_str());

        if(id - 1 < (int) limit){
            if((int) labels.size() < id){
                labels.resize(id);
            }

            std::string full_path(path + "/" + file_name);

            std::ifstream file(full_path);
            int value;
            file >> value;
            labels[id - 1] = static_cast<Label>(value);
        }
    }
}

template<template<typename...> class Container = std::vector, typename Image, typename Functor>
Container<Image> read_images(const std::string& path, std::size_t limit, Functor func){
    Container<Image> images;
    read_images<Container, Image>(images, path, limit, func);
    return images;
}

template<template<typename...> class Container = std::vector, typename Label = uint8_t>
Container<Label> read_labels(const std::string& path, std::size_t limit){
    Container<Label> labels;
    read_labels<Container, Label>(labels, path, limit);
    return labels;
}

} //end of namespace text
} //end of namespace dll
