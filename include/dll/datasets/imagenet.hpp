//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <vector>
#include <unordered_map>
#include <utility>
#include <string>

#include <dirent.h>

// Only for image loading...
#include <opencv2/highgui/highgui.hpp>

namespace dll {

namespace imagenet {

inline void read_files(std::vector<std::pair<size_t, size_t>>& files, std::unordered_map<size_t, float>& label_map, const std::string& file_path){
    files.reserve(1200000);

    struct dirent* entry;
    auto dir = opendir(file_path.c_str());

    while ((entry = readdir(dir))) {
        std::string file_name(entry->d_name);

        if (file_name.find("n") != 0) {
            continue;
        }

        std::string label_name(file_name.begin() + 1, file_name.end());
        size_t label = std::atoi(label_name.c_str());

        auto l = label_map.size();
        label_map[label] = l;

        struct dirent* sub_entry;
        auto sub_dir = opendir((file_path + "/" + file_name).c_str());

        while ((sub_entry = readdir(sub_dir))) {
            std::string image_name(sub_entry->d_name);

            if (image_name.find("n") != 0){
                continue;
            }

            std::string image_number(image_name.begin() + image_name.find('_') + 1, image_name.end() - 5);
            size_t image = std::atoi(image_number.c_str());

            files.emplace_back(label, image);
        }
    }
}

struct image_iterator : std::iterator<
                                     std::input_iterator_tag,
                                     etl::fast_dyn_matrix<float, 3, 256, 256>,
                                     ptrdiff_t,
                                     etl::fast_dyn_matrix<float, 3, 256, 256>*,
                                     etl::fast_dyn_matrix<float, 3, 256, 256>&
                                 > {

    using value_type = etl::fast_dyn_matrix<float, 3, 256, 256>;

    std::string imagenet_path;
    std::shared_ptr<std::vector<std::pair<size_t, size_t>>> files;
    std::shared_ptr<std::unordered_map<size_t, float>> labels;

    size_t index;

    image_iterator(const std::string& imagenet_path, std::shared_ptr<std::vector<std::pair<size_t, size_t>>> files, std::shared_ptr<std::unordered_map<size_t, float>> labels, size_t index) :
        imagenet_path(imagenet_path), files(files), labels(labels), index(index)
    {
        // Nothing else to init
    }

    image_iterator(image_iterator&& rhs) = default;
    image_iterator(const image_iterator& rhs) = default;

    image_iterator& operator=(image_iterator&& rhs) = default;
    image_iterator& operator=(const image_iterator& rhs) = default;

    image_iterator& operator++(){
        ++index;
        return *this;
    }

    // Note: DLL will never call this function because in batch mode, but must
    // still compile
    image_iterator operator++(int){
        cpp_unreachable("Should never be called");

        return *this;
    }

    value_type operator*() {
        auto& image_file = (*files)[index];

        auto label = std::string("/n") + (image_file.first < 10000000 ? "0" : "") + std::to_string(image_file.first);

        auto image_path =
            std::string(imagenet_path) + "/train" + label + label +
            "_" + std::to_string(image_file.second) + ".JPEG";

        auto mat = cv::imread(image_path.c_str(), cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);

        value_type image;

        if (!mat.data || mat.empty()) {
            std::cerr << "ERROR: Failed to read image: " << image_path << std::endl;
            image = 0;
            return image;
        }

        if (mat.cols != 256 || mat.rows != 256) {
            std::cerr << "ERROR: Image of invalid size: " << image_path << std::endl;
            image = 0;
            return image;
        }

        if (cpp_likely(mat.channels() == 3)) {
            for (size_t x = 0; x < 256; ++x) {
                for (size_t y = 0; y < 256; ++y) {
                    auto pixel = mat.at<cv::Vec3b>(y, x);

                    image(0, x, y) = pixel.val[0];
                    image(1, x, y) = pixel.val[1];
                    image(2, x, y) = pixel.val[2];
                }
            }
        } else {
            for (size_t x = 0; x < 256; ++x) {
                for (size_t y = 0; y < 256; ++y) {
                    image(0, x, y) = mat.at<unsigned char>(y, x);
                }
            }

            image(1) = 0;
            image(2) = 0;
        }

        return image;
    }

    bool operator==(const image_iterator& rhs) const {
        return index == rhs.index;
    }

    bool operator!=(const image_iterator& rhs) const {
        return index != rhs.index;
    }
};

struct label_iterator : std::iterator<
                                     std::input_iterator_tag,
                                     float,
                                     ptrdiff_t,
                                     float*,
                                     float&
                                 > {

    std::shared_ptr<std::vector<std::pair<size_t, size_t>>> files;
    std::shared_ptr<std::unordered_map<size_t, float>> labels;

    size_t index;

    label_iterator(std::shared_ptr<std::vector<std::pair<size_t, size_t>>> files, std::shared_ptr<std::unordered_map<size_t, float>> labels, size_t index) :
        files(files), labels(labels), index(index)
    {
        // Nothing else to init
    }

    label_iterator(const label_iterator& rhs) = default;
    label_iterator(label_iterator&& rhs) = default;

    label_iterator& operator=(const label_iterator& rhs) = default;
    label_iterator& operator=(label_iterator&& rhs) = default;

    label_iterator& operator++(){
        ++index;
        return *this;
    }

    label_iterator operator++(int){
        auto it = *this;
        ++index;
        return it;
    }

    float operator*() const {
        return (*labels)[(*files)[index].first];
    }

    bool operator==(const label_iterator& rhs) const {
        return index == rhs.index;
    }

    bool operator!=(const label_iterator& rhs) const {
        return index != rhs.index;
    }
};

} // end of namespace imagenet

/*!
 * \brief Creates a dataset around CIFAR-10
 * \param folder The folder in which the CIFAR-10 files are
 * \param parameters The parameters of the generator
 * \return The CIFAR-10 dataset
 */
template<typename... Parameters>
auto make_imagenet_dataset(const std::string& folder, Parameters&&... /*parameters*/){
    auto train_files = std::make_shared<std::vector<std::pair<size_t, size_t>>>();
    auto labels      = std::make_shared<std::unordered_map<size_t, float>>();

    imagenet::read_files(*train_files, *labels, std::string(folder) + "train");

    // Initial shuffle
    std::random_device rd;
    std::default_random_engine engine(rd());
    std::shuffle(train_files->begin(), train_files->end(), engine);

    // The image iterators
    imagenet::image_iterator iit(folder, train_files, labels, 0);
    imagenet::image_iterator iend(folder, train_files, labels, train_files->size());

    // The label iterators
    imagenet::label_iterator lit(train_files, labels, 0);
    imagenet::label_iterator lend(train_files, labels, train_files->size());

    return make_dataset_holder(
        "imagenet",
        make_generator(iit, iend, lit, lend, train_files->size(), 1000, dll::outmemory_data_generator_desc<Parameters..., dll::categorical>{}),
        make_generator(iit, iend, lit, lend, train_files->size(), 1000, dll::outmemory_data_generator_desc<Parameters..., dll::categorical>{}));
}

} // end of namespace dll
