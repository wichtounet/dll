//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <fstream>
#include <iostream>
#include <memory>

#include "mnist_reader.hpp"

namespace {

uint32_t read_header(const std::unique_ptr<char[]>& buffer, size_t position){
    auto header = reinterpret_cast<uint32_t*>(buffer.get());

    auto value = *(header + position);
    return (value << 24) | ((value << 8) & 0x00FF0000) | ((value >> 8) & 0X0000FF00) | (value >> 24);
}

std::vector<std::vector<uint8_t>> read_mnist_image_file(const std::string& path){
    std::ifstream file;
    file.open(path, std::ios::in | std::ios::binary | std::ios::ate);

    if(!file){
        std::cout << "Error opening file" << std::endl;
    } else {
        auto size = file.tellg();
        std::unique_ptr<char[]> buffer(new char[size]);

        //Read the entire file at once
        file.seekg(0, std::ios::beg);
        file.read(buffer.get(), size);
        file.close();

        auto magic = read_header(buffer, 0);

        if(magic != 0x803){
            std::cout << "Invalid magic number, probably not a MNIST file" << std::endl;
        } else {
            auto count = read_header(buffer, 1);
            auto rows = read_header(buffer, 2);
            auto columns = read_header(buffer, 3);

            if(size < count * rows * columns + 16){
                std::cout << "The file is not large enough to hold all the data, probably corrupted" << std::endl;
            } else {
                //Skip the header
                auto image_buffer = buffer.get() + 16;

                std::vector<std::vector<uint8_t>> images;
                images.reserve(count);

                for(size_t i = 0; i < count; ++i){
                    images.emplace_back(rows * columns);

                    for(size_t j = 0; j < rows * columns; ++j){
                        images[i][j] = *image_buffer++;
                    }
                }

                return std::move(images);
            }
        }
    }

    return {};
}

std::vector<uint8_t> read_mnist_label_file(const std::string& path){
    std::ifstream file;
    file.open(path, std::ios::in | std::ios::binary | std::ios::ate);

    if(!file){
        std::cout << "Error opening file" << std::endl;
    } else {
        auto size = file.tellg();
        std::unique_ptr<char[]> buffer(new char[size]);

        //Read the entire file at once
        file.seekg(0, std::ios::beg);
        file.read(buffer.get(), size);
        file.close();

        auto magic = read_header(buffer, 0);

        if(magic != 0x801){
            std::cout << "Invalid magic number, probably not a MNIST file" << std::endl;
        } else {
            auto count = read_header(buffer, 1);

            if(size < count + 8){
                std::cout << "The file is not large enough to hold all the data, probably corrupted" << std::endl;
            } else {
                //Skip the header
                auto label_buffer = buffer.get() + 8;

                std::vector<uint8_t> labels(count);

                for(size_t i = 0; i < count; ++i){
                    labels[i] = *label_buffer++;
                }

                return std::move(labels);
            }
        }
    }

    return {};
}


} //end of anonymous namespace

std::vector<std::vector<uint8_t>> mnist::read_training_images(){
    return read_mnist_image_file("datasets/mnist/train-images-idx3-ubyte");
}

std::vector<std::vector<uint8_t>> mnist::read_test_images(){
    return read_mnist_image_file("datasets/mnist/t10k-images-idx3-ubyte");
}

std::vector<uint8_t> mnist::read_training_labels(){
    return read_mnist_label_file("datasets/mnist/train-labels-idx1-ubyte");
}

std::vector<uint8_t> mnist::read_test_labels(){
    return read_mnist_label_file("datasets/mnist/t10k-labels-idx1-ubyte");
}
