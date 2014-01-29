cmake_minimum_required(VERSION 2.8.8)

project("rbm")

#Boost is necessary
FIND_PACKAGE(Boost REQUIRED)
INCLUDE_DIRECTORIES(SYSTEM ${Boost_INCLUDE_DIRS})

# Activate warnings and C++11 support
if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
    set(WARNING_FLAGS                   "-pedantic -Wuninitialized -Wall -Wextra -Wshadow -Wno-long-long -Winit-self")
    set(CMAKE_CXX_FLAGS                 "-pipe -pthread -std=c++11 ${WARNING_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS          "")

    set(CMAKE_CXX_FLAGS_DEBUG           "-g -fstack-protector-all")
    set(CMAKE_EXE_LINKER_FLAGS_DEBUG    "")

    set(CMAKE_CXX_FLAGS_RELEASE         "-g -DNDEBUG -O3 -fomit-frame-pointer -march=native")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE  "")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS           "-std=c++11 -Qunused-arguments")
    set(CMAKE_CXX_FLAGS_DEBUG     "-g -pedantic -Wuninitialized -Wsometimes-uninitialized -Wall -Wextra -Wno-long-long -Winit-self -Wdocumentation")
    set(CMAKE_CXX_FLAGS_RELEASE   "-DNDEBUG -O3 -march=native")
else()
    MESSAGE(FATAL_ERROR "ERROR: this project can only be compiled on gcc and clang for now")
endif()

# The compiler and the tests should go to the bin directory
set(EXECUTABLE_OUTPUT_PATH bin)

# All the headers are in the include directory
include_directories(include)

# Create a fake library grouping all object files

file(
    GLOB_RECURSE
    source_files
    src/*.cpp
)

file(GLOB executable_files src/rbm_simple.cpp src/rbm_mnist.cpp src/dbn_mnist.cpp src/fast_vector_test.cpp)
list(REMOVE_ITEM source_files ${executable_files})

add_library(DBN OBJECT ${source_files})

# Create the executable

add_executable(rbm_simple $<TARGET_OBJECTS:DBN> src/rbm_simple.cpp)
add_executable(rbm_mnist $<TARGET_OBJECTS:DBN> src/rbm_mnist.cpp)
add_executable(dbn_mnist $<TARGET_OBJECTS:DBN> src/dbn_mnist.cpp)
add_executable(fast_vector_test src/fast_vector_test.cpp)
