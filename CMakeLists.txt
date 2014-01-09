cmake_minimum_required(VERSION 2.8.8)

project("rbm")

# Activate warnings and C++11 support
if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS                 "-pipe -pthread -std=c++11")
    set(CMAKE_EXE_LINKER_FLAGS          "")

    set(CMAKE_CXX_FLAGS_DEBUG           "-g -pedantic -Wuninitialized -Wall -Wextra -Wno-long-long -Winit-self -fstack-protector-all")
    set(CMAKE_EXE_LINKER_FLAGS_DEBUG    "")

    set(CMAKE_CXX_FLAGS_RELEASE         "-g -DLOGGING_DISABLE -DNDEBUG -O3 -fomit-frame-pointer -march=native")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE  "")

    set(CMAKE_CXX_FLAGS_COVERAGE        "${CMAKE_CXX_FLAGS} --coverage")
    set(CMAKE_EXE_LINKER_FLAGS_COVERAGE "--coverage")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS           "-std=c++11 -Qunused-arguments")
    set(CMAKE_CXX_FLAGS_DEBUG     "-g -pedantic -Wuninitialized -Wsometimes-uninitialized -Wall -Wextra -Wno-long-long -Winit-self -Wdocumentation")
    set(CMAKE_CXX_FLAGS_RELEASE   "-DLOGGING_DISABLE -DNDEBUG -O3 -march=native")
else()
    MESSAGE(FATAL_ERROR "ERROR: budgetwarrior can only be compiled on gcc and clang for now")
endif()

# The compiler and the tests should go to the bin directory
set(EXECUTABLE_OUTPUT_PATH bin)

# All the headers are in the include directory
include_directories(include)

# Create the executable

add_executable(simple_rbm src/rbm_simple.cpp)
