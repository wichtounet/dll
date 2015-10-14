//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <deque>

#include "cpp_utils/string.hpp"

#include "dll_test.hpp"
#include "dll/processor/processor.hpp"

namespace {

bool starts_with(const std::string& str, const std::string& search){
    return std::mismatch(search.begin(), search.end(), str.begin()).first == search.end();
}

std::string extract_value(const std::string& str, const std::string& search){
    return {str.begin() + str.find(search) + search.size(), str.end()};
}

bool get_error(const std::vector<std::string>& lines, double& error, const char* begin){
    for(auto& line : lines){
        if(starts_with(line, begin)){
            error = std::stod(extract_value(line, begin));
            return true;
        }
    }

    return false;
}

bool get_ft_error(const std::vector<std::string>& lines, double& error){
    return get_error(lines, error, "Test Classification Error:");
}

bool get_test_error(const std::vector<std::string>& lines, double& error){
    return get_error(lines, error, "Error rate: ");
}

bool get_last_rec_error(std::string epoch, const std::vector<std::string>& lines, double& error){
    bool found = false;
    auto begin = epoch + " - Reconstruction error: ";

    for(auto& line : lines){
        if(starts_with(line, begin)){
            auto sub = std::string(line.begin() + begin.size(), line.begin() + line.find(" - Sparsity"));
            error = std::stod(sub);
            found = true;
        }
    }

    return found;
}

std::vector<std::string> get_result(const dll::processor::options& opt, const std::vector<std::string>& actions, const std::string& source_file){
    auto result = dll::processor::process_file_result(opt, actions, "test/processor/" + source_file);

    std::cout << result << std::endl;

    std::stringstream stream(result);
    std::string current_line;
    std::vector<std::string> lines;

    while(std::getline(stream, current_line)) {
        std::string processed(cpp::trim(current_line));

        if(!processed.empty()){
            lines.emplace_back(std::move(processed));
        }
    }

    return lines;
}

dll::processor::options default_options(){
    dll::processor::options opt;
    opt.mkl = true;
    opt.quiet = true;
    opt.cache = false;
    return opt;
}

} // end of anonymous namespace

#define FT_ERROR_BELOW(min)                 \
    {                                       \
    double ft_error = 1.0;                  \
    REQUIRE(get_ft_error(lines, ft_error)); \
    std::cout << "ft_error:" << ft_error << std::endl;    \
    REQUIRE(ft_error < min);                \
    }

#define TEST_ERROR_BELOW(min)                   \
    {                                           \
    double test_error = 1.0;                    \
    REQUIRE(get_test_error(lines, test_error)); \
    std::cout << "test_error:" << test_error << std::endl;    \
    REQUIRE(test_error < min);                  \
    }

#define REC_ERROR_BELOW(epoch, min)                         \
    {                                                       \
    double rec_error = 1.0;                                 \
    REQUIRE(get_last_rec_error(epoch, lines, rec_error));   \
    std::cout << "rec_error:" << rec_error << std::endl;    \
    REQUIRE(rec_error < min);                               \
    }

// Dense (SGD)

TEST_CASE( "unit/processor/dense/sgd/1", "[unit][dense][dbn][mnist][sgd][proc]" ) {
    auto lines = get_result(default_options(), {"train", "test"}, "dense_sgd_1.conf");
    REQUIRE(!lines.empty());

    FT_ERROR_BELOW(5e-2);
    TEST_ERROR_BELOW(0.3);
}

// RBM

TEST_CASE( "unit/processor/rbm/1", "[unit][rbm][dbn][mnist][proc]" ) {
    auto lines = get_result(default_options(), {"pretrain"}, "rbm_1.conf");
    REQUIRE(!lines.empty());

    REC_ERROR_BELOW("epoch 24", 0.01);
}

TEST_CASE( "unit/processor/rbm/2", "[unit][rbm][dbn][mnist][proc]" ) {
    auto lines = get_result(default_options(), {"pretrain"}, "rbm_2.conf");
    REQUIRE(!lines.empty());

    REC_ERROR_BELOW("epoch 24", 0.01);
}

TEST_CASE( "unit/processor/rbm/3", "[unit][rbm][dbn][mnist][proc]" ) {
    auto lines = get_result(default_options(), {"pretrain"}, "rbm_3.conf");
    REQUIRE(!lines.empty());

    REC_ERROR_BELOW("epoch 99", 0.15);
}

TEST_CASE( "unit/processor/rbm/4", "[unit][rbm][dbn][mnist][proc]" ) {
    auto lines = get_result(default_options(), {"pretrain"}, "rbm_4.conf");
    REQUIRE(!lines.empty());

    REC_ERROR_BELOW("epoch 49", 0.01);
}

// CRBM

TEST_CASE( "unit/processor/crbm/1", "[unit][crbm][dbn][mnist][proc]" ) {
    auto lines = get_result(default_options(), {"pretrain"}, "crbm_1.conf");
    REQUIRE(!lines.empty());

    REC_ERROR_BELOW("epoch 24", 0.01);
}

TEST_CASE( "unit/processor/crbm/2", "[unit][crbm][dbn][mnist][proc]" ) {
    auto lines = get_result(default_options(), {"pretrain"}, "crbm_2.conf");
    REQUIRE(!lines.empty());

    REC_ERROR_BELOW("epoch 24", 0.01);
}
