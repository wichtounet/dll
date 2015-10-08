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

} // end of anonymous namespace

TEST_CASE( "unit/processor/dense/sgd/1", "[unit][dense][dbn][mnist][sgd]" ) {
    dll::processor::options opt;
    opt.mkl = true;
    opt.quiet = true;

    std::vector<std::string> actions{"train", "test"};

    auto result = dll::processor::process_file_result(opt, actions, "test/processor/dense_sgd_1.conf");

    std::stringstream stream(result);
    std::string current_line;
    std::vector<std::string> lines;

    while(std::getline(stream, current_line)) {
        std::string processed(cpp::trim(current_line));

        if(!processed.empty()){
            lines.emplace_back(std::move(processed));
        }
    }

    REQUIRE(!lines.empty());

    double ft_error = 1.0;
    REQUIRE(get_ft_error(lines, ft_error));
    REQUIRE(ft_error < 5e-2);

    double test_error = 1.0;
    REQUIRE(get_test_error(lines, test_error));
    REQUIRE(test_error < 0.3);
}
