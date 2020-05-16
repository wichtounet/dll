//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <deque>

#include "cpp_utils/string.hpp"

#include "dll_test.hpp"
#include "dll/processor/processor.hpp"

namespace {

bool starts_with(const std::string& str, const std::string& search) {
    return std::mismatch(search.begin(), search.end(), str.begin()).first == search.end();
}

std::string extract_value(const std::string& str, const std::string& search) {
    return {str.begin() + str.find(search) + search.size(), str.end()};
}

bool get_error(const std::vector<std::string>& lines, double& error, const char* begin) {
    for (auto& line : lines) {
        if (starts_with(line, begin)) {
            error = std::stod(extract_value(line, begin));
            return true;
        }
    }

    return false;
}

bool get_ft_error(const std::vector<std::string>& lines, double& error) {
    return get_error(lines, error, "Train Classification Error:");
}

bool get_test_error(const std::vector<std::string>& lines, double& error) {
    return get_error(lines, error, "Error rate: ");
}

bool get_last_rec_error(std::string epoch, const std::vector<std::string>& lines, double& error) {
    bool found = false;
    auto begin = epoch + " - Reconstruction error: ";

    for (auto& line : lines) {
        if (starts_with(line, begin)) {
            auto sub = std::string(line.begin() + begin.size(), line.begin() + line.find(" - Sparsity"));
            error    = std::stod(sub);
            found    = true;
        }
    }

    return found;
}

bool get_rec_error(std::string epoch, size_t index, const std::vector<std::string>& lines, double& error) {
    bool found = false;
    auto begin = epoch + " - Reconstruction error: ";

    size_t i = 0;

    for (auto& line : lines) {
        if (starts_with(line, begin)) {
            if(i++ == index){
                auto sub = std::string(line.begin() + begin.size(), line.begin() + line.find(" - Sparsity"));
                error    = std::stod(sub);
                found    = true;
            }
        }
    }

    return found;
}

bool get_last_sparsity(std::string epoch, const std::vector<std::string>& lines, double& error) {
    bool found = false;
    auto begin = epoch + " - Reconstruction error: ";

    for (auto& line : lines) {
        if (starts_with(line, begin)) {
            auto sub = std::string(line.begin() + line.find(" - Sparsity: ") + 13, line.end());
            error    = std::stod(sub);
            found    = true;
        }
    }

    return found;
}

bool get_sparsity(std::string epoch, size_t index, const std::vector<std::string>& lines, double& error) {
    bool found = false;
    auto begin = epoch + " - Reconstruction error: ";

    size_t i = 0;

    for (auto& line : lines) {
        if (starts_with(line, begin)) {
            if (i++ == index) {
                auto sub = std::string(line.begin() + line.find(" - Sparsity: ") + 13, line.end());
                error    = std::stod(sub);
                found    = true;
            }
        }
    }

    return found;
}

std::vector<std::string> get_result(const dll::processor::options& opt, const std::vector<std::string>& actions, const std::string& source_file) {
    auto result = dll::processor::process_file_result(opt, actions, "test/processor/" + source_file);

    std::cout << result << std::endl;

    std::stringstream stream(result);
    std::string current_line;
    std::vector<std::string> lines;

    while (std::getline(stream, current_line)) {
        std::string processed(cpp::trim(current_line));

        if (!processed.empty()) {
            lines.emplace_back(std::move(processed));
        }
    }

    return lines;
}

dll::processor::options default_options() {
    dll::processor::options opt;
    opt.mkl   = true;
    opt.quiet = true;
    opt.cache = false;
    return opt;
}

} // end of anonymous namespace

#define FT_ERROR_BELOW(min)                                \
    {                                                      \
        double ft_error = 1.0;                             \
        REQUIRE(get_ft_error(lines, ft_error));            \
        std::cout << "ft_error:" << ft_error << std::endl; \
        REQUIRE(ft_error < (min));                         \
    }

#define TEST_ERROR_BELOW(min)                                  \
    {                                                          \
        double test_error = 1.0;                               \
        REQUIRE(get_test_error(lines, test_error));            \
        std::cout << "test_error:" << test_error << std::endl; \
        REQUIRE(test_error < (min));                           \
    }

#define REC_ERROR_BELOW2(epoch, min)                          \
    {                                                         \
        double rec_error = 1.0;                               \
        REQUIRE(get_last_rec_error(epoch, lines, rec_error)); \
        std::cout << "rec_error:" << rec_error << std::endl;  \
        REQUIRE(rec_error < (min));                           \
    }

#define REC_ERROR_BELOW3(epoch, min, index)                     \
    {                                                           \
        double rec_error = 1.0;                                 \
        REQUIRE(get_rec_error(epoch, index, lines, rec_error)); \
        std::cout << "rec_error:" << rec_error << std::endl;    \
        REQUIRE(rec_error < (min));                             \
    }

// Simulate macro overloading
#define GET_MACRO(_1,_2,_3,NAME,...) NAME

#define REC_ERROR_BELOW(...) GET_MACRO(__VA_ARGS__, REC_ERROR_BELOW3, REC_ERROR_BELOW2, FAKE)(__VA_ARGS__)

#define SPARSITY_BELOW2(epoch, min)                         \
    {                                                       \
        double sparsity = 1.0;                              \
        REQUIRE(get_last_sparsity(epoch, lines, sparsity)); \
        std::cout << "sparsity:" << sparsity << std::endl;  \
        REQUIRE(sparsity < (min));                          \
    }

#define SPARSITY_BELOW3(epoch, min, index)                    \
    {                                                         \
        double sparsity = 1.0;                                \
        REQUIRE(get_sparsity(epoch, index, lines, sparsity)); \
        std::cout << "sparsity:" << sparsity << std::endl;    \
        REQUIRE(sparsity < (min));                            \
    }

#define SPARSITY_BELOW(...) GET_MACRO(__VA_ARGS__, SPARSITY_BELOW3, SPARSITY_BELOW2, FAKE)(__VA_ARGS__)

// Dense (SGD)

DLL_TEST_CASE("unit/processor/dense/sgd/1", "[unit][dense][dbn][mnist][sgd][proc]") {
    auto lines = get_result(default_options(), {"auto"}, "dense_sgd_1.conf");
    REQUIRE(!lines.empty());

    FT_ERROR_BELOW(5e-2);
    TEST_ERROR_BELOW(0.3);
}

DLL_TEST_CASE("unit/processor/dense/sgd/2", "[unit][dense][dbn][mnist][sgd][proc]") {
    auto lines = get_result(default_options(), {"train", "test"}, "dense_sgd_2.conf");
    REQUIRE(!lines.empty());

    FT_ERROR_BELOW(5e-2);
    TEST_ERROR_BELOW(0.3);
}

// Conv+Dense (SGD)

DLL_TEST_CASE("unit/processor/conv/sgd/1", "[unit][conv][dense][dbn][mnist][sgd][proc]") {
    auto lines = get_result(default_options(), {"train", "test"}, "conv_sgd_1.conf");
    REQUIRE(!lines.empty());

    FT_ERROR_BELOW(0.1);
    TEST_ERROR_BELOW(0.2);
}

DLL_TEST_CASE("unit/processor/conv/sgd/2", "[unit][conv][dense][dbn][mnist][sgd][proc]") {
    auto lines = get_result(default_options(), {"train", "test"}, "conv_sgd_2.conf");
    REQUIRE(!lines.empty());

    FT_ERROR_BELOW(1e-3);
    TEST_ERROR_BELOW(0.3);
}

DLL_TEST_CASE("unit/processor/conv/sgd/3", "[unit][conv][dense][dbn][mnist][sgd][proc]") {
    auto lines = get_result(default_options(), {"train", "test"}, "conv_sgd_3.conf");
    REQUIRE(!lines.empty());

    FT_ERROR_BELOW(5e-2);
    TEST_ERROR_BELOW(0.35); // tanh are much lower
}

// Not include in standard unit tests (covered by unit/processor/conv/sgd/5)
DLL_TEST_CASE("unit/processor/conv/sgd/4", "[unit_full][conv][dense][dbn][mnist][sgd][proc]") {
    auto lines = get_result(default_options(), {"train", "test"}, "conv_sgd_4.conf");
    REQUIRE(!lines.empty());

    FT_ERROR_BELOW(1e-3);
    TEST_ERROR_BELOW(0.3);
}

DLL_TEST_CASE("unit/processor/conv/sgd/5", "[unit][conv][dense][dbn][mnist][sgd][proc]") {
    auto lines = get_result(default_options(), {"train", "test"}, "conv_sgd_5.conf");
    REQUIRE(!lines.empty());

    FT_ERROR_BELOW(1e-3);
    TEST_ERROR_BELOW(0.3);
}

DLL_TEST_CASE("unit/processor/conv/sgd/6", "[unit][conv][dense][dbn][mnist][sgd][proc]") {
    auto lines = get_result(default_options(), {"train", "test"}, "conv_sgd_6.conf");
    REQUIRE(!lines.empty());

    FT_ERROR_BELOW(1e-3);
    TEST_ERROR_BELOW(0.2);
}

// Conv + Pool + Dense

DLL_TEST_CASE("unit/processor/conv/pool/sgd/1", "[unit][conv][dense][dbn][mnist][sgd][proc]") {
    auto lines = get_result(default_options(), {"train", "test"}, "conv_pool_sgd_1.conf");
    REQUIRE(!lines.empty());

    FT_ERROR_BELOW(1e-3);
    TEST_ERROR_BELOW(0.2);
}

DLL_TEST_CASE("unit/processor/conv/pool/sgd/2", "[unit][conv][dense][dbn][mnist][sgd][proc]") {
    auto lines = get_result(default_options(), {"train", "test"}, "conv_pool_sgd_2.conf");
    REQUIRE(!lines.empty());

    FT_ERROR_BELOW(1e-3);
    TEST_ERROR_BELOW(0.2);
}

// RBM

DLL_TEST_CASE("unit/processor/rbm/1", "[unit][rbm][dbn][mnist][proc]") {
    auto lines = get_result(default_options(), {"pretrain"}, "rbm_1.conf");
    REQUIRE(!lines.empty());

    REC_ERROR_BELOW("epoch 24", 0.01);
}

DLL_TEST_CASE("unit/processor/rbm/2", "[unit][rbm][dbn][mnist][proc]") {
    auto lines = get_result(default_options(), {"pretrain"}, "rbm_2.conf");
    REQUIRE(!lines.empty());

    REC_ERROR_BELOW("epoch 24", 0.01);
}

DLL_TEST_CASE("unit/processor/rbm/3", "[unit][rbm][dbn][mnist][proc]") {
    auto lines = get_result(default_options(), {"pretrain"}, "rbm_3.conf");
    REQUIRE(!lines.empty());

    REC_ERROR_BELOW("epoch 99", 0.15);
}

DLL_TEST_CASE("unit/processor/rbm/4", "[unit][rbm][dbn][mnist][proc]") {
    auto lines = get_result(default_options(), {"pretrain"}, "rbm_4.conf");
    REQUIRE(!lines.empty());

    REC_ERROR_BELOW("epoch 49", 0.01);
}

DLL_TEST_CASE("unit/processor/rbm/5", "[unit][rbm][dbn][mnist][proc]") {
    auto lines = get_result(default_options(), {"pretrain"}, "rbm_5.conf");
    REQUIRE(!lines.empty());

    REC_ERROR_BELOW("epoch 49", 0.01);
}

DLL_TEST_CASE("unit/processor/rbm/6", "[unit][rbm][dbn][mnist][proc]") {
    auto lines = get_result(default_options(), {"pretrain"}, "rbm_6.conf");
    REQUIRE(!lines.empty());

    REC_ERROR_BELOW("epoch 99", 0.15);
}

DLL_TEST_CASE("unit/processor/rbm/7", "[unit][rbm][dbn][mnist][proc]") {
    auto lines = get_result(default_options(), {"pretrain"}, "rbm_7.conf");
    REQUIRE(!lines.empty());

    REC_ERROR_BELOW("epoch 49", 0.01);
    SPARSITY_BELOW("epoch 49", 0.12);
}

DLL_TEST_CASE("unit/processor/rbm/8", "[unit][rbm][dbn][mnist][proc]") {
    auto lines = get_result(default_options(), {"pretrain"}, "rbm_8.conf");
    REQUIRE(!lines.empty());

    REC_ERROR_BELOW("epoch 49", 0.03);
    SPARSITY_BELOW("epoch 49", 0.12);
}

DLL_TEST_CASE("unit/processor/rbm/9", "[unit][rbm][dbn][mnist][proc]") {
    auto lines = get_result(default_options(), {"pretrain"}, "rbm_9.conf");
    REQUIRE(!lines.empty());

    REC_ERROR_BELOW("epoch 49", 0.2);
}

// CRBM

DLL_TEST_CASE("unit/processor/crbm/1", "[unit][crbm][dbn][mnist][proc]") {
    auto lines = get_result(default_options(), {"pretrain"}, "crbm_1.conf");
    REQUIRE(!lines.empty());

    REC_ERROR_BELOW("epoch 24", 0.01);
}

DLL_TEST_CASE("unit/processor/crbm/2", "[unit][crbm][dbn][mnist][proc]") {
    auto lines = get_result(default_options(), {"pretrain"}, "crbm_2.conf");
    REQUIRE(!lines.empty());

    REC_ERROR_BELOW("epoch 24", 0.01);
}

// CRBM (MP)

DLL_TEST_CASE("unit/processor/crbm_mp/1", "[unit][crbm_mp][dbn][mnist][proc]") {
    auto lines = get_result(default_options(), {"pretrain"}, "crbm_mp_1.conf");
    REQUIRE(!lines.empty());

    REC_ERROR_BELOW("epoch 49", 0.01);
}

// DBN (SGD)

DLL_TEST_CASE("unit/processor/dbn/sgd/1", "[unit][dense][dbn][mnist][sgd][proc]") {
    auto lines = get_result(default_options(), {"pretrain", "train", "test"}, "dbn_sgd_1.conf");
    REQUIRE(!lines.empty());

    FT_ERROR_BELOW(5e-2);
    TEST_ERROR_BELOW(0.3);
}

// DBN (CG)

DLL_TEST_CASE("unit/processor/dbn/cg/1", "[unit][dense][dbn][mnist][sgd][proc]") {
    auto lines = get_result(default_options(), {"pretrain", "train", "test"}, "dbn_cg_1.conf");
    REQUIRE(!lines.empty());

    FT_ERROR_BELOW(5e-2);
    TEST_ERROR_BELOW(0.3);
}

// Conv DBN

// Disable for time reasons (unit/processor/cdbn/2 is testing more anyway)
DLL_TEST_CASE("unit/processor/cdbn/1", "[unit_full][dbn][mnist][conv][proc]") {
    auto lines = get_result(default_options(), {"pretrain"}, "cdbn_1.conf");
    REQUIRE(!lines.empty());

    REC_ERROR_BELOW("epoch 24", 0.01, 0);
    REC_ERROR_BELOW("epoch 24", 0.03, 1);
}

DLL_TEST_CASE("unit/processor/cdbn/2", "[unit][dbn][mnist][conv][proc]") {
    auto lines = get_result(default_options(), {"pretrain"}, "cdbn_2.conf");
    REQUIRE(!lines.empty());

    REC_ERROR_BELOW("epoch 24", 0.025, 0);
    REC_ERROR_BELOW("epoch 24", 0.05, 1);

    SPARSITY_BELOW("epoch 24", 0.4, 0);
    SPARSITY_BELOW("epoch 24", 0.35, 1);
}
