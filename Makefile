default: release_debug/bin/dllp

.PHONY: default release debug all clean

include make-utils/flags.mk
include make-utils/cpp-utils.mk

CXX_FLAGS += -pedantic -Werror -ftemplate-backtrace-limit=0

$(eval $(call use_libcxx))

RELEASE_FLAGS += -fno-rtti

CXX_FLAGS += -Ietl/lib/include -Ietl/include/ -Imnist/include/ -ICatch/include -Inice_svm/include
LD_FLAGS += -lpthread

OPENCV_LD_FLAGS=-lopencv_core -lopencv_imgproc -lopencv_highgui
LIBSVM_LD_FLAGS=-lsvm
TEST_LD_FLAGS=$(LIBSVM_LD_FLAGS)

CXX_FLAGS += -DETL_VECTORIZE_FULL

# Activate NaN Debugging
ifeq (,$(DLL_PERF))
DEBUG_FLAGS += -DNAN_DEBUG
endif

DLL_BLAS_PKG ?= mkl

# Activate BLAS mode on demand
ifneq (,$(ETL_MKL))
CXX_FLAGS += -DETL_MKL_MODE $(shell pkg-config --cflags $(DLL_BLAS_PKG))
LD_FLAGS += $(shell pkg-config --libs $(DLL_BLAS_PKG))

# Disable warning for MKL
ifneq (,$(findstring clang,$(CXX)))
CXX_FLAGS += -Wno-tautological-compare
endif

else
ifneq (,$(ETL_BLAS))
CXX_FLAGS += -DETL_BLAS_MODE $(shell pkg-config --cflags cblas)
LD_FLAGS += $(shell pkg-config --libs cblas)

# Disable warning for MKL
ifneq (,$(findstring clang,$(CXX)))
CXX_FLAGS += -Wno-tautological-compare
endif

endif
endif

# Disable timers on demand
ifneq (,$(DLL_NO_TIMERS))
CXX_FLAGS += -DDLL_NO_TIMERS
endif

# Enable coverage if enabled
ifneq (,$(DLL_COVERAGE))
$(eval $(call enable_coverage_release))
endif

CPP_FILES=$(wildcard view/*.cpp)
PROCESSOR_CPP_FILES=$(wildcard processor/src/*.cpp)
TEST_CPP_FILES=$(wildcard test/src/*.cpp)
PROCESSOR_TEST_CPP_FILES := $(filter-out processor/src/main.cpp,$(PROCESSOR_CPP_FILES))

TEST_FILES=$(TEST_CPP_FILES) $(PROCESSOR_TEST_CPP_FILES)

# Compile all the sources
$(eval $(call auto_folder_compile,processor/src,-Iprocessor/include))
$(eval $(call auto_folder_compile,test/src,-Itest/include))
$(eval $(call auto_folder_compile,view/src))
$(eval $(call auto_folder_compile,workbench/src,-DDLL_SILENT))

# Generate executable for the prepropcessor
$(eval $(call add_executable,dllp,$(PROCESSOR_CPP_FILES)))
$(eval $(call add_executable_set,dllp,dllp))

# Generate executable for the test executable
$(eval $(call add_executable,dll_test,$(TEST_FILES),$(TEST_LD_FLAGS)))
$(eval $(call add_executable_set,dll_test,dll_test))

# Generate executables for visualization
$(eval $(call add_executable,dll_view_rbm,view/src/rbm_view.cpp,$(OPENCV_LD_FLAGS)))
$(eval $(call add_executable,dll_view_crbm,view/src/crbm_view.cpp,$(OPENCV_LD_FLAGS)))
$(eval $(call add_executable,dll_view_crbm_mp,view/src/crbm_mp_view.cpp,$(OPENCV_LD_FLAGS)))
$(eval $(call add_executable_set,dll_view_rbm,dll_view_rbm))
$(eval $(call add_executable_set,dll_view_crbm,dll_view_crbm))
$(eval $(call add_executable_set,dll_view_crbm_mp,dll_view_crbm_mp))
$(eval $(call add_executable_set,dll_view,dll_view_rbm, dll_view_crbm, dll_view_crbm_mp))

# Generate executables for performance analysis
$(eval $(call add_executable,dll_perf,workbench/src/perf_paper.cpp))
$(eval $(call add_executable,dll_perf_conv,workbench/src/perf_paper_conv.cpp))
$(eval $(call add_executable_set,dll_perf,dll_perf))
$(eval $(call add_executable_set,dll_perf_conv,dll_perf_conv))

release: release_dllp release_dll_test release_dll_view
release_debug: release_debug_dllp release_debug_dll_test release_debug_dll_view
debug: debug_dllp debug_dll_test debug_dll_view

all: release debug release_debug

debug_test: debug_dll_test
	./debug/bin/dll_test [unit]

release_test: release_dll_test
	./release/bin/dll_test [unit]

release_debug_test: release_debug_dll_test
	./release_debug/bin/dll_test [unit]

test: all
	./debug/bin/dll_test
	./release/bin/dll_test
	./release_debug/bin/dll_test

CLANG_FORMAT ?= clang-format-3.7
CLANG_MODERNIZE ?= clang-modernize-3.7
CLANG_TIDY ?= clang-tidy-3.7

format:
	git ls-files "*.hpp" "*.cpp" "*.inl" | xargs ${CLANG_FORMAT} -i -style=file

modernize:
	find include test processor view -name "*.hpp" -o -name "*.cpp" > dll_file_list
	${CLANG_MODERNIZE} -add-override -loop-convert -pass-by-value -use-auto -use-nullptr -p ${PWD} -include-from=dll_file_list
	rm etl_file_list

# clang-tidy with some false positive checks removed
tidy:
	${CLANG_TIDY} -checks='*,-llvm-include-order,-clang-analyzer-alpha.core.PointerArithm,-clang-analyzer-alpha.deadcode.UnreachableCode,-clang-analyzer-alpha.core.IdenticalExpr' -p ${PWD} test/src/*.cpp processor/src/*.cpp -header-filter='include/dll/*' &> tidy_report_light
	echo "The report from clang-tidy is availabe in tidy_report_light"

# clang-tidy with all the checks
tidy_all:
	${CLANG_TIDY} -checks='*' -p ${PWD} test/*.cpp processor/src/*.cpp -header-filter='include/dll/*' &> tidy_report_all
	echo "The report from clang-tidy is availabe in tidy_report_all"

prefix = /usr
bindir = $(prefix)/bin
incdir = $(prefix)/include

install: release_debug/bin/dllp
	@ echo "Installation of dll"
	@ echo "============================="
	@ echo ""
	install release_debug/bin/dllp $(bindir)/dllp
	cp -r include/dll $(incdir)/
	cp -r etl/include/etl $(incdir)/
	cp -r etl/lib/include/cpp_utils $(incdir)/
	cp -r mnist/include/mnist $(incdir)/

update_tests: release_dll_test
	bash tools/generate_tests.sh

doc:
	doxygen Doxyfile

clean: base_clean
	rm -rf latex/ html/

-include tests.mk

include make-utils/cpp-utils-finalize.mk
