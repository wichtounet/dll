default: release_debug/bin/dllp

.PHONY: default release debug all clean

include make-utils/flags.mk
include make-utils/cpp-utils.mk

CXX_FLAGS += -pedantic -Werror -ftemplate-backtrace-limit=0

# If asked, use libcxx (optional)
ifneq (,$(DLL_LIBCXX))
$(eval $(call use_libcxx))
endif

ifneq (,$(findstring clang,$(CXX)))
CXX_FLAGS += -Wno-error=documentation
endif

RELEASE_FLAGS += -fno-rtti

CXX_FLAGS += -Ietl/lib/include -Ietl/include/ -Imnist/include/ -ICatch/include -Inice_svm/include
LD_FLAGS += -lpthread

OPENCV_LD_FLAGS=-lopencv_core -lopencv_imgproc -lopencv_highgui
LIBSVM_LD_FLAGS=-lsvm
TEST_LD_FLAGS=$(LIBSVM_LD_FLAGS)

CXX_FLAGS += -DETL_PARALLEL -DETL_VECTORIZE_FULL

# Activate NaN Debugging (if not in perf mode)
ifeq (,$(DLL_PERF))
DEBUG_FLAGS += -DNAN_DEBUG
endif

ifneq (,$(DLL_PERF_FLAGS))
CXX_FLAGS += $(DLL_PERF_FLAGS)
endif

DLL_BLAS_PKG ?= mkl

# Try to detect parallel mkl
ifneq (,$(findstring threads,$(DLL_BLAS_PKG)))
CXX_FLAGS += -DETL_BLAS_THREADS
endif

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

# On demand activation of full GPU support
ifneq (,$(ETL_GPU))
CXX_FLAGS += -DETL_GPU

CXX_FLAGS += $(shell pkg-config --cflags cublas)
CXX_FLAGS += $(shell pkg-config --cflags cufft)
CXX_FLAGS += $(shell pkg-config --cflags cudnn)

LD_FLAGS += $(shell pkg-config --libs cublas)
LD_FLAGS += $(shell pkg-config --libs cufft)
LD_FLAGS += $(shell pkg-config --libs cudnn)

ifneq (,$(findstring clang,$(CXX)))
CXX_FLAGS += -Wno-documentation
endif
else

# On demand activation of cublas support
ifneq (,$(ETL_CUBLAS))
CXX_FLAGS += -DETL_CUBLAS_MODE $(shell pkg-config --cflags cublas)
LD_FLAGS += $(shell pkg-config --libs cublas)

ifneq (,$(findstring clang,$(CXX)))
CXX_FLAGS += -Wno-documentation
endif
endif

# On demand activation of cufft support
ifneq (,$(ETL_CUFFT))
CXX_FLAGS += -DETL_CUFFT_MODE $(shell pkg-config --cflags cufft)
LD_FLAGS += $(shell pkg-config --libs cufft)

ifneq (,$(findstring clang,$(CXX)))
CXX_FLAGS += -Wno-documentation
endif
endif

# On demand activation of cudnn support
ifneq (,$(ETL_CUDNN))
CXX_FLAGS += -DETL_CUDNN_MODE $(shell pkg-config --cflags cudnn)
LD_FLAGS += $(shell pkg-config --libs cudnn)

ifneq (,$(findstring clang,$(CXX)))
CXX_FLAGS += -Wno-documentation
endif
endif

endif

# On demand activation of egblas support
ifneq (,$(ETL_EGBLAS))
CXX_FLAGS += -DETL_EGBLAS_MODE $(shell pkg-config --cflags egblas)
LD_FLAGS += $(shell pkg-config --libs egblas)
endif

# Enable Clang sanitizers in debug mode
ifneq (,$(findstring clang,$(CXX)))
ifeq (,$(ETL_CUBLAS))
ifneq (,$(DLL_SAN_THREAD))
DEBUG_FLAGS += -fsanitize=undefined,thread
else
DEBUG_FLAGS += -fsanitize=address,undefined
endif
endif
endif

# Activate hybrid compilation by default
ifneq (,$(DLL_QUICK))
CXX_FLAGS += -DDLL_QUICK
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
PROCESSOR_TEST_CPP_FILES := $(filter-out processor/src/main.cpp,$(PROCESSOR_CPP_FILES))

UNIT_TEST_CPP_FILES=$(wildcard test/src/unit/*.cpp)
PERF_TEST_CPP_FILES=$(wildcard test/src/perf/*.cpp)
MISC_TEST_CPP_FILES=$(wildcard test/src/misc/*.cpp)

UNIT_TEST_FILES=$(UNIT_TEST_CPP_FILES) $(PROCESSOR_TEST_CPP_FILES)
PERF_TEST_FILES=$(PERF_TEST_CPP_FILES) $(PROCESSOR_TEST_CPP_FILES)
MISC_TEST_FILES=$(MISC_TEST_CPP_FILES) $(PROCESSOR_TEST_CPP_FILES)

# Compile all the sources
$(eval $(call auto_folder_compile,processor/src,-Iprocessor/include))
$(eval $(call auto_folder_compile,test/src/unit,-Itest/include))
$(eval $(call auto_folder_compile,test/src/perf,-Itest/include))
$(eval $(call auto_folder_compile,test/src/misc,-Itest/include))
$(eval $(call auto_folder_compile,view/src))
$(eval $(call auto_folder_compile,workbench/src,-DDLL_SILENT))

# Generate executable for the prepropcessor
$(eval $(call add_executable,dllp,$(PROCESSOR_CPP_FILES)))
$(eval $(call add_executable_set,dllp,dllp))

# Generate executable for the test executables
$(eval $(call add_executable,dll_test_unit,$(UNIT_TEST_FILES),$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_perf,$(PERF_TEST_FILES),$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_misc,$(MISC_TEST_FILES),$(TEST_LD_FLAGS)))
$(eval $(call add_executable_set,dll_test_unit,dll_test_unit))
$(eval $(call add_executable_set,dll_test_perf,dll_test_perf))
$(eval $(call add_executable_set,dll_test_misc,dll_test_misc))

# Generate executables for visualization
$(eval $(call add_executable,dll_view_rbm,view/src/rbm_view.cpp,$(OPENCV_LD_FLAGS)))
$(eval $(call add_executable,dll_view_crbm,view/src/crbm_view.cpp,$(OPENCV_LD_FLAGS)))
$(eval $(call add_executable,dll_view_crbm_mp,view/src/crbm_mp_view.cpp,$(OPENCV_LD_FLAGS)))
$(eval $(call add_executable_set,dll_view_rbm,dll_view_rbm))
$(eval $(call add_executable_set,dll_view_crbm,dll_view_crbm))
$(eval $(call add_executable_set,dll_view_crbm_mp,dll_view_crbm_mp))
$(eval $(call add_executable_set,dll_view,dll_view_rbm, dll_view_crbm, dll_view_crbm_mp))

# Generate executables for performance analysis
$(eval $(call add_executable,dll_sgd_perf,workbench/src/sgd_perf.cpp))
$(eval $(call add_executable,dll_sgd_debug,workbench/src/sgd_debug.cpp))
$(eval $(call add_executable,dll_dae,workbench/src/dae.cpp))
$(eval $(call add_executable,dll_rbm_dae,workbench/src/rbm_dae.cpp))
$(eval $(call add_executable,dll_perf_paper,workbench/src/perf_paper.cpp))
$(eval $(call add_executable,dll_perf_paper_conv,workbench/src/perf_paper_conv.cpp))
$(eval $(call add_executable,dll_perf_conv,workbench/src/perf_conv.cpp))
$(eval $(call add_executable,dll_conv_types,workbench/src/conv_types.cpp))
$(eval $(call add_executable,dll_dyn_perf,workbench/src/dyn_perf.cpp))

# Analysis of performance and compilation time
$(eval $(call add_executable,dll_compile_rbm_one,workbench/src/compile_rbm_one.cpp))
$(eval $(call add_executable,dll_compile_dyn_rbm_one,workbench/src/compile_dyn_rbm_one.cpp))
$(eval $(call add_executable,dll_compile_rbm,workbench/src/compile_rbm.cpp))
$(eval $(call add_executable,dll_compile_dyn_rbm,workbench/src/compile_dyn_rbm.cpp))
$(eval $(call add_executable,dll_compile_hybrid_rbm_one,workbench/src/compile_hybrid_rbm_one.cpp))
$(eval $(call add_executable,dll_compile_hybrid_rbm,workbench/src/compile_hybrid_rbm.cpp))
$(eval $(call add_executable,dll_compile_crbm_one,workbench/src/compile_crbm_one.cpp))
$(eval $(call add_executable,dll_compile_dyn_crbm_one,workbench/src/compile_dyn_crbm_one.cpp))
$(eval $(call add_executable,dll_compile_crbm,workbench/src/compile_crbm.cpp))
$(eval $(call add_executable,dll_compile_dyn_crbm,workbench/src/compile_dyn_crbm.cpp))
$(eval $(call add_executable,dll_compile_hybrid_crbm_one,workbench/src/compile_hybrid_crbm_one.cpp))
$(eval $(call add_executable,dll_compile_hybrid_crbm,workbench/src/compile_hybrid_crbm.cpp))

$(eval $(call add_executable_set,dll_perf_paper,dll_perf_paper))
$(eval $(call add_executable_set,dll_perf_paper_conv,dll_perf_paper_conv))
$(eval $(call add_executable_set,dll_perf_conv,dll_perf_conv))
$(eval $(call add_executable_set,dll_conv_types,dll_conv_types))

release: release_dllp release_dll_test_unit release_dll_test_perf release_dll_test_misc release_dll_view
release_debug: release_debug_dllp release_debug_dll_test_unit release_debug_dll_test_perf release_debug_dll_test_misc release_debug_dll_view
debug: debug_dllp debug_dll_test_unit debug_dll_test_perf debug_dll_test_misc debug_dll_view

all: release debug release_debug

debug_test: debug_dll_test_unit
	./debug/bin/dll_test_unit

release_test: release_dll_test_unit
	./release/bin/dll_test_unit

release_debug_test: release_debug_dll_test_unit
	./release_debug/bin/dll_test_unit

test: all
	./debug/bin/dll_test_unit
	./release/bin/dll_test_unit
	./release_debug/bin/dll_test_unit

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

install_headers:
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
