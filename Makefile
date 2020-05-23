default: release_debug/bin/dllp

.PHONY: default release debug all clean

include make-utils/flags.mk
include make-utils/cpp-utils.mk

# Use C++20j
$(eval $(call use_cpp20))

CXX_FLAGS += -pedantic -Werror -ftemplate-backtrace-limit=0

# If asked, use libcxx (optional)
ifneq (,$(DLL_LIBCXX))
$(eval $(call use_libcxx))
endif

ifneq (,$(findstring clang,$(CXX)))
CXX_FLAGS += -Wno-documentation
endif

RELEASE_FLAGS += -fno-rtti

CXX_FLAGS += -Ietl/lib/include -Ietl/include/ -Imnist/include/ -Icifar-10/include/ -Idoctest -Inice_svm/include
LD_FLAGS += -lpthread

OPENCV_LD_FLAGS=-lopencv_core -lopencv_imgproc -lopencv_highgui
LIBSVM_LD_FLAGS=-lsvm
TEST_LD_FLAGS=$(LIBSVM_LD_FLAGS)

CXX_FLAGS += -DETL_PARALLEL -DETL_VECTORIZE_FULL

# Use the recommended limit
CXX_FLAGS += -ftemplate-depth=1024

# Tune GCC warnings
ifeq (,$(findstring clang,$(CXX)))
ifneq (,$(findstring g++,$(CXX)))
CXX_FLAGS += -Wno-ignored-attributes -Wno-misleading-indentation
endif
endif

# Sometimes more performance
#CXX_FLAGS += -DETL_CONV4_PREFER_BLAS

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

# Disable documentation warnings (too many false positives)
ifneq (,$(findstring clang,$(CXX)))
CXX_FLAGS += -Wno-documentation
endif

# On demand activation of full GPU support
ifneq (,$(ETL_GPU))
CXX_FLAGS += -DETL_GPU -DETL_EGBLAS_MODE

CXX_FLAGS += $(shell pkg-config --cflags cublas)
CXX_FLAGS += $(shell pkg-config --cflags cufft)
CXX_FLAGS += $(shell pkg-config --cflags cudnn)
CXX_FLAGS += $(shell pkg-config --cflags curand)
CXX_FLAGS += $(shell pkg-config --cflags egblas)

LD_FLAGS += $(shell pkg-config --libs cublas)
LD_FLAGS += $(shell pkg-config --libs cufft)
LD_FLAGS += $(shell pkg-config --libs cudnn)
LD_FLAGS += $(shell pkg-config --libs curand)
LD_FLAGS += $(shell pkg-config --libs egblas)
else

# On demand activation of cublas support
ifneq (,$(ETL_CUBLAS))
CXX_FLAGS += -DETL_CUBLAS_MODE $(shell pkg-config --cflags cublas)
LD_FLAGS += $(shell pkg-config --libs cublas)
endif

# On demand activation of cufft support
ifneq (,$(ETL_CUFFT))
CXX_FLAGS += -DETL_CUFFT_MODE $(shell pkg-config --cflags cufft)
LD_FLAGS += $(shell pkg-config --libs cufft)
endif

# On demand activation of cudnn support
ifneq (,$(ETL_CUDNN))
CXX_FLAGS += -DETL_CUDNN_MODE $(shell pkg-config --cflags cudnn)
LD_FLAGS += $(shell pkg-config --libs cudnn)
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
$(eval $(call enable_coverage_release_debug))
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
$(eval $(call auto_folder_compile,examples/src))

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

# Generate individual test executables (faster debugging)
$(eval $(call add_executable,dll_test_unit_augmentation,test/src/unit/test.cpp test/src/unit/augmentation.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_unit_bn,test/src/unit/test.cpp test/src/unit/bn.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_unit_conv_augmentation,test/src/unit/test.cpp test/src/unit/conv_augmentation.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_unit_cae,test/src/unit/test.cpp test/src/unit/cae.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_unit_cdbn_1,test/src/unit/test.cpp test/src/unit/cdbn_1.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_unit_cdbn_2,test/src/unit/test.cpp test/src/unit/cdbn_2.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_unit_cdbn_types,test/src/unit/test.cpp test/src/unit/cdbn_types.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_unit_conv_1,test/src/unit/test.cpp test/src/unit/conv_1.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_unit_conv_2,test/src/unit/test.cpp test/src/unit/conv_2.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_unit_conv_3,test/src/unit/test.cpp test/src/unit/conv_3.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_unit_conv_same,test/src/unit/test.cpp test/src/unit/conv_same.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_unit_conv_types,test/src/unit/test.cpp test/src/unit/conv_types.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_unit_crbm,test/src/unit/test.cpp test/src/unit/crbm.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_unit_crbm_mp,test/src/unit/test.cpp test/src/unit/crbm_mp.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_unit_crbm_mp_types,test/src/unit/test.cpp test/src/unit/crbm_mp_types.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_unit_crbm_types,test/src/unit/test.cpp test/src/unit/crbm_types.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_unit_dbn,test/src/unit/test.cpp test/src/unit/dbn.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_unit_dbn_ae,test/src/unit/test.cpp test/src/unit/dbn_ae.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_unit_dbn_types,test/src/unit/test.cpp test/src/unit/dbn_types.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_unit_dense,test/src/unit/test.cpp test/src/unit/dense.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_unit_dense_types,test/src/unit/test.cpp test/src/unit/dense_types.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_unit_dyn_crbm,test/src/unit/test.cpp test/src/unit/dyn_crbm.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_unit_dyn_crbm_mp,test/src/unit/test.cpp test/src/unit/dyn_crbm_mp.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_unit_dyn_dbn,test/src/unit/test.cpp test/src/unit/dyn_dbn.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_unit_dyn_dense,test/src/unit/test.cpp test/src/unit/dyn_dense.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_unit_dyn_rbm,test/src/unit/test.cpp test/src/unit/dyn_rbm.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_unit_initializer,test/src/unit/test.cpp test/src/unit/initializer.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_unit_lcn,test/src/unit/test.cpp test/src/unit/lcn.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_unit_processor,test/src/unit/test.cpp test/src/unit/processor.cpp $(PROCESSOR_TEST_CPP_FILES),$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_unit_random,test/src/unit/test.cpp test/src/unit/random.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_unit_rbm,test/src/unit/test.cpp test/src/unit/rbm.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_unit_rbm_types,test/src/unit/test.cpp test/src/unit/rbm_types.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_unit_rectifier,test/src/unit/test.cpp test/src/unit/rectifier.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_unit_text_reader,test/src/unit/test.cpp test/src/unit/text_reader.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_unit_unit,test/src/unit/test.cpp test/src/unit/unit.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_unit_embedding,test/src/unit/test.cpp test/src/unit/embedding.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_unit_rnn,test/src/unit/test.cpp test/src/unit/rnn.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_unit_lstm,test/src/unit/test.cpp test/src/unit/lstm.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_unit_reg,test/src/unit/test.cpp test/src/unit/reg.cpp,$(TEST_LD_FLAGS)))

# Generate individual misc executables (faster debugging)
$(eval $(call add_executable,dll_test_misc_autoencoder,test/src/misc/test.cpp test/src/misc/autoencoder.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_misc_autoencoder_dbn,test/src/misc/test.cpp test/src/misc/autoencoder_dbn.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_misc_cdbn_pooling,test/src/misc/test.cpp test/src/misc/cdbn_pooling.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_misc_cdbn_sgd_1,test/src/misc/test.cpp test/src/misc/cdbn_sgd_1.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_misc_cdbn_sgd_2,test/src/misc/test.cpp test/src/misc/cdbn_sgd_2.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_misc_cdbn_sgd_3,test/src/misc/test.cpp test/src/misc/cdbn_sgd_3.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_misc_cdbn_sgd_4,test/src/misc/test.cpp test/src/misc/cdbn_sgd_4.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_misc_cifar,test/src/misc/test.cpp test/src/misc/cifar.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_misc_conv,test/src/misc/test.cpp test/src/misc/conv.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_misc_conv_autoencoder,test/src/misc/test.cpp test/src/misc/conv_autoencoder.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_misc_conv_dbn,test/src/misc/test.cpp test/src/misc/conv_dbn.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_misc_conv_dbn_2,test/src/misc/test.cpp test/src/misc/conv_dbn_2.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_misc_conv_dbn_mp,test/src/misc/test.cpp test/src/misc/conv_dbn_mp.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_misc_crbm,test/src/misc/test.cpp test/src/misc/crbm.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_misc_crbm_mp,test/src/misc/test.cpp test/src/misc/crbm_mp.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_misc_crbm_mp_relu,test/src/misc/test.cpp test/src/misc/crbm_mp_relu.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_misc_crbm_mp_sparsity,test/src/misc/test.cpp test/src/misc/crbm_mp_sparsity.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_misc_crbm_relu,test/src/misc/test.cpp test/src/misc/crbm_relu.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_misc_crbm_sparsity,test/src/misc/test.cpp test/src/misc/crbm_sparsity.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_misc_dbn,test/src/misc/test.cpp test/src/misc/dbn.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_misc_dbn_sgd,test/src/misc/test.cpp test/src/misc/dbn_sgd.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_misc_dbn_svm,test/src/misc/test.cpp test/src/misc/dbn_svm.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_misc_dbn_transform,test/src/misc/test.cpp test/src/misc/dbn_transform.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_misc_dense,test/src/misc/test.cpp test/src/misc/dense.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_misc_dyn_conv,test/src/misc/test.cpp test/src/misc/dyn_conv.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_misc_dyn_conv_dbn,test/src/misc/test.cpp test/src/misc/dyn_conv_dbn.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_misc_dyn_crbm,test/src/misc/test.cpp test/src/misc/dyn_crbm.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_misc_dyn_crbm_mp,test/src/misc/test.cpp test/src/misc/dyn_crbm_mp.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_misc_dyn_dbn,test/src/misc/test.cpp test/src/misc/dyn_dbn.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_misc_dyn_dbn_cg,test/src/misc/test.cpp test/src/misc/dyn_dbn_cg.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_misc_dyn_dbn_sgd,test/src/misc/test.cpp test/src/misc/dyn_dbn_sgd.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_misc_dyn_dense,test/src/misc/test.cpp test/src/misc/dyn_dense.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_misc_dyn_rbm,test/src/misc/test.cpp test/src/misc/dyn_rbm.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_misc_hybrid,test/src/misc/test.cpp test/src/misc/hybrid.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_misc_lenet_dyn_rbm,test/src/misc/test.cpp test/src/misc/lenet_dyn_rbm.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_misc_lenet_mix,test/src/misc/test.cpp test/src/misc/lenet_mix.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_misc_lenet_rbm,test/src/misc/test.cpp test/src/misc/lenet_rbm.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_misc_rbm,test/src/misc/test.cpp test/src/misc/rbm.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_misc_rbm_pcd,test/src/misc/test.cpp test/src/misc/rbm_pcd.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_misc_rbm_relu,test/src/misc/test.cpp test/src/misc/rbm_relu.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_misc_rbm_sparsity,test/src/misc/test.cpp test/src/misc/rbm_sparsity.cpp,$(TEST_LD_FLAGS)))
$(eval $(call add_executable,dll_test_misc_rbm_smart,test/src/misc/test.cpp test/src/misc/rbm_smart.cpp,$(TEST_LD_FLAGS)))


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
$(eval $(call add_executable,dll_conv_sgd_perf,workbench/src/conv_sgd_perf.cpp))
$(eval $(call add_executable,dll_imagenet_perf,workbench/src/imagenet_perf.cpp))
$(eval $(call add_executable,dll_sgd_debug,workbench/src/sgd_debug.cpp))
$(eval $(call add_executable,dll_dae,workbench/src/dae.cpp))
$(eval $(call add_executable,dll_rbm_dae,workbench/src/rbm_dae.cpp))
$(eval $(call add_executable,dll_perf_paper,workbench/src/perf_paper.cpp))
$(eval $(call add_executable,dll_perf_paper_conv,workbench/src/perf_paper_conv.cpp))
$(eval $(call add_executable,dll_perf_conv,workbench/src/perf_conv.cpp))
$(eval $(call add_executable,dll_conv_types,workbench/src/conv_types.cpp))
$(eval $(call add_executable,dll_dyn_perf,workbench/src/dyn_perf.cpp))

# Perf examples
$(eval $(call add_executable,dll_mnist_mlp_perf,workbench/src/mnist_mlp_perf.cpp))
$(eval $(call add_executable,dll_mnist_cnn_perf,workbench/src/mnist_cnn_perf.cpp))
$(eval $(call add_executable,dll_mnist_ae_perf,workbench/src/mnist_ae_perf.cpp))
$(eval $(call add_executable,dll_mnist_deep_ae_perf,workbench/src/mnist_deep_ae_perf.cpp))
$(eval $(call add_executable,dll_mnist_dbn_perf,workbench/src/mnist_dbn_perf.cpp))

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

# Examples
$(eval $(call add_executable,dll_mnist_dbn,examples/src/mnist_dbn.cpp))
$(eval $(call add_executable_set,dll_mnist_dbn,dll_mnist_dbn))
$(eval $(call add_executable,dll_mnist_mlp,examples/src/mnist_mlp.cpp))
$(eval $(call add_executable_set,dll_mnist_mlp,dll_mnist_mlp))
$(eval $(call add_executable,dll_mnist_cnn,examples/src/mnist_cnn.cpp))
$(eval $(call add_executable_set,dll_mnist_cnn,dll_mnist_cnn))
$(eval $(call add_executable,dll_mnist_ae,examples/src/mnist_ae.cpp))
$(eval $(call add_executable_set,dll_mnist_ae,dll_mnist_ae))
$(eval $(call add_executable,dll_mnist_deep_ae,examples/src/mnist_deep_ae.cpp))
$(eval $(call add_executable_set,dll_mnist_deep_ae,dll_mnist_deep_ae))
$(eval $(call add_executable,dll_imagenet_cnn,examples/src/imagenet_cnn.cpp,$(OPENCV_LD_FLAGS)))
$(eval $(call add_executable_set,dll_imagenet_cnn,dll_imagenet_cnn))
$(eval $(call add_executable,dll_char_cnn,examples/src/char_cnn.cpp))
$(eval $(call add_executable_set,dll_char_cnn,dll_char_cnn))
$(eval $(call add_executable,dll_mnist_rnn,examples/src/mnist_rnn.cpp))
$(eval $(call add_executable_set,dll_mnist_rnn,dll_mnist_rnn))
$(eval $(call add_executable,dll_mnist_lstm,examples/src/mnist_lstm.cpp))
$(eval $(call add_executable_set,dll_mnist_lstm,dll_mnist_lstm))

$(eval $(call add_executable_set,dll_perf_paper,dll_perf_paper))
$(eval $(call add_executable_set,dll_perf_paper_conv,dll_perf_paper_conv))
$(eval $(call add_executable_set,dll_perf_conv,dll_perf_conv))
$(eval $(call add_executable_set,dll_conv_types,dll_conv_types))

# Build sets for workbench sources
debug_workbench: debug/bin/dll_sgd_perf debug/bin/dll_conv_sgd_perf debug/bin/dll_imagenet_perf debug/bin/dll_sgd_debug debug/bin/dll_dae debug/bin/dll_rbm_dae debug/bin/dll_perf_paper debug/bin/dll_perf_paper_conv debug/bin/dll_perf_conv debug/bin/dll_conv_types debug/bin/dll_dyn_perf
release_debug_workbench: release_debug/bin/dll_sgd_perf release_debug/bin/dll_conv_sgd_perf release_debug/bin/dll_imagenet_perf release_debug/bin/dll_sgd_debug release_debug/bin/dll_dae release_debug/bin/dll_rbm_dae release_debug/bin/dll_perf_paper release_debug/bin/dll_perf_paper_conv release_debug/bin/dll_perf_conv release_debug/bin/dll_conv_types release_debug/bin/dll_dyn_perf
release_workbench: release/bin/dll_sgd_perf release/bin/dll_conv_sgd_perf release/bin/dll_imagenet_perf release/bin/dll_sgd_debug release/bin/dll_dae release/bin/dll_rbm_dae release/bin/dll_perf_paper release/bin/dll_perf_paper_conv release/bin/dll_perf_conv release/bin/dll_conv_types release/bin/dll_dyn_perf

# Build sets for the examples
debug_examples: debug/bin/dll_mnist_mlp debug/bin/dll_mnist_cnn debug/bin/dll_mnist_ae debug/bin/dll_mnist_deep_ae debug/bin/dll_mnist_dbn
release_debug_examples: release_debug/bin/dll_mnist_mlp release_debug/bin/dll_mnist_cnn release_debug/bin/dll_mnist_ae release_debug/bin/dll_mnist_deep_ae release_debug/bin/dll_mnist_dbn
release_examples: release/bin/dll_mnist_mlp release/bin/dll_mnist_cnn release/bin/dll_mnist_ae release/bin/dll_mnist_deep_ae release/bin/dll_mnist_dbn

# Build sets for perf examples
debug_examples_perf: debug/bin/dll_mnist_mlp_perf debug/bin/dll_mnist_cnn_perf debug/bin/dll_mnist_ae_perf debug/bin/dll_mnist_deep_ae_perf debug/bin/dll_mnist_dbn_perf
release_debug_examples_perf: release_debug/bin/dll_mnist_mlp_perf release_debug/bin/dll_mnist_cnn_perf release_debug/bin/dll_mnist_ae_perf release_debug/bin/dll_mnist_deep_ae_perf release_debug/bin/dll_mnist_dbn_perf
release_examples_perf: release/bin/dll_mnist_mlp_perf release/bin/dll_mnist_cnn_perf release/bin/dll_mnist_ae_perf release/bin/dll_mnist_deep_ae_perf release/bin/dll_mnist_dbn_perf

debug: debug_dllp debug_dll_test_unit debug_dll_test_perf debug_dll_test_misc debug_dll_view debug_examples
release_debug: release_debug_dllp release_debug_dll_test_unit release_debug_dll_test_perf release_debug_dll_test_misc release_debug_dll_view release_debug_examples
release: release_dllp release_dll_test_unit release_dll_test_perf release_dll_test_misc release_dll_view release_examples

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
	cp -r cifar-10/include/cifar $(incdir)/

install_headers:
	cp -r include/dll $(incdir)/
	cp -r etl/include/etl $(incdir)/
	cp -r etl/lib/include/cpp_utils $(incdir)/
	cp -r mnist/include/mnist $(incdir)/
	cp -r cifar-10/include/cifar $(incdir)/

update_tests: release_dll_test
	bash tools/generate_tests.sh

doc:
	doxygen Doxyfile

clean: base_clean
	rm -rf latex/ html/

-include tests.mk

include make-utils/cpp-utils-finalize.mk
