default: release

.PHONY: default release debug all clean

include make-utils/flags.mk
include make-utils/cpp-utils.mk

ifeq ($(CXX),clang++)
	CXX_FLAGS += -stdlib=libc++
endif

CXX_FLAGS += -Ietl/include/ -Imnist/include/ -ICatch/include -Werror
OPENCV_LD_FLAGS=-lopencv_core -lopencv_imgproc -lopencv_highgui

CPP_FILES=$(wildcard test_compile/*.cpp)

TEST_CPP_FILES=$(wildcard test/*.cpp)
TEST_FILES=$(TEST_CPP_FILES:test/%=%)

DEBUG_D_FILES=$(CPP_FILES:%.cpp=debug/%.cpp.d) $(TEST_CPP_FILES:%.cpp=debug/%.cpp.d)
RELEASE_D_FILES=$(CPP_FILES:%.cpp=release/%.cpp.d) $(TEST_CPP_FILES:%.cpp=release/%.cpp.d)

$(eval $(call folder_compile,test_compile))
$(eval $(call test_folder_compile,))

$(eval $(call add_executable,compile_rbm,test_compile/compile_rbm.cpp))
$(eval $(call add_executable,compile_conv_rbm,test_compile/compile_conv_rbm.cpp))
$(eval $(call add_executable,compile_conv_rbm_mp,test_compile/compile_conv_rbm_mp.cpp))
$(eval $(call add_executable,compile_dbn,test_compile/compile_dbn.cpp))
$(eval $(call add_executable,compile_conv_dbn,test_compile/compile_conv_dbn.cpp))
$(eval $(call add_executable,compile_ocv_1,test_compile/rbm_view.cpp,$(OPENCV_LD_FLAGS)))
$(eval $(call add_executable,compile_ocv_2,test_compile/crbm_view.cpp,$(OPENCV_LD_FLAGS)))
$(eval $(call add_executable,compile_ocv_3,test_compile/crbm_mp_view.cpp,$(OPENCV_LD_FLAGS)))

$(eval $(call add_test_executable,dll_test,$(TEST_FILES)))

$(eval $(call add_executable_set,compile_rbm,compile_rbm))
$(eval $(call add_executable_set,compile_conv_rbm,compile_conv_rbm))
$(eval $(call add_executable_set,compile_conv_rbm_mp,compile_conv_rbm_mp))
$(eval $(call add_executable_set,compile_dbn,compile_dbn))
$(eval $(call add_executable_set,compile_conv_dbn,compile_conv_dbn))
$(eval $(call add_executable_set,dll_test,dll_test))

release: release_compile_rbm release_compile_conv_rbm release_compile_dbn release_compile_conv_dbn release_compile_conv_rbm_mp release_dll_test
debug: debug_compile_rbm debug_compile_conv_rbm debug_compile_dbn debug_compile_conv_dbn debug_compile_conv_rbm_mp debug_dll_test

all: release debug

debug_test: debug
	./debug/bin/dll_test

release_test: release
	./release/bin/dll_test

test: all
	./debug/bin/dll_test
	./release/bin/dll_test

update_tests: release_dll_test
	bash tools/generate_tests.sh

doc:
	doxygen Doxyfile

clean:
	rm -rf release/
	rm -rf debug/
	rm -rf latex/ html/

-include tests.mk

-include $(DEBUG_D_FILES)
-include $(RELEASE_D_FILES)