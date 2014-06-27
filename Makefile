default: release

.PHONY: default release debug all clean

include make-utils/flags.mk
include make-utils/cpp-utils.mk

CXX_FLAGS += -Ietl/include/

CPP_FILES=$(wildcard test/*.cpp)

DEBUG_D_FILES=$(CPP_FILES:%.cpp=debug/%.cpp.d)
RELEASE_D_FILES=$(CPP_FILES:%.cpp=release/%.cpp.d)

$(eval $(call test_folder_compile,))

$(eval $(call add_test_executable,compile_rbm,compile_rbm.cpp))
$(eval $(call add_test_executable,compile_conv_rbm,compile_conv_rbm.cpp))
$(eval $(call add_test_executable,compile_dbn,compile_dbn.cpp))

$(eval $(call add_executable_set,compile_rbm,compile_rbm))
$(eval $(call add_executable_set,compile_conv_rbm,compile_conv_rbm))
$(eval $(call add_executable_set,compile_dbn,compile_dbn))

release: release_compile_rbm release_compile_conv_rbm release_compile_dbn
debug: debug_compile_rbm debug_compile_conv_rbm debug_compile_dbn

all: release debug
test: all

clean:
	rm -rf release/
	rm -rf debug/

-include $(DEBUG_D_FILES)
-include $(RELEASE_D_FILES)