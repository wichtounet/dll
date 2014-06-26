default: release

.PHONY: default release debug all clean

include make-utils/flags.mk
include make-utils/cpp-utils.mk

$(eval $(call test_folder_compile,))

$(eval $(call add_test_executable,compile_rbm,compile_rbm.cpp))

$(eval $(call add_executable_set,compile_rbm,compile_rbm))

release: release_compile_rbm
debug: debug_compile_rbm

all: release debug
test: all

clean:
	rm -rf release/
	rm -rf debug/

-include $(DEBUG_D_FILES)
-include $(RELEASE_D_FILES)