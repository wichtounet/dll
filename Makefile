default: release

.PHONY: default release debug all clean

CPP_FILES=$(wildcard src/*.cpp)

DEBUG_D_FILES=$(CPP_FILES:%.cpp=debug/%.cpp.d)
RELEASE_D_FILES=$(CPP_FILES:%.cpp=release/%.cpp.d)

DEBUG_O_FILES=$(CPP_FILES:%.cpp=debug/%.cpp.o)
RELEASE_O_FILES=$(CPP_FILES:%.cpp=release/%.cpp.o)

NONEXEC_CPP_FILES := $(filter-out src/rbm_simple.cpp,$(CPP_FILES))
NONEXEC_CPP_FILES := $(filter-out src/rbm_mnist.cpp,$(NONEXEC_CPP_FILES))
NONEXEC_CPP_FILES := $(filter-out src/dbn_mnist.cpp,$(NONEXEC_CPP_FILES))
NONEXEC_CPP_FILES := $(filter-out src/fast_vector_test.cpp,$(NONEXEC_CPP_FILES))

NON_EXEC_DEBUG_O_FILES=$(NONEXEC_CPP_FILES:%.cpp=debug/%.cpp.o)
NON_EXEC_RELEASE_O_FILES=$(NONEXEC_CPP_FILES:%.cpp=release/%.cpp.o)

CC=clang++
LD=clang++

WARNING_FLAGS=-Wextra -Wall -Wno-unused-function -Qunused-arguments -Wunitialized -Wsometimes-unitialized -Wno-long-long -Winit-self -Wdocumentation
CXX_FLAGS=-Iinclude -std=c++1y -stdlib=libc++
LD_FLAGS=$(CXX_FLAGS)

DEBUG_FLAGS=-g
RELEASE_FLAGS=-g -DNDEBUG -Ofast -march=native -fvectorize -fslp-vectorize-aggressive -fomit-frame-pointer

debug/src/%.cpp.o: src/%.cpp
	@ mkdir -p debug/src/
	$(CC) $(CXX_FLAGS) $(DEBUG_FLAGS) -o $@ -c $<

release/src/%.cpp.o: src/%.cpp
	@ mkdir -p release/src/
	$(CC) $(CXX_FLAGS) $(RELEASE_FLAGS) -o $@ -c $<

debug/bin/%: debug/src/%.cpp.o $(NON_EXEC_DEBUG_O_FILES)
	@ mkdir -p debug/bin/
	$(LD) $(LD_FLAGS) $(DEBUG_FLAGS) -o $@ $?

release/bin/%: release/src/%.cpp.o $(NON_EXEC_RELEASE_O_FILES)
	@ mkdir -p release/bin/
	$(LD) $(LD_FLAGS) $(RELEASE_FLAGS) -o $@ $?

debug/src/%.cpp.d: $(CPP_FILES)
	@ mkdir -p debug/src/
	@ $(CC) $(CXX_FLAGS) $(DEBUG_FLAGS) -MM -MT debug/src/$*.cpp.o src/$*.cpp | sed -e 's@^\(.*\)\.o:@\1.d \1.o:@' > $@

release/src/%.cpp.d: $(CPP_FILES)
	@ mkdir -p release/src/
	@ $(CC) $(CXX_FLAGS) $(RELEASE_FLAGS) -MM -MT release/src/$*.cpp.o src/$*.cpp | sed -e 's@^\(.*\)\.o:@\1.d \1.o:@' > $@

release: release/bin/rbm_simple release/bin/rbm_mnist release/bin/dbn_mnist release/bin/fast_vector_test
debug: debug/bin/rbm_simple debug/bin/rbm_mnist debug/bin/dbn_mnist debug/bin/fast_vector_test

all: release debug

clean:
	rm -rf release/
	rm -rf debug/

-include $(DEBUG_D_FILES)
-include $(RELEASE_D_FILES)