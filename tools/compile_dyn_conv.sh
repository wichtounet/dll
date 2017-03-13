#!/bin/bash

make clean > /dev/null

echo "Compile 1 CRBM"
time make release_debug/workbench/src/compile_crbm_one.cpp.o > /dev/null
time make release_debug/bin/dll_compile_crbm_one > /dev/null

echo "Compile 1 Dynamic CRBM"
time make release_debug/workbench/src/compile_dyn_crbm_one.cpp.o > /dev/null
time make release_debug/bin/dll_compile_dyn_crbm_one > /dev/null

echo "Compile 1 Hybrid CRBM"
time make release_debug/workbench/src/compile_hybrid_crbm_one.cpp.o > /dev/null
time make release_debug/bin/dll_compile_hybrid_crbm_one > /dev/null

echo "Compile 5 CRBMs"
time make release_debug/workbench/src/compile_crbm.cpp.o > /dev/null
time make release_debug/bin/dll_compile_crbm > /dev/null

echo "Compile 5 Dynamic CRBMs"
time make release_debug/workbench/src/compile_dyn_crbm.cpp.o > /dev/null
time make release_debug/bin/dll_compile_dyn_crbm > /dev/null

echo "Compile 5 Hybrid CRBMs"
time make release_debug/workbench/src/compile_hybrid_crbm.cpp.o > /dev/null
time make release_debug/bin/dll_compile_hybrid_crbm > /dev/null
