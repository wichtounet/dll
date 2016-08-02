#!/bin/bash

make clean > /dev/null

echo "Compile 1 RBM"
time make release_debug/bin/dll_compile_rbm_one > /dev/null

echo "Compile 1 Dynamic RBM"
time make release_debug/bin/dll_compile_dyn_rbm_one > /dev/null

echo "Compile 5 RBMs"
time make release_debug/bin/dll_compile_rbm > /dev/null

echo "Compile 5 Dynamic RBMs"
time make release_debug/bin/dll_compile_dyn_rbm > /dev/null
