#!/bin/bash

export ETL_MKL=true
export DLL_QUICK=true
export LD=true

# Let's save some memory
killall zapccs
killall vivaldi-bin

export CXX=clang++

echo "Debug clang (-j1)"
make clean > /dev/null
time make -j1 debug/bin/dll_test_unit > /dev/null

echo "Debug clang (-j2)"
make clean > /dev/null
time make -j2 debug/bin/dll_test_unit > /dev/null

echo "Debug clang (-j3)"
make clean > /dev/null
time make -j3 debug/bin/dll_test_unit > /dev/null

echo "Debug clang (-j4)"
make clean > /dev/null
time make -j4 debug/bin/dll_test_unit > /dev/null

echo "Release clang (-j1)"
make clean > /dev/null
time make -j1 release_debug/bin/dll_test_unit > /dev/null

echo "Release clang (-j2)"
make clean > /dev/null
time make -j2 release_debug/bin/dll_test_unit > /dev/null

echo "Release clang (-j3)"
make clean > /dev/null
time make -j3 release_debug/bin/dll_test_unit > /dev/null

echo "Release clang (-j4)"
make clean > /dev/null
time make -j4 release_debug/bin/dll_test_unit > /dev/null

export CXX=g++-4.9.3

echo "Debug gcc-4.9.3 (-j1)"
make clean > /dev/null
time make -j1 debug/bin/dll_test_unit > /dev/null

echo "Debug gcc-4.9.3 (-j2)"
make clean > /dev/null
time make -j2 debug/bin/dll_test_unit > /dev/null

echo "Debug gcc-4.9.3 (-j3)"
make clean > /dev/null
time make -j3 debug/bin/dll_test_unit > /dev/null

echo "Debug gcc-4.9.3 (-j4)"
make clean > /dev/null
time make -j4 debug/bin/dll_test_unit > /dev/null

echo "Release gcc-4.9.3 (-j1)"
make clean > /dev/null
time make -j1 release_debug/bin/dll_test_unit > /dev/null

echo "Release gcc-4.9.3 (-j2)"
make clean > /dev/null
time make -j2 release_debug/bin/dll_test_unit > /dev/null

echo "Release gcc-4.9.3 (-j3)"
make clean > /dev/null
time make -j3 release_debug/bin/dll_test_unit > /dev/null

echo "Release gcc-4.9.3 (-j4)"
make clean > /dev/null
time make -j4 release_debug/bin/dll_test_unit > /dev/null

export CXX=g++-5.3.0

echo "Debug gcc-5.3.0 (-j1)"
make clean > /dev/null
time make -j1 debug/bin/dll_test_unit > /dev/null

echo "Debug gcc-5.3.0 (-j2)"
make clean > /dev/null
time make -j2 debug/bin/dll_test_unit > /dev/null

echo "Debug gcc-5.3.0 (-j3)"
make clean > /dev/null
time make -j3 debug/bin/dll_test_unit > /dev/null

echo "Debug gcc-5.3.0 (-j4)"
make clean > /dev/null
time make -j4 debug/bin/dll_test_unit > /dev/null

echo "Release gcc-5.3.0 (-j1)"
make clean > /dev/null
time make -j1 release_debug/bin/dll_test_unit > /dev/null

echo "Release gcc-5.3.0 (-j2)"
make clean > /dev/null
time make -j2 release_debug/bin/dll_test_unit > /dev/null

echo "Release gcc-5.3.0 (-j3)"
make clean > /dev/null
time make -j3 release_debug/bin/dll_test_unit > /dev/null

echo "Release gcc-5.3.0 (-j4)"
make clean > /dev/null
time make -j4 release_debug/bin/dll_test_unit > /dev/null

export CXX=/home/wichtounet/Downloads/zapcc_10/bin/zapcc++

echo "Debug zapcc (-j1)"
killall zapccs
make clean > /dev/null
time make -j1 debug/bin/dll_test_unit > /dev/null

echo "Debug zapcc (-j2)"
killall zapccs
make clean > /dev/null
time make -j2 debug/bin/dll_test_unit > /dev/null

echo "Debug zapcc (-j3)"
killall zapccs
make clean > /dev/null
time make -j3 debug/bin/dll_test_unit > /dev/null

echo "Debug zapcc (-j4)"
killall zapccs
make clean > /dev/null
time make -j4 debug/bin/dll_test_unit > /dev/null

echo "Release zapcc (-j1)"
killall zapccs
make clean > /dev/null
time make -j1 release_debug/bin/dll_test_unit > /dev/null

echo "Release zapcc (-j2)"
killall zapccs
make clean > /dev/null
time make -j2 release_debug/bin/dll_test_unit > /dev/null

echo "Release zapcc (-j3)"
killall zapccs
make clean > /dev/null
time make -j3 release_debug/bin/dll_test_unit > /dev/null

echo "Release zapcc (-j4)"
killall zapccs
make clean > /dev/null
time make -j4 release_debug/bin/dll_test_unit > /dev/null
