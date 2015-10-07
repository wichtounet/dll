#!/bin/bash

./release/bin/dll_test --list-test-names-only > tests.tmp

rm -f tests.mk
rm -f tests_list

while read line
do
    if [[ $line != "All available test cases:" ]]
    then
        if [[ $line != *" test cases" ]]
        then
            if [[ $line != "" ]]
            then
                #At this point only test cases are present
                echo $line >> tests_list
            fi
        fi
    fi
done < tests.tmp

while read line
do
    escaped_line=`echo $line | sed 's/\//-/g'`

    echo "debug_test_$line" ': debug_dll_test' >> tests.mk
    echo -e "\t @ mkdir -p test_reports" >> tests.mk
    echo -e "\t @ echo \"Run $line\" > test_reports/test_$escaped_line.log" >> tests.mk
    echo -e "\t @" './debug/bin/dll_test' "$line >> test_reports/test_$escaped_line.log || true" >> tests.mk
    echo >> tests.mk

    echo "release_test_$line" ': release_dll_test' >> tests.mk
    echo -e "\t @ mkdir -p test_reports" >> tests.mk
    echo -e "\t @ echo \"Run $line\" > test_reports/test_$escaped_line.log" >> tests.mk
    echo -e "\t @" './release/bin/dll_test' "$line >> test_reports/test_$escaped_line.log || true" >> tests.mk
    echo >> tests.mk

    echo "release_debug_test_$line" ': release_debug_dll_test' >> tests.mk
    echo -e "\t @ mkdir -p test_reports" >> tests.mk
    echo -e "\t @ echo \"Run $line\" > test_reports/test_$escaped_line.log" >> tests.mk
    echo -e "\t @" './release_debug/bin/dll_test' "$line >> test_reports/test_$escaped_line.log || true" >> tests.mk
    echo >> tests.mk

done < tests_list

echo -n "debug_test_all: " >> tests.mk

while read line
do
    echo -n "debug_test_$line " >> tests.mk
done < tests_list
echo "" >> tests.mk

echo -e "\t @ bash ./tools/test_report.sh" >> tests.mk
echo "" >> tests.mk

echo -n "release_test_all: " >> tests.mk

while read line
do
    echo -n "release_test_$line " >> tests.mk
done < tests_list
echo "" >> tests.mk

echo -e "\t @ bash ./tools/test_report.sh" >> tests.mk
echo "" >> tests.mk

echo -n "release_debug_test_all: " >> tests.mk

while read line
do
    echo -n "release_debug_test_$line " >> tests.mk
done < tests_list
echo "" >> tests.mk

echo -e "\t @ bash ./tools/test_report.sh" >> tests.mk
echo "" >> tests.mk

echo -n ".PHONY: release_test_all debug_test_all release_debug_test_all" >> tests.mk

while read line
do
    echo -n "debug_test_$line release_test_$line release_debug_test_$line" >> tests.mk
done < tests_list
echo "" >> tests.mk

rm tests_list
rm tests.tmp
