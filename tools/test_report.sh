#!/bin/bash

count=0
total=`ls -l test_reports/ | wc -l`

for file in test_reports/*
do
    if grep -q "All tests passed" $file
    then
        # Do nothing
        echo -n ""
    else
        count=$count+1
        cat $file
    fi

done

if [[ $count == 0 ]]
then
    echo "All tests succeeded"
else
    echo "$count/$total failed test cases"
fi
