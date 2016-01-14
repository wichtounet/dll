#!/bin/bash

make clean

for f in test/src/*.cpp
do
    results=$(/usr/bin/time -v make debug/$f.o 2>&1)
    rss=$(echo "$results" | grep "Maximum resident set size" | rev | cut -d" " -f1 | rev)
    elapsed=$(echo "$results" | grep "Elapsed" | rev | cut -d" " -f1 | rev)
    memory=$(echo "scale=2; $rss/1024" | bc -l)
    echo "$f => $elapsed => ${memory}MB (rss:$rss)"
done
