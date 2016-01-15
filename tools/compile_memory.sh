#!/bin/bash

make clean

max_memory="0.0"
total="0.0"

start=$(date "+%s.%N")

for f in test/src/*.cpp
do
    results=$(/usr/bin/time -v make debug/$f.o 2>&1)
    rss=$(echo "$results" | grep "Maximum resident set size" | rev | cut -d" " -f1 | rev)
    elapsed=$(echo "$results" | grep "Elapsed" | rev | cut -d" " -f1 | rev)
    memory=$(echo "scale=2; $rss/1024" | bc -l)
    echo "$f => $elapsed => ${memory}MB (rss:$rss)"

    if [ $(echo "$max_memory < $memory" | bc) -eq 1 ]
    then
        max_memory=$memory
    fi
done

end=$(date "+%s.%N")

runtime=$(echo "scale=3; ($end - $start) / 1.0" | bc -l)

echo "Max memory: $max_memory"
echo "Total time: $runtime"
