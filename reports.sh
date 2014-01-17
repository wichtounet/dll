#!/bin/bash

mkdir -p reports/finals/

for epoch in reports/epoch_*
do
    for file in $epoch/*.dat
    do
        echo "unset colorbox" > test.plt
        echo "unset xtics" >> test.plt
        echo "unset ytics" >> test.plt
        echo "unset border" >> test.plt
        echo "set tmargin 0" >> test.plt
        echo "set tmargin 0" >> test.plt
        echo "set bmargin 0" >> test.plt
        echo "set rmargin 0" >> test.plt
        echo "set lmargin 0" >> test.plt
        echo "set terminal png size 50,50 enhanced font '/usr/share/fonts/liberation-fonts/LiberationSans-Regular.ttf'" >> test.plt
        echo "set output '$file.png'" >> test.plt
        echo "set palette gray" >> test.plt
        echo "plot '$file' binary array=28x28 notitle with image" >> test.plt
        echo "quit" >> test.plt

        gnuplot < test.plt
    done

    montage -mode concatenate -tile 6x6 $epoch/*.dat.png $epoch/final.png

    cp $epoch/final.png reports/finals/`basename $epoch`.png
done

rm test.plt