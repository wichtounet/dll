#!/bin/bash

mkdir -p reports/finals/

for epoch in reports/epoch_*
do
    for file in $epoch/h_*.dat
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
        #echo "set palette gray" >> test.plt
        echo "plot '$file' binary array=28x28 notitle with image" >> test.plt
        echo "quit" >> test.plt

        gnuplot < test.plt

        echo "unset ytics" > test.plt
        echo "set terminal png size 200,200 enhanced font '/usr/share/fonts/liberation-fonts/LiberationSans-Regular.ttf'" >> test.plt
        echo "set output '$epoch/weights.png'" >> test.plt
        echo "binwidth=0.01" >> test.plt
        echo "bin(x,width)=width*floor(x/width)" >> test.plt
        echo "plot '$epoch/weights.dat' using (bin(\$1,binwidth)):(1.0) smooth freq notitle with boxes" >> test.plt
        echo "set output '$epoch/hiddens.png'" >> test.plt
        echo "plot '$epoch/hiddens.dat' using (bin(\$1,binwidth)):(1.0) smooth freq notitle with boxes" >> test.plt
        echo "set output '$epoch/visibles.png'" >> test.plt
        echo "plot '$epoch/visibles.dat' using (bin(\$1,binwidth)):(1.0) smooth freq notitle with boxes" >> test.plt
        echo "quit" >> test.plt

        gnuplot < test.plt
    done

    montage -mode concatenate -tile 6x6 $epoch/h_*.dat.png $epoch/hiddens_weights.png
    montage -mode concatenate -tile 3x  $epoch/hiddens.png $epoch/weights.png $epoch/visibles.png $epoch/histograms.png

    cp $epoch/hiddens_weights.png reports/finals/hiddens_weights_`basename $epoch`.png
    cp $epoch/histograms.png reports/finals/histograms_`basename $epoch`.png
done

rm test.plt