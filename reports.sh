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

        gnuplot < test.plt > /dev/null

        echo "unset ytics" > test.plt
        echo "set terminal png size 250,250 enhanced font '/usr/share/fonts/liberation-fonts/LiberationSans-Regular.ttf'" >> test.plt
        echo "binwidth=0.01" >> test.plt
        echo "bin(x,width)=width*floor(x/width)" >> test.plt

        #echo "stats '$epoch/hiddens.dat'" >> test.plt
        #echo "set xtics STATS_min_y,(abs(STATS_max_y)+abs(STATS_min_y))/2,STATS_max_y" >> test.plt
        echo "set output '$epoch/hiddens.png'" >> test.plt
        echo "plot '$epoch/hiddens.dat' using (bin(\$1,binwidth)):(1.0) smooth freq notitle with boxes" >> test.plt

        echo "binwidth=0.001" >> test.plt

        #echo "stats '$epoch/weights.dat'" >> test.plt
        #echo "set xtics STATS_min_y,(abs(STATS_max_y)+abs(STATS_min_y))/2,STATS_max_y" >> test.plt
        echo "set output '$epoch/weights.png'" >> test.plt
        echo "plot '$epoch/weights.dat' using (bin(\$1,binwidth)):(1.0) smooth freq notitle with boxes" >> test.plt

        #echo "stats '$epoch/visibles.dat'" >> test.plt
        #echo "set xtics STATS_min_y,(abs(STATS_max_y)+abs(STATS_min_y))/2,STATS_max_y" >> test.plt
        echo "set output '$epoch/visibles.png'" >> test.plt
        echo "plot '$epoch/visibles.dat' using (bin(\$1,binwidth)):(1.0) smooth freq notitle with boxes" >> test.plt

        echo "binwidth=0.0001" >> test.plt

        #echo "stats '$epoch/hiddens_inc.dat'" >> test.plt
        #echo "set xtics STATS_min_y,(abs(STATS_max_y)+abs(STATS_min_y))/2,STATS_max_y" >> test.plt
        echo "set output '$epoch/hiddens_inc.png'" >> test.plt
        echo "plot '$epoch/hiddens_inc.dat' using (bin(\$1,binwidth)):(1.0) smooth freq notitle with boxes" >> test.plt

        #echo "stats '$epoch/weights_inc.dat'" >> test.plt
        #echo "set xtics STATS_min_y,(abs(STATS_max_y)+abs(STATS_min_y))/2,STATS_max_y" >> test.plt
        echo "set output '$epoch/weights_inc.png'" >> test.plt
        echo "plot '$epoch/weights_inc.dat' using (bin(\$1,binwidth)):(1.0) smooth freq notitle with boxes" >> test.plt

        #echo "stats '$epoch/visibles_inc.dat'" >> test.plt
        #echo "set xtics STATS_min_y,(abs(STATS_max_y)+abs(STATS_min_y))/2,STATS_max_y" >> test.plt
        echo "set output '$epoch/visibles_inc.png'" >> test.plt
        echo "plot '$epoch/visibles_inc.dat' using (bin(\$1,binwidth)):(1.0) smooth freq notitle with boxes" >> test.plt
        echo "quit" >> test.plt

        gnuplot < test.plt > /dev/null 2> /dev/null
    done

    montage -mode concatenate -tile 10x10 $epoch/h_*.dat.png $epoch/hiddens_weights.png
    montage -mode concatenate -tile 3x2 $epoch/hiddens.png $epoch/weights.png $epoch/visibles.png $epoch/hiddens_inc.png $epoch/weights_inc.png $epoch/visibles_inc.png $epoch/histograms.png

    cp $epoch/hiddens_weights.png reports/finals/hiddens_weights_`basename $epoch`.png
    cp $epoch/histograms.png reports/finals/histograms_`basename $epoch`.png
done

rm test.plt