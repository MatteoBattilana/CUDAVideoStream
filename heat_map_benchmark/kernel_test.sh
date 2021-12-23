#!/bin/bash
for i in {1..256}
do
    k=$(( 4*i ))
    avg=`sudo /usr/local/cuda/bin/nvprof ./heatMap $k 2>&1`
    all=`echo "$avg" | grep "time generation" | awk '{print $4}'`

    kern=`echo "$avg" | grep kernel | awk '{print $6}'`
    kernt=`echo "$avg" | grep kernel | awk '{print $4}'`
    hd1=`echo "$avg" | grep "CUDA memcpy HtoD" | awk '{print $4}'`
    hd2=`echo "$avg" | grep "CUDA memcpy HtoD" | awk '{print $2}'`
    dh1=`echo "$avg" | grep "CUDA memcpy DtoH" | awk '{print $4}'`
    dh2=`echo "$avg" | grep "CUDA memcpy DtoH" | awk '{print $2}'`
    echo "$k $all $kern $kernt $hd1 $hd2 $dh1 $dh2" >> times
done
