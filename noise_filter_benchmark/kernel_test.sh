#!/bin/bash
for i in {1..256}
do
    k=$(( 4*i ))
    `/usr/local/cuda/bin/nvcc -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio -lpthread -L/usr/local/lib -lv4l2  -std=c++11 -g -lineinfo --use_fast_math -I/usr/include/opencv4 v2.cu -o noiseFilter -DTILE_SIZE=$k`
    avg=`sudo /usr/local/cuda/bin/nvprof ./noiseFilter $k 2>&1`
    all=`echo "$avg" | grep "time generation" | awk '{print $4}'`

    kern=`echo "$avg" | grep convolution_kernel | awk '{print $6}'`
    kernt=`echo "$avg" | grep convolution_kernel | awk '{print $4}'`
    hd1=`echo "$avg" | grep "CUDA memcpy HtoD" | awk '{print $4}'`
    hd2=`echo "$avg" | grep "CUDA memcpy HtoD" | awk '{print $2}'`
    dh1=`echo "$avg" | grep "CUDA memcpy DtoH" | awk '{print $4}'`
    dh2=`echo "$avg" | grep "CUDA memcpy DtoH" | awk '{print $2}'`
    echo "$k $all $kern $kernt $hd1 $hd2 $dh1 $dh2" >> times_k3
done
