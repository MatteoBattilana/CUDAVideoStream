#include <stdio.h>
#include <fcntl.h>    /* For O_RDWR */
#include <sys/ioctl.h>
#include <unistd.h>   /* For open(), creat() */
#include <netdb.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <unistd.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <time.h>
#include <stdint.h>
#include "opencv2/opencv.hpp"
#include <pthread.h>
#include <cmath>
using namespace cv;

#define SINE 
#define H 1080
#define W 1920
#define C 3

__global__ void kernel(uint8_t *current, uint8_t *previous, int maxSect, uint8_t* d_heat_pixels) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int start = x * maxSect;
    int max = start + maxSect;
    for (int i = start; i < max; i=i+C) {
        int pixelDiff = fabsf(current[i] - previous[i]) + fabsf(current[i+1] - previous[i+1]) + fabsf(current[i+2] - previous[i+2]); 
        #ifdef SINE
        float diff1 = pixelDiff/(255*2.0);
        int r = fminf(fmaxf(sinf(M_PI*diff1 - M_PI/2.0)*255.0, 0.0),255.0);
        int g = fminf(fmaxf(sinf(M_PI*diff1)*255.0, 0.0),255.0);
        int b = fminf(fmaxf(sinf(M_PI*diff1 + M_PI/2.0)*255.0, 0.0),255.0);
        d_heat_pixels[i] = b;
        d_heat_pixels[i+1] = g;
        d_heat_pixels[i+2] = r;
        #else
        d_heat_pixels[i+2] = pixelDiff > 30 ? 255: 0;
        d_heat_pixels[i+1] = 0;
        d_heat_pixels[i] = 0;
        #endif
    }
}
   

int main() {
    uint8_t *d_current, *d_previous;
    uint8_t *d_heat_pixels;

    cudaMalloc((void **)&d_current, W*H*C * sizeof *d_current);
    cudaMalloc((void **)&d_previous, W*H*C * sizeof *d_previous);
    cudaMalloc((void **)&d_heat_pixels, W*H*C * sizeof *d_heat_pixels);

    Mat image1, image2, res;
    VideoCapture cap;
    if (!cap.open("/dev/video0")) return 1;
    auto codec = cv::VideoWriter::fourcc('M','J','P','G');
    cap.set(cv::CAP_PROP_FOURCC, codec);
    cap.set(3, W);
    cap.set(4, H);
    cap >> image1;
    res = image1.clone();

    while (1) {
        cap >> image2;
        
        namedWindow("Original", WINDOW_GUI_NORMAL);
        imshow("Original", image2);
        if (waitKey(10) == 27) {
            break;  // stop capturing by pressing ESC
        }


        auto start = std::chrono::high_resolution_clock::now();
        
        cudaMemcpy(d_previous, image1.data,  W*H*C * sizeof *image1.data, cudaMemcpyHostToDevice);
        cudaMemcpy(d_current, image2.data,  W*H*C * sizeof *image2.data, cudaMemcpyHostToDevice);
        kernel<<<1, 300>>>(d_current, d_previous, (W*H*C)/300, d_heat_pixels);
        cudaMemcpy(res.data, d_heat_pixels, W*H*C * sizeof *res.data, cudaMemcpyDeviceToHost);

        auto end = std::chrono::high_resolution_clock::now();
        auto elaps = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        printf("\rHeatmap time generation: %.3f ms", (float)elaps.count() * 1e-6);
        fflush(stdout);

        namedWindow("HeatMap", WINDOW_GUI_NORMAL);
        imshow("HeatMap", res);
        if (waitKey(10) == 27) {
            break;  // stop capturing by pressing ESC
        }

        image1 = image2.clone();
    }
    
    return 0;
}