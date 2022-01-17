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
#include <cstdlib>
using namespace cv;

int B;
int C;
int S;
int G;

VideoCapture cap;

void onTrackbar_changed(int, void*)
{
    cap.set(cv::CAP_PROP_BRIGHTNESS, float(B));
    cap.set(cv::CAP_PROP_CONTRAST, float(C));
    cap.set(cv::CAP_PROP_SATURATION, float(S));
    cap.set(cv::CAP_PROP_GAIN, float(G));
}


#define H 1080
#define W 1920
#define LR_THRESHOLDS 20

typedef long4 chunk_t;


__global__ void kernel(uint8_t *current, uint8_t *previous, int maxSect, uint8_t *noise_visualization){
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int start = x * maxSect;
    int max = start + maxSect;
    chunk_t cc, pc;
    bool redColor = false;
    int df;
    int size = sizeof(chunk_t);

    for (int i = start; i < max; i++) {
        cc = ((chunk_t *)current)[i];
        pc = ((chunk_t *)previous)[i];

        for (int j = 0; j < size; j++) {
            df = ((uint8_t *)&cc)[j] - ((uint8_t *)&pc)[j];

            if ((df < -LR_THRESHOLDS || df > LR_THRESHOLDS)){
                redColor = true;
            }
            
            if(redColor && ((i*size)+ j ) % 3 == 2){
                noise_visualization[(i*size)+j] = 255;
                redColor = false;
            }
        }
    }
}
   

int main(int argc, char *argv[]) {

    Mat image1, image2, res;
    if (!cap.open("/dev/video0")) return 1;
    auto codec = cv::VideoWriter::fourcc('M','J','P','G');
    cap.set(cv::CAP_PROP_FOURCC, codec);
    cap.set(cv::CAP_PROP_BRIGHTNESS, 10);
    cap.set(3, W);
    cap.set(4, H);
    B=cap.get(cv::CAP_PROP_BRIGHTNESS);
    C=cap.get(cv::CAP_PROP_CONTRAST );
    S=cap.get(cv::CAP_PROP_SATURATION);
    G=cap.get(cv::CAP_PROP_GAIN);

    namedWindow("Original", WINDOW_GUI_EXPANDED);
    createTrackbar( "Brightness","Original", &B, 100, onTrackbar_changed );
    createTrackbar( "Contrast","Original", &C, 100,onTrackbar_changed );
    createTrackbar( "Saturation","Original", &S, 100,onTrackbar_changed);
    createTrackbar( "Gain","Original", &G, 100,onTrackbar_changed);

    cap >> image1;
    res = image1.clone();

    int threads = 1024;
    if(argc == 2){
        threads = atoi(argv[1]);
    }
    printf("Number of threads set to: %d\n", threads);


    uint8_t *d_current, *d_previous;
    uint8_t *d_heat_pixels;

    cudaMalloc((void **)&d_current, W*H*3 * sizeof *d_current);
    cudaMalloc((void **)&d_previous, W*H*3 * sizeof *d_previous);
    cudaMalloc((void **)&d_heat_pixels, W*H*3 * sizeof *d_heat_pixels);

    cudaMemcpy(d_previous, image1.data,  W*H*3 * sizeof *image1.data, cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    for (int i = 0;  i < 20; i++){
        cap >> image2;
        
        // imshow("Original", image2);
        // if (waitKey(10) == 27) {
        //     break;  // stop capturing by pressing ESC
        // }
        
        uint8_t* tmp = d_current;
        d_current = d_previous;
        d_previous = tmp;

        start = std::chrono::high_resolution_clock::now();
        cudaMemset ( d_heat_pixels , 0 , W*H*3 ) ;
        
        cudaMemcpy(d_current, image2.data,  W*H*3 * sizeof *image2.data, cudaMemcpyHostToDevice);
        kernel<<<1, threads>>>(d_current, d_previous, (((W*H*3)/threads)/(sizeof(chunk_t))), d_heat_pixels);
        cudaMemcpy(res.data, d_heat_pixels, W*H*3 * sizeof *res.data, cudaMemcpyDeviceToHost);

        end = std::chrono::high_resolution_clock::now();
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