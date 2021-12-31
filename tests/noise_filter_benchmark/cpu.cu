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
#include "v4l.h"
#include <cmath>

using namespace cv;

#define K 3
#define W 1920
#define H 1080
#define C 3
#define TILE_SIZE 10
#define BLOCK_SIZE (TILE_SIZE + K - 1)
#define SIGMA 5

__constant__ float dev_k[K*K];

__global__ void convolution_kernel(int *image, int *R)
{
    __shared__ int N_ds[BLOCK_SIZE][BLOCK_SIZE*C];

    int tx = threadIdx.x;   //W
    int ty = threadIdx.y;   //H
    int row_o = blockIdx.y*TILE_SIZE + ty;
    int col_o = blockIdx.x*TILE_SIZE + tx;
    int row_i = row_o - K/2;
    int col_i = col_o - K/2;

    if(row_i >= 0 && row_i < H && col_i >= 0 && col_i < W){
        N_ds[ty][tx*C] = image[row_i*W*C + col_i*C];
        N_ds[ty][tx*C+1] = image[row_i*W*C + col_i*C + 1];
        N_ds[ty][tx*C+2] = image[row_i*W*C + col_i*C + 2];
    } else {
        N_ds[ty][tx*C] = 0;
        N_ds[ty][tx*C+1] = 0;
        N_ds[ty][tx*C+2] = 0;
    }

    __syncthreads();


    for (int color = 0; color < C; color++){
        int output = 0;
        if(ty < TILE_SIZE && tx < TILE_SIZE){
            for(int i = 0; i < K; i++)
                for(int j = 0; j < K; j++){
                    output += dev_k[i*K+j] * N_ds[i+ty][(j+tx)*C+color];
                }

            if(row_o < H && col_o < W){
                R[row_o*W*C + col_o*C + color] = output;
            }
        }
    }
}

void apply_filter(float* k, int* y, int* res, int i, int j){
    // i is the vertical position
    // j is the horizontal position
    for (int color = 0; color < C; color++){
        int ver_start_position = i - K/2;
        int sum = 0;
        for (int c = 0; c < K; c++){
            int hor_start_position = j - K/2;
            for(int d = 0; d < K; d++){
                if(hor_start_position >= 0 && hor_start_position < W && ver_start_position >= 0 && ver_start_position < H){
                    sum += y[(hor_start_position + ver_start_position*W)*C + color ] * k[c*K + d];
                }
                hor_start_position++;
            }
            ver_start_position++;
        }
        res[(i*W + j)*C + color] = sum;
    }
}

void convolution(float* k, int* y, int* res){
    for(int i = 0; i < H; i++) {
        for(int j = 0; j < W; j++){
            apply_filter(k, y, res, i, j);
        }
    }
}

float* computeGaussianKernel(float sigma){
    double sum = 0;
    float* k = (float*)malloc(K*K*sizeof(float));
    for (int i = 0; i < K; i++){
        for (int j = 0; j < K; j++){
            double x = i - (K - 1) / 2.0;
            double y = j - (K - 1) / 2.0;
            k[i*K+j] = (1.0/(2.0*M_PI*sigma*sigma)) * exp(-((x*x + y*y)/(2.0*sigma*sigma)));
            sum += k[i*K+j];
        }
    }

    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            k[i*K+j] /= sum;
        }
    }


    return k;
}

int main(void)
{
    Mat imageCV;
    //float * k = computeGaussianKernel(SIGMA);
    float k[K*K] = {1.0/9.0, 1.0/9.0, 1.0/9.0,1.0/9.0, 1.0/9.0, 1.0/9.0,1.0/9.0, 1.0/9.0, 1.0/9.0};
    //float k[K*K] = {1.0/16, 2.0/16, 1.0/16, 2.0/16, 1.0/4, 1.0/8, 1.0/16, 1.0/8, 1.0/16};
    printf("Kernel: \n");
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            printf("%f ", k[i*K+j]);
        }
        printf("\n");
    }

    int *img = (int*) malloc(H*W*C*sizeof(int));
    int *res = (int*) malloc(H*W*C*sizeof(int));
    

    VideoCapture cap;
    if (!cap.open(0, CAP_V4L2)) return 1;
    auto codec = cv::VideoWriter::fourcc('M','J','P','G');
    cap.set(cv::CAP_PROP_FOURCC, codec);

    cap.set(CAP_PROP_FRAME_WIDTH, W);
    cap.set(CAP_PROP_FRAME_HEIGHT,H);

    while (1) {
        cap >> imageCV;
        Mat imageCVMan = imageCV.clone();



    for (int y = 0; y < H; y++){
        for (int x = 0; x < W; x++){
            Vec3b intensity = imageCV.at<Vec3b>(y, x);
            uchar blue = intensity.val[0];
            uchar green = intensity.val[1];
            uchar red = intensity.val[2];
            img[x*C + y*W*C] = (int) red;
            img[x*C + y*W*C + 1] = (int) green;
            img[x*C + y*W*C + 2] = (int) blue;
        }
    }

    // // convolution(k, img, res);

    // // for(int i = 0; i < H; i++) {
    // //     for(int j = 0; j < W; j++){
    // //         printf("%d ", res[i*W + j*C]);
    // //     }
    // //     printf("\n");
    // // }

    // // printf("\n");
    // // for(int i = 0; i < H; i++) {
    // //     for(int j = 0; j < W; j++){
    // //         printf("%d ", res[i*W + j*C + 1]);
    // //     }
    // //     printf("\n");
    // // }

    // // printf("\n");
    // // for(int i = 0; i < H; i++) {
    // //     for(int j = 0; j < W; j++){
    // //         printf("%d ", res[i*W + j*C + 2]);
    // //     }
    // //     printf("\n");
    // // }

    int * res2 = (int*)malloc(H*W*C*sizeof(int));
    int* dev_i;
    int* dev_r;
    cudaMemcpyToSymbol(dev_k, k, K*K * sizeof(float) );
    cudaMalloc(&dev_i, H*W*C * sizeof(int));
    cudaMalloc(&dev_r, H*W*C * sizeof(int));
    

        auto t_start = std::chrono::high_resolution_clock::now();
        cudaMemcpy(dev_i, img, H*W*C * sizeof(int), cudaMemcpyHostToDevice);
        
        dim3 blockSize, gridSize;
        blockSize.x = BLOCK_SIZE, blockSize.y = BLOCK_SIZE, blockSize.z = 1;
        gridSize.x = ceil((float)W/TILE_SIZE),
        gridSize.y = ceil((float)H/TILE_SIZE),
        gridSize.z = 1;
        convolution_kernel<<<gridSize, blockSize>>>(dev_i, dev_r);
        cudaMemcpy(res2, dev_r, H*W*C * sizeof(int), cudaMemcpyDeviceToHost);
        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();

        printf("Elp: %f\n", elapsed_time_ms);
    
    
    // // printf("\n\nCUDA\nR:\n");
    // // for(int i = 0; i < H; i++) {
    // //     for(int j = 0; j < W; j++){
    // //         printf("%d ", res2[i*W + j*C]);
    // //     }
    // //     printf("\n");
    // // }

    // // printf("G: \n");
    // // for(int i = 0; i < H; i++) {
    // //     for(int j = 0; j < W; j++){
    // //         printf("%d ", res2[i*W + j*C + 1]);
    // //     }
    // //     printf("\n");
    // // }

    // // printf("B: \n");
    // // for(int i = 0; i < H; i++) {
    // //     for(int j = 0; j < W; j++){
    // //         printf("%d ", res2[i*W + j*C + 2]);
    // //     }
    // //     printf("\n");
    // // }


    for (int y = 0; y < H; y++){
        for (int x = 0; x < W; x++){
            Vec3b & intensity = imageCVMan.at<Vec3b>(y, x);
            intensity.val[2] = 0;
            intensity.val[1] = 0;
            intensity.val[0] = 0;
        }
    }

    for (int y = 0; y < H; y++){
        for (int x = 0; x < W; x++){
            Vec3b & intensity = imageCVMan.at<Vec3b>(y, x);
            intensity.val[2] = max(min(255,res2[x*C + y*W*C]),0);
            intensity.val[1] = max(min(255,res2[x*C + y*W*C + 1]),0);
            intensity.val[0] = max(min(255,res2[x*C + y*W*C + 2]),0);
            //printf("%d %d %d  ", res[x*W + y*C], res[x*W + y*C + 1], res[x*W + y*C + 2]);
        }
    }


        namedWindow("hi", WINDOW_GUI_NORMAL);
        imshow("hi", imageCVMan);
        if (waitKey(10) == 27) break;  // stop capturing by pressing ESC
    }
}
