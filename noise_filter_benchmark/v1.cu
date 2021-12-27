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
#define W 3
#define H 3
#define C 3
#define TILE_SIZE 3
#define BLOCK_SIZE (TILE_SIZE + K - 1)
#define SIGMA 1

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
        N_ds[ty][tx*C] = image[row_i*W*C+ col_i * C];
        N_ds[ty][tx*C+1] = image[row_i*W*C+ col_i*C + 1 ];
        N_ds[ty][tx*C+2] = image[row_i*W*C+ col_i*C + 2 ];
    } else {
        N_ds[ty][tx*C] = 0;
        N_ds[ty][tx*C+1] = 0;
        N_ds[ty][tx*C+1] = 0;
    }

    __syncthreads();

    if(tx ==0 && ty ==0){
        printf("Shared: \n");
        for (int i = 0; i < BLOCK_SIZE; i++) {
            for (int j = 0; j < BLOCK_SIZE*C; j++) {
                printf("%d ", N_ds[i][j]);
            }
            printf("\n");
        }
        printf("\n");
    }


    float outputR = 0.0;
    float outputG = 0.0;
    float outputB = 0.0;
    if(ty < TILE_SIZE && tx < TILE_SIZE){
        for(int i = 0; i < K; i++)
            for(int j = 0; j < K; j++){
                outputR += dev_k[i*K+j] * N_ds[i+ty][(j+tx)*C];
                outputG += dev_k[i*K+j] * N_ds[i+ty][(j+tx)*C + 1];
                outputB += dev_k[i*K+j] * N_ds[i+ty][(j+tx)*C+2];
            }

        if(row_o < H && col_o < W){
            R[row_o*W*C + col_o*C] = outputR;
            R[row_o*W*C + col_o*C + 1] = outputG;
            R[row_o*W*C + col_o*C + 2] = outputB;
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

    int img[W*H*C] = {1, 0, 120, 2,0,139, 3,0,90, 4,0,99, 5,0,126, 6,0,106, 7,0,46, 8,0,75, 9,0,88};
    int *res = (int*) malloc(H*W*sizeof(int));
        printf("\n\nInitial\n\n");
    for(int i = 0; i < H; i++) {
        for(int j = 0; j < W; j++){
            printf("%d ", img[i*W*C + j*C + 2]);
        }
        printf("\n");
    }
    
    int* dev_i;
    int* dev_r;
    cudaMemcpyToSymbol(dev_k, k, K*K * sizeof(float) );
    cudaMalloc(&dev_i, H*W*C* sizeof(int));
    cudaMalloc(&dev_r, H*W *C* sizeof(int));
    
    cudaMemcpy(dev_i, img, H*W * C *sizeof(int), cudaMemcpyHostToDevice);
        
    dim3 blockSize, gridSize;
    blockSize.x = BLOCK_SIZE, blockSize.y = BLOCK_SIZE, blockSize.z = 1;
    gridSize.x = ceil((float)W/TILE_SIZE),
    gridSize.y = ceil((float)H/TILE_SIZE),
    gridSize.z = 1;
    convolution_kernel<<<gridSize, blockSize>>>(dev_i, dev_r);
    cudaMemcpy(res, dev_r, H*W *C* sizeof(int), cudaMemcpyDeviceToHost);

    
    
    printf("\n\nCUDA\nR:\n");
    for(int i = 0; i < H; i++) {
        for(int j = 0; j < W; j++){
            printf("%d ", res[i*W*C + j*C]);
        }
        printf("\n");
    }

        printf("\n\nCUDA\nG:\n");
    for(int i = 0; i < H; i++) {
        for(int j = 0; j < W; j++){
            printf("%d ", res[i*W*C + j*C + 1]);
        }
        printf("\n");
    }

        printf("\n\nCUDA\nB:\n");
    for(int i = 0; i < H; i++) {
        for(int j = 0; j < W; j++){
            printf("%d ", res[i*W*C + j*C + 2]);
        }
        printf("\n");
    }

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
}
