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

#define H 1080
#define W 1920
#define C 3
#define LR_THRESHOLDS 20
#define K 3
#define TILE_SIZE 10
#define BLOCK_SIZE (TILE_SIZE + K - 1)

typedef int4 chunk_t;

__constant__ float dev_k[K*K];

__global__ void convolution_kernel(uint8_t *image, uint8_t *R)
{
    __shared__ uint8_t N_ds[BLOCK_SIZE][BLOCK_SIZE*3];

    int tx = threadIdx.x;   //W
    int ty = threadIdx.y;   //H
    int row_o = blockIdx.y*TILE_SIZE + ty;
    int col_o = blockIdx.x*TILE_SIZE + tx;
    int row_i = row_o - K/2;
    int col_i = col_o - K/2;

    if(row_i >= 0 && row_i < H && col_i >= 0 && col_i < W){
        N_ds[ty][tx*3] = image[row_i*W*3+ col_i * C];
        N_ds[ty][tx*3+1] = image[row_i*W*3+ col_i*3 + 1 ];
        N_ds[ty][tx*3+2] = image[row_i*W*3+ col_i*3 + 2 ];
    } else {
        N_ds[ty][tx*3] = 0;
        N_ds[ty][tx*3+1] = 0;
        N_ds[ty][tx*3+1] = 0;
    }

    __syncthreads();

    if(row_o < H && col_o < W){
        float outputR = 0.0;
        float outputG = 0.0;
        float outputB = 0.0;
        if(ty < TILE_SIZE && tx < TILE_SIZE){
            for(int i = 0; i < K; i++)
                for(int j = 0; j < K; j++){
                    outputR += dev_k[i*K+j] * N_ds[i+ty][(j+tx)*3];
                    outputG += dev_k[i*K+j] * N_ds[i+ty][(j+tx)*3 + 1];
                    outputB += dev_k[i*K+j] * N_ds[i+ty][(j+tx)*3 +2];
                }

                R[row_o*W*3 + col_o*3] = outputR;
                R[row_o*W*3 + col_o*3 + 1] = outputG;
                R[row_o*W*3 + col_o*3 + 2] = outputB;
        }
    }
}

__global__ void kernel(uint8_t *current, uint8_t *previous, int maxSect, uint8_t* d_heat_pixels) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int start = x * maxSect;
    int max = start + maxSect;
    chunk_t cc, pc;

    uint8_t redColor = 0;
    int8_t df;
    for (int i = start; i < max; i++) {

        cc = ((chunk_t *)current)[i];
        pc = ((chunk_t *)previous)[i];
        for (int j = 0; j < sizeof cc; j++) {
            df = ((uint8_t *)&cc)[j] - ((uint8_t *)&pc)[j];

            if (df < -LR_THRESHOLDS || df > LR_THRESHOLDS) {
                redColor = 255;
            }
            
            if((i*(sizeof cc)+j) % 3 == 2){
                d_heat_pixels[i*(sizeof cc)+j] = redColor;
                redColor = 0;
            } else {
                d_heat_pixels[i*(sizeof cc)+j] = 0;
            }
        }
    }
}

int getCountDifference(uint8_t *orig, uint8_t *mod){
    int count = 0;
    for (int i = 0; i < H*W*3; i++){
        if(abs(orig[i] - mod[i]) > 20)
            count++;
    }

    return count;
}

float* computeMeanKernel(){
    float* k = (float*)malloc(K*K*sizeof(float));
    for (int i = 0; i < K; i++){
        for (int j = 0; j < K; j++){
            k[i*K+j] = 1.0/(K*K);
        }
    }

    return k;
}

float* dummyKernel(){
    float* k = (float*)malloc(K*K*sizeof(float));
    for (int i = 0; i < K; i++){
        for (int j = 0; j < K; j++){
            k[i*K+j] = 0;
        }
    }
    k[(K/2)*K + K/2] = 1;

    return k;
}


float* computeGaussianKernel(float sigma){
    float sum = 0;
    float* k = (float*)malloc(K*K*sizeof(float));
    for (int i = 0; i < K; i++){
        for (int j = 0; j < K; j++){
            float x = i - (K - 1) / 2.0;
            float y = j - (K - 1) / 2.0;
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

int main(int argc, char *argv[]) {
    int threads = 1024;
    int sigma = 7;
    if(argc == 2){
        threads = atoi(argv[1]);
    }

    printf("Number of threads set to: %d\n", threads);
    printf("Mask dimension set to: %d\n", K);
    printf("Tile dimension set to: %d\n", TILE_SIZE);
    
    //float k[K*K] = {0,0,0,0,1,0,0,0,0};
    //float k[K*K] = {1.0/9.0, 1.0/9.0, 1.0/9.0,1.0/9.0, 1.0/9.0, 1.0/9.0,1.0/9.0, 1.0/9.0, 1.0/9.0};
    //float k[K*K] = {1.0/16, 2.0/16, 1.0/16, 2.0/16, 1.0/4, 1.0/8, 1.0/16, 1.0/8, 1.0/16};
    //float k[K*K] = {1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0};

    float * k = dummyKernel();
    // float * k = computeGaussianKernel(sigma);
    printf("Kernel: \n");
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            printf("%f ", k[i*K+j]);
        }
        printf("\n");
    }

    uint8_t *d_current, *d_previous, *d_current_filtered, *d_heat_map;

    cudaMalloc((void **)&d_current, W*H*C * sizeof *d_current);
    cudaMalloc((void **)&d_previous, W*H*C * sizeof *d_previous);
    cudaMalloc((void **)&d_heat_map, W*H*C * sizeof *d_heat_map);
    cudaMalloc((void **)&d_current_filtered, W*H*C * sizeof *d_current_filtered);
    cudaMemcpyToSymbol(dev_k, k, K*K * sizeof(float) );

    Mat image1 = imread("f1.jpg");
    Mat image2 = imread("f2.jpg");
    Mat res = imread("f2.jpg");
    Mat res2 = imread("f2.jpg");

    cudaMemcpy(d_previous, image1.data,  W*H*C * sizeof *image1.data, cudaMemcpyHostToDevice);

    dim3 blockSize, gridSize;
    blockSize.x = BLOCK_SIZE, blockSize.y = BLOCK_SIZE, blockSize.z = 1;
    gridSize.x = ceil((float)W/TILE_SIZE),
    gridSize.y = ceil((float)H/TILE_SIZE),
    gridSize.z = 1;

    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    for (int a = 0; a < 50; a++){
        image1 = imread("f1.jpg");
        image2 = imread("f2.jpg");
        res = imread("f2.jpg");
        res2 = imread("f2.jpg");

        
        int orig_diff = getCountDifference(image1.data, image2.data);

        cudaMemcpy(d_current, image1.data,  W*H*C * sizeof *image1.data, cudaMemcpyHostToDevice);
        convolution_kernel<<<gridSize, blockSize>>>(d_current, d_current_filtered);
        cudaMemcpy(image1.data, d_current_filtered, W*H*C * sizeof *image1.data, cudaMemcpyDeviceToHost);
        
        cudaMemcpy(d_current, image2.data,  W*H*C * sizeof *image2.data, cudaMemcpyHostToDevice);

        start = std::chrono::high_resolution_clock::now();
        convolution_kernel<<<gridSize, blockSize>>>(d_current, d_current_filtered);
        

        cudaMemcpy(image2.data, d_current_filtered, W*H*C * sizeof *image2.data, cudaMemcpyDeviceToHost);
        cudaMemcpy(res.data, d_current_filtered, W*H*C * sizeof *res.data, cudaMemcpyDeviceToHost);
        end = std::chrono::high_resolution_clock::now();

        
        int mod_diff = getCountDifference(image1.data, res.data);
        // printf("Filter: %d  ->  %d\n", orig_diff, mod_diff);

        cudaMemcpy(d_previous, image1.data,  W*H*C * sizeof *image1.data, cudaMemcpyHostToDevice);
        kernel<<<1, threads>>>(d_current_filtered, d_previous, (((W*H*C)/threads)/(sizeof(chunk_t))), d_heat_map);
        cudaMemcpy(res.data, d_previous, W*H*C * sizeof *res.data, cudaMemcpyDeviceToHost);
        cudaMemcpy(res2.data, d_heat_map, W*H*C * sizeof *res2.data, cudaMemcpyDeviceToHost);
       
        namedWindow("Original", WINDOW_GUI_NORMAL);
        imshow("Original", res);
        if (waitKey(10) == 27) {
            break;  // stop capturing by pressing ESC
        }
        fflush(stdout);
        namedWindow("HeatMap", WINDOW_GUI_NORMAL);
        imshow("HeatMap", res2);
        if (waitKey(10) == 27) {
            break;  // stop capturing by pressing ESC
        }
        auto elaps = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        printf("\rHeatmap time generation: %.3f ms ", (float)elaps.count() * 1e-6);


    }
    
    return 0;
}