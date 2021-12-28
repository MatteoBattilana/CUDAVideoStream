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
#define K 5
#define TILE_SIZE 10
#define BLOCK_SIZE (TILE_SIZE + K - 1)

typedef int4 chunk_t;

__device__ uint8_t median(uint8_t *array, int N){
    bool swapped = true;
    for (int a = 0; a < N && swapped; a++){
        swapped = false;
        for (int i = 0; i < N - 1; i++){
            if(array[i] > array[i+1]){
                uint8_t tmp = array[i];
                array[i] = array[i+1];
                array[i+1] = tmp;
                swapped = true;
            }
        }
    }

    return array[N/2];
}

__global__ void median_kernel(uint8_t *image, uint8_t *R)
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

    if(row_o < H && col_o < W && ty < TILE_SIZE && tx < TILE_SIZE){
        uint8_t medR[K*K];
        uint8_t medG[K*K];
        uint8_t medB[K*K];
        int r = 0;
        int g = 0;
        int b = 0;
        for(int i = 0; i < K; i++)
            for(int j = 0; j < K; j++){
                medR[r++] = N_ds[i+ty][(j+tx)*3];
                medG[g++] = N_ds[i+ty][(j+tx)*3 + 1];
                medB[b++] = N_ds[i+ty][(j+tx)*3 + 2];
            }

        R[row_o*W*3 + col_o*3] = median(medR, K*K);
        R[row_o*W*3 + col_o*3 + 1] = median(medG, K*K);
        R[row_o*W*3 + col_o*3 + 2] = median(medB, K*K);
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

int main(int argc, char *argv[]) {
    int threads = 1024;
    if(argc == 2){
        threads = atoi(argv[1]);
    }

    printf("Number of threads set to: %d\n", threads);
    printf("Mask dimension set to: %d\n", K);
    printf("Tile dimension set to: %d\n", TILE_SIZE);
    
    uint8_t *d_current, *d_previous, *d_current_filtered, *d_heat_map;

    cudaMalloc((void **)&d_current, W*H*C * sizeof *d_current);
    cudaMalloc((void **)&d_previous, W*H*C * sizeof *d_previous);
    cudaMalloc((void **)&d_heat_map, W*H*C * sizeof *d_heat_map);
    cudaMalloc((void **)&d_current_filtered, W*H*C * sizeof *d_current_filtered);

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
        median_kernel<<<gridSize, blockSize>>>(d_current, d_current_filtered);
        cudaMemcpy(image1.data, d_current_filtered, W*H*C * sizeof *image1.data, cudaMemcpyDeviceToHost);
        
        cudaMemcpy(d_current, image2.data,  W*H*C * sizeof *image2.data, cudaMemcpyHostToDevice);

        start = std::chrono::high_resolution_clock::now();
        median_kernel<<<gridSize, blockSize>>>(d_current, d_current_filtered);
        

        cudaMemcpy(image2.data, d_current_filtered, W*H*C * sizeof *image2.data, cudaMemcpyDeviceToHost);

        if(a % 100 == 0) imwrite("img_man_g.jpg", image2);

        cudaMemcpy(res.data, d_current_filtered, W*H*C * sizeof *res.data, cudaMemcpyDeviceToHost);
        end = std::chrono::high_resolution_clock::now();

        
        int mod_diff = getCountDifference(image1.data, res.data);
        printf("Filter: %d  ->  %d\n", orig_diff, mod_diff);

        cudaMemcpy(d_previous, image1.data,  W*H*C * sizeof *image1.data, cudaMemcpyHostToDevice);
        kernel<<<1, threads>>>(d_current_filtered, d_previous, (((W*H*C)/threads)/(sizeof(chunk_t))), d_heat_map);
        cudaMemcpy(res.data, d_previous, W*H*C * sizeof *res.data, cudaMemcpyDeviceToHost);
        cudaMemcpy(res2.data, d_heat_map, W*H*C * sizeof *res2.data, cudaMemcpyDeviceToHost);
        if(a % 100 == 0) imwrite("img_man_red_g.jpg", res2);
       
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