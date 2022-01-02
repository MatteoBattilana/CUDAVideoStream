#include <stdint.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../include/common.h"
#include "../include/kernels.cuh"
#include "../include/utils.hpp"

#define CDIM3(__pdim) (*((dim3 *)__pdim))

using namespace diff::cuda;
using namespace diff::utils;

typedef long4 chunk_t;

__constant__ float dev_k[K*K];

__global__ void convolution_kernel(uint8_t *image, uint8_t *R) {
    __shared__ uint8_t N_ds[BLOCK_SIZE][BLOCK_SIZE*3];

    int tx = threadIdx.x;   //1920
    int ty = threadIdx.y;   //1080
    int row_o = blockIdx.y*TILE_SIZE + ty;
    int col_o = blockIdx.x*TILE_SIZE + tx;
    int row_i = row_o - K/2;
    int col_i = col_o - K/2;

    if(row_i >= 0 && row_i < 1080 && col_i >= 0 && col_i < 1920){
        N_ds[ty][tx*3] = image[row_i*1920*3+ col_i * 3];
        N_ds[ty][tx*3+1] = image[row_i*1920*3+ col_i*3 + 1 ];
        N_ds[ty][tx*3+2] = image[row_i*1920*3+ col_i*3 + 2 ];
    } else {
        N_ds[ty][tx*3] = 0;
        N_ds[ty][tx*3+1] = 0;
        N_ds[ty][tx*3+1] = 0;
    }

    __syncthreads();

    if(row_o < 1080 && col_o < 1920){
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

                R[row_o*1920*3 + col_o*3] = outputR;
                R[row_o*1920*3 + col_o*3 + 1] = outputG;
                R[row_o*1920*3 + col_o*3 + 2] = outputB;
        }
    }
}

__global__ void heat_map(uint8_t *current, uint8_t *previous, int maxSect, uint8_t *noise_visualization){
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int start = x * maxSect;
    int max = start + maxSect;
    chunk_t cc, pc;
    int size = sizeof(chunk_t);
    int df = 0;

    for (int i = start; i < max; i++) {
        cc = ((chunk_t *)current)[i];
        pc = ((chunk_t *)previous)[i];

        for (int j = 0; j < size; j++) {
            df += fabsf(((uint8_t *)&cc)[j] - ((uint8_t *)&pc)[j]);

            if(((i*size)+ j ) % 3 == 2){
                float diff1 = df/(255*2.0);
                int r = fminf(fmaxf(sinf(M_PI*diff1 - M_PI/2.0)*255.0, 0.0),255.0);
                int g = fminf(fmaxf(sinf(M_PI*diff1)*255.0, 0.0),255.0);
                int b = fminf(fmaxf(sinf(M_PI*diff1 + M_PI/2.0)*255.0, 0.0),255.0);
                noise_visualization[i*size+j-2] = b;
                noise_visualization[i*size+j-1] = g;
                noise_visualization[i*size+j] = r;
                df = 0;
            }
        }
    }
}

__global__ void red_black_map(uint8_t *current, uint8_t *previous, int maxSect, uint8_t *noise_visualization){
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int start = x * maxSect;
    int max = start + maxSect;
    chunk_t cc, pc;
    uint8_t redColor = 0;
    int df;
    int size = sizeof(chunk_t);

    for (int i = start; i < max; i++) {
        cc = ((chunk_t *)current)[i];
        pc = ((chunk_t *)previous)[i];

        for (int j = 0; j < size; j++) {
            df = ((uint8_t *)&cc)[j] - ((uint8_t *)&pc)[j];

            if ((df < -LR_THRESHOLDS || df > LR_THRESHOLDS)){
                redColor = 255;
            }
            
            if(((i*size)+ j ) % 3 == 2){
                noise_visualization[(i*size)+j] = redColor;
                redColor = 0;
            }
        }
    }
}

//(d_xs, d_pos, max4, d_noise_visualization);
__global__ void red_black_map_overlap(unsigned int *pos, int *xs, int maxSect, uint8_t *noise_visualization){
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int start = x * maxSect;
    int max = start + maxSect;

    for (int i = start; i < max && i < *pos; i++) {
        noise_visualization[xs[i] + (2-xs[i]%3)] = 255;
    }
}


/*
Optimizations done:
 - compute difference int by int instead of uint8 by uint8
 - same stuff on updating current[i]
 - vectorized instructions (int2 or greater)
*/
__global__ void kernel2(uint8_t *current, uint8_t *previous, uint8_t *diff, int maxSect, unsigned int *pos, int *xs) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int npos;
    int df;
    chunk_t cc, pc;
    bool currUpdateRequired = false;

    if (!x) *pos = 0;
    __syncthreads();

    int start = x * maxSect;
    int max = start + maxSect;

    #pragma unroll
    for (int i = start; i < max; i++) {

        cc = ((chunk_t *)current)[i];
        pc = ((chunk_t *)previous)[i];

        #pragma unroll
        for (int j = 0; j < sizeof cc; j++) {
            df = ((uint8_t *)&cc)[j] - ((uint8_t *)&pc)[j];
            if (df < -LR_THRESHOLDS || df > LR_THRESHOLDS) {
                npos = atomicInc(pos, 6220801);
                diff[npos] = df;
                xs[npos] = (i*sizeof cc) + j;
            } else {

#ifdef KERNEL2_NEGFEED_OPT
                ((uint8_t *)&cc)[j] -= df;
                currUpdateRequired = true;
#else 
                current[(i*sizeof cc) + j] -= df;
#endif
            }
        }

#ifdef KERNEL2_NEGFEED_OPT
        if (currUpdateRequired) {
            ((chunk_t *)current)[i] = cc;
            currUpdateRequired = false;
        }
#endif

    }

}

// access byte by byte
__global__ void kernel_char(uint8_t *current, uint8_t *matrix, int N, int offset, int matrixWidth, int currWidth) {
    int thid = threadIdx.x + blockIdx.x * blockDim.x;

    int start = thid * N;
    int max = start + N;

    for (int i = start; i < max; i++) {
        int x = offset + i % matrixWidth;
        int y = i / matrixWidth;
        current[y * currWidth + x] = matrix[i];
    } 

}

// access chunk_t by chunk_t
__global__ void kernel2_char(uint8_t *current, uint8_t *matrix, int N, int offset, int matrixWidth, int currWidth) {
    int thid = threadIdx.x + blockIdx.x * blockDim.x;
    // chunk_t cc;

    int start = thid * N;
    int max = start + N;

    for (int i = start; i < max; i++) {

        int reali = i * sizeof(chunk_t);
        int x = offset + reali % matrixWidth;
        int y = reali / matrixWidth;

        int idx = (y * currWidth + x) / sizeof(chunk_t);
        // cc = ((chunk_t *)current)[idx];

        // #pragma unroll
        // for (int j = 0; j < sizeof cc; j++) {
        //     ((uint8_t *)&cc)[j] = matrix[reali + j];
        // }

        ((chunk_t *)current)[idx] = ((chunk_t *)matrix)[i];
        // ((chunk_t *)current)[idx] = cc;
    } 

}

diff::cuda::CUDACore::CUDACore(uint8_t *charsPx, matsz& charsSz, float *k, int total, uint8_t *sampleMatData, matsz& frameSz) {

    this->fullArea = 3 * charsSz.area();
    int totcpy = fullArea * sizeof *this->d_charsPx * (sizeof(CHARS_STR) - 1);
    cudaMalloc((void **)&this->d_charsPx, totcpy);
    cudaMemcpy(this->d_charsPx, charsPx, totcpy, cudaMemcpyHostToDevice);

    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    this->total = total;
    this->charsSz = charsSz;
    this->frameSz = frameSz;

    cudaMemcpyToSymbol(dev_k, k, K*K * sizeof(float));
    cudaMalloc((void **)&d_diff, total * sizeof *d_diff);
    cudaMalloc((void **)&d_xs, total * sizeof *d_xs);
    cudaMalloc((void **)&d_current, total * sizeof *d_current);
    cudaMalloc((void **)&d_previous, total * sizeof *d_previous);
    cudaMalloc((void **)&d_filtered, total * sizeof *d_filtered);
    cudaMalloc((void **)&d_noise_visualization, total * sizeof *d_noise_visualization);

    cudaMalloc((void **)&d_pos, sizeof *d_pos);

	nMaxThreads = prop.maxThreadsPerBlock;
    maxAtTime = total / nMaxThreads;
    cudaMemcpy(d_current, sampleMatData, total * sizeof *sampleMatData, cudaMemcpyHostToDevice);

    max4 = ceil(1.0 * maxAtTime / sizeof(chunk_t));

    dim3 blockSize, gridSize;
    blockSize.x = BLOCK_SIZE, blockSize.y = BLOCK_SIZE, blockSize.z = 1;
    gridSize.x = ceil((float)1920/TILE_SIZE),
    gridSize.y = ceil((float)1080/TILE_SIZE),
    gridSize.z = 1;

    this->pblockSize = new dim3(blockSize);
    this->pgridSize = new dim3(gridSize);

    for (int i = nMaxThreads; i > 0; i--) {
        float frac = fullArea / (i * 1.0);
        int frac_int = fullArea / i;
        if ((int)ceil(frac) == frac_int && frac_int % sizeof(chunk_t) == 0) {
            nThreadsToUse = i;
            eachThreadDoes = frac_int / sizeof(chunk_t);
            break;
        }
    }

}

void diff::cuda::CUDACore::exec_core(uint8_t *frameData, uint8_t *showReadyNData, std::string& text, unsigned int *h_pos, int *h_xs) {

    /******************** ********************/
    /* GPU NAIF
    /******************** ********************/
    // cudaMemset(d_pos, 0, sizeof *d_pos);

    // cudaMemcpy(d_previous, previous.data, total, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_current, pready->pframe->data, total, cudaMemcpyHostToDevice);
    // kernel<<<1, nMaxThreads, 0>>>(d_current, d_previous, d_diff, maxAtTime, d_pos, d_xs);
    // cudaMemcpy(pready->pframe->data, d_diff, total, cudaMemcpyDeviceToHost);
    // cudaMemcpy(pready->h_xs, d_xs, total * sizeof *d_xs, cudaMemcpyDeviceToHost);
    // cudaMemcpy(&pready->h_pos, d_pos, sizeof *d_pos, cudaMemcpyDeviceToHost);

    // cudaMemcpy(previous.data, d_current, total, cudaMemcpyDeviceToHost);




    /******************** ********************/
    /* GPU NAIF - async version and no previous copy
    /******************** ********************/

    // current-previous swap
    diff::utils::swap(d_current, d_previous);

    // cudaMemsetAsync(d_pos, 0, sizeof *d_pos);

    // Copy in the current pointer and run 

    #ifdef NOISE_FILTER
    cudaMemcpyAsync(d_filtered, frameData, total, cudaMemcpyHostToDevice);
    convolution_kernel<<<CDIM3(pgridSize), CDIM3(pblockSize)>>>(d_filtered, d_current);
    #else
    cudaMemcpyAsync(d_current, frameData, total, cudaMemcpyHostToDevice);
    #endif


    // cudaMemcpy(d_text, overImageText.c_str(), overImageText.length(), cudaMemcpyHostToDevice);
    // kernel3_char<<<1, nThreadsToUse, (sizeof(CHARS_STR) - 1) * fullArea>>>(d_current, d_charsPx, d_text, overImageText.length(), eachThreadDoes, 3 * txtsz.width, 3 * pready->pframe->cols, fullArea);

    for (int offset = 0, j = 0; j < text.length(); j++, offset += charsSz.width*3) {
        int idx;
        for (int i = 0; i < (sizeof(CHARS_STR) - 1); i++) {
            if (CHARS_STR[i] == text.at(j)) {
                idx = i;
                break;
            }
        }

        kernel2_char<<<1, nThreadsToUse>>>(d_current, d_charsPx + idx * fullArea, eachThreadDoes, offset, 3 * charsSz.width, 3 * frameSz.width);
    }

    // kernel<<<1, nMaxThreads, 0>>>(d_current, d_previous, d_diff, maxAtTime, d_pos, d_xs);
    kernel2<<<1, nMaxThreads, 0>>>(d_current, d_previous, d_diff, max4, d_pos, d_xs);

    cudaMemcpyAsync(h_pos, d_pos, sizeof *d_pos, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Noise visualization
    #ifdef NOISE_VISUALIZER
        #if NOISE_VISUALIZER == 1         
        heat_map<<<1, nMaxThreads, 0>>>(d_current, d_previous, max4, d_noise_visualization);
        cudaMemcpyAsync(showReadyNData, d_noise_visualization, total, cudaMemcpyDeviceToHost);
        #elif NOISE_VISUALIZER == 2
        red_black_map<<<1, nMaxThreads, 0>>>(d_current, d_previous, max4, d_noise_visualization);
        cudaMemcpyAsync(showReadyNData, d_noise_visualization, total, cudaMemcpyDeviceToHost);
        #elif NOISE_VISUALIZER == 3
        red_black_map_overlap<<<1, nMaxThreads, 0>>>(d_pos, d_xs, (*h_pos)/nMaxThreads, d_previous);
        cudaMemcpyAsync(showReadyNData, d_previous, total, cudaMemcpyDeviceToHost);
        #endif
    #endif

    cudaMemcpyAsync(frameData, d_diff, *h_pos, cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(h_xs, d_xs, *h_pos * sizeof *d_xs, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

}

size_t diff::cuda::CUDACore::chunkt_size() {
    return sizeof(chunk_t);
}

void diff::cuda::CUDACore::alloc_arrays(uint8_t **h_frame, uint8_t **n_frame, uint8_t **o_frame, int **h_xs, int r, int c) {
    cudaMallocHost((void **)h_frame, 3 * r * c * sizeof **h_frame + sizeof(chunk_t));
    cudaMallocHost((void **)n_frame, 3 * r * c * sizeof **n_frame + sizeof(chunk_t));
    cudaMallocHost((void **)o_frame, 3 * r * c * sizeof **o_frame + sizeof(chunk_t));
    cudaMallocHost((void **)h_xs, 3 * r * c * sizeof **h_xs + sizeof(chunk_t));
}
