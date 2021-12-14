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
#include <pthread.h>
#include "v4l.h"

__global__ void kernel(unsigned int *pos, uint8_t *xs, uint8_t *diff) {
    int tid = threadIdx.x;
    int npos;
    
    for (int i = 2 * tid; i < 2 * tid + 2; i++) {
        npos = atomicInc(pos, 20);
        diff[npos] = i;
        xs[npos] = i;
    }
}

__global__ void kernel2(unsigned int *pos, uint8_t *xs, uint8_t *diff) {
    int tid = threadIdx.x;
    int npos;
    
    for (int i = 2 * tid; i < 2 * tid + 2; i++) {
        // npos = atomicInc(pos, 20);
        diff[i] = i;
        xs[i] = i;
    }
}

int main() {

    unsigned int *d_pos, *h_pos;

    uint8_t *d_xs, *d_diff;
    cudaMalloc((void **)&d_xs, 20);
    cudaMalloc((void **)&d_diff, 20);
    cudaMalloc((void **)&d_pos, sizeof *d_pos);

    uint8_t *h_xs, *h_diff;
    uint8_t *h_xs2, *h_diff2;
    cudaMallocHost((void **)&h_xs, 20);
    cudaMallocHost((void **)&h_diff, 20);
    cudaMallocHost((void **)&h_pos, sizeof *h_pos);
    cudaMallocHost((void **)&h_xs2, 20);
    cudaMallocHost((void **)&h_diff2, 20);

    kernel<<<1, 10>>>(d_pos, d_xs, d_diff);
    cudaMemcpy(h_xs, d_xs, 20, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_diff, d_diff, 20, cudaMemcpyDeviceToHost);

    kernel2<<<1, 10>>>(d_pos, d_xs, d_diff);
    cudaMemcpy(h_xs2, d_xs, 20, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_diff2, d_diff, 20, cudaMemcpyDeviceToHost);

    uint8_t *minchia = new uint8_t[20];

    // for (int i = 0; i < 20; i++) {
    //     minchia[h_xs2[i]] = h_diff2[i];
    // }

    // for (int i = 0; i < 20; i++) {
    //     printf("%d) %d\n", i, minchia[i]);
    // }

    // for (int i = 0; i < 20; i++) {
    //     if (h_diff[h_xs[i]] != h_diff2[h_xs2[i]]) {
    //         printf("oh!\n");
    //     }
    // }
    
    // for (int i = 0; i < 20; i++) {
    //     printf("%d] %d\t%d] %d\n", i, h_xs[i], i, h_diff[i]);
    // }

    // printf("..........\n");
    // for (int i = 0; i < 20; i++) {
    //     printf("%d] %d\t%d] %d\n", i, h_xs2[i], i, h_diff2[i]);
    // }

    return 0;
}