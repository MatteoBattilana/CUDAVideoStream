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

__global__ void kernel(uint8_t *a, uint8_t *b) {
    int tid = threadIdx.x;
    int *pa, *pb;
    int ca, cb;

    for (int i = 2 * tid; i < 2 * tid + 2; i++) {

        ca = ((int *)a)[i];
        cb = ((int *)b)[i];

        // printf("ca %d\n", ca);

        for (int j = 0; j < 4; j++) {
            // printf("%d] new ca[%d %d]: %d - %d\n", i, i, j, ((uint8_t *)&ca)[j], ((uint8_t *)&cb)[j]);
            ((uint8_t *)&ca)[j] -= ((uint8_t *)&cb)[j];
        }

        ((int *)a)[i] = ca;
    }
    
}

__global__ void kernel2(uint8_t *a, uint8_t *b) {
    int tid = threadIdx.x;
    int *pa, *pb;
    int ca, cb;

    for (int i = 2 * tid; i < 2 * tid + 2; i++) {

        ca = ((int *)a)[i];
        cb = ((int *)b)[i];

        // printf("ca %d\n", ca);

        ((int *)a)[i] = __vsub4(ca, cb);

        // #pragma unroll
        // for (int j = 0; j < 4; j++) {
        //     // printf("%d] new ca[%d %d]: %d - %d\n", i, i, j, ((uint8_t *)&ca)[j], ((uint8_t *)&cb)[j]);
        //     ((uint8_t *)&ca)[j] -= ((uint8_t *)&cb)[j];
        // }

        // ((int *)a)[i] = ca;
    }
    
}

int main() {

    uint8_t *d_a, *d_b;
    cudaMalloc((void **)&d_b, 20);
    cudaMalloc((void **)&d_a, 20);


    uint8_t h_a[] = { 3, 4, 8, 7, 9, 4, 4, 6, 3, 4, 8, 7, 9, 4, 4, 6 };
    uint8_t h_b[] = { 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3 };

    cudaMemcpy(d_a, h_a, 16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, 16, cudaMemcpyHostToDevice);

    for (int i = 0; i < 30000; i++)
        kernel2<<<1, 2>>>(d_a, d_b);

    cudaMemcpy(h_a, d_a, 16, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 16; i++) {
        printf("[%d] %d ", i, h_a[i]);
    }

    printf("\n");

    return 0;
}