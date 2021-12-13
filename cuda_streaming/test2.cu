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

int main() {

    int tot = 3 * 1920 * 1080;

    uint8_t *host, *device;
    cudaMallocHost((void **)&host, tot);
    cudaMalloc((void **)&device, tot);

    while(1) {
        auto begin = std::chrono::high_resolution_clock::now();
        cudaMemcpyAsync(device, host, tot, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();

        auto elaps = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        printf("\rFOR: %5.2f ms\n", (float)elaps.count() * 1e-6);
        fflush(stdout);
    }

    return 0;
}