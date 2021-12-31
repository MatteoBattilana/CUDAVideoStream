#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <time.h>
#include <stdint.h>

__global__ void kernel(uint8_t *buffer, int maxSect) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;

    // printf("%d .. %d\n", x * maxSect, x * maxSect + maxSect);
    // printf("%d\n", x);

    int max = x * maxSect + maxSect;
    for (int i = x * maxSect; i < max; i++) {
        buffer[i] += 1;
    }
}


int main() {
    struct cudaDeviceProp prop;
    int width = 1920;
    int height = 1080;
    uint8_t *d_buffer;

    cudaGetDeviceProperties(&prop, 0);
    fprintf(stderr, "maxThreads per Block: %d\n", prop.maxThreadsPerBlock);
    fprintf(stderr, "maxThreads per Multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
    fprintf(stderr, "warpSize: %d\n", prop.warpSize);
    fprintf(stderr, "MP count: %d\n", prop.multiProcessorCount);

    const int total = 3 * width * height;

    cudaMalloc((void **)&d_buffer, total * sizeof *d_buffer);
    uint8_t *h_buffer = new uint8_t[total * sizeof *h_buffer];
    memset(h_buffer, 0, total * sizeof *h_buffer);


    int maxAtTime = total / prop.maxThreadsPerBlock;

    int max = 100;
    while(max--) {
        clock_t start = clock();
        memset(h_buffer, 0, total * sizeof *h_buffer);

        cudaMemcpy(d_buffer, h_buffer, total * sizeof *h_buffer, cudaMemcpyHostToDevice);
        kernel<<<3, prop.maxThreadsPerBlock>>>(d_buffer, maxAtTime / 3);
        cudaMemcpy(h_buffer, d_buffer, total * sizeof *h_buffer, cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();
        fprintf(stderr, "\rexec time: %.2f ms", (float(clock()) - start) / CLOCKS_PER_SEC * 1e3);

        for (int i = 0; i < total * sizeof *h_buffer; i++) {
            if (h_buffer[i] != 1) {
                fprintf(stderr, "\nwtf!!\n");
                exit(1);
            }
        }
    }

    fprintf(stderr, "\n");

    return 0;
}