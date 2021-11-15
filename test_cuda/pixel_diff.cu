#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdint.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

__global__ void diff_kernel(int *current, int *prev, int *diff, int sectors) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   
    for (int a = 0; a < sectors; a++){
        diff[i*sectors + a] = current[i*sectors + a] - prev[i*sectors + a];
    }
}

int main(int argc, char **argv) {
    int *current = (int*)malloc(3*1920*1080*sizeof(int));
    int *prev = (int*)malloc(3*1920*1080*sizeof(int));
    int *diff = (int*)malloc(3*1920*1080*sizeof(int));
    for (int i = 0; i < 3*1920*1080; i++) {
        current[i] = (int) (rand() % 10);
        prev[i] = (i%2==0 ? (int) (rand() % 10) : current[i]);
    }
    
    int *current_d;
    int *prev_d;
    int *diff_d;

    // CUDA malloc
    cudaMalloc((void**)&current_d, 3*1920*1080*sizeof(int));	
    cudaMalloc((void**)&prev_d, 3*1920*1080*sizeof(int));	
    cudaMalloc((void**)&diff_d, 3*1920*1080*sizeof(int));	
    cudaMemcpy( current_d, current,3*1920*1080*sizeof(int), cudaMemcpyHostToDevice );
    cudaMemcpy( prev_d, prev, 3*1920*1080*sizeof(int), cudaMemcpyHostToDevice );

    int sectors = (3*1920*1080)/6480;
    while(1){
        clock_t start = clock();
        diff_kernel <<< 1, sectors >>> (current_d, prev_d, diff_d, 6480);	
        cudaDeviceSynchronize();	
        cudaMemcpy( diff, diff_d, 3*1920*1080*sizeof(int), cudaMemcpyDeviceToHost );
        printf("Loop time %fms\n", float(clock() - start) / CLOCKS_PER_SEC * 1e3);
        
        // Check difference is correct
        int c = 0;
        int c1 = 0;
        for (int i = 0; i < 3*1920*1080; i++) {
            if (current[i] != prev[i])
                c++;
            if (diff[i] != 0)
                c1++;
        }
        if(c != c1){
            printf("#ERROR Diff: %d %d\n", c, c1);
            break;
        }
    }
    
    int c = 0;
    int c1 = 0;
    for (int i = 0; i < 3*1920*1080; i++) {
        if (current[i] != prev[i])
            c++;
        if (diff[i] != 0)
            c1++;
    }
    printf("Diff: %d %d\n", c, c1);

    return 0;
}
