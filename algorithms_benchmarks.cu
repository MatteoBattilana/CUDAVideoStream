#include <stdio.h>
#include <math.h>  

int * generateImage(int h, int w) {
    int * image = (int*)malloc(h*w*sizeof(int)*3);
    for(int i = 0; i < h*w*3; i++) {
        image[i] = rand() % 255;
    }
    return image;
}

int checkDifference(int * frame1, int * frame2, int * diff_cuda, int h, int w) {
    for(int i = 0; i < h; i++){
        for (int a =0; a<w; a++){
            if(frame1[i*h+a] - frame2[i*h+a] != diff_cuda[i*h+a]){
                printf("%d not match!\n", i*h+a);
                return -1;
            }
        }
    }
    return 0;
}

__global__ void kernel1(int *current, int *previous, int *diff, int maxSect) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int max = x * maxSect + maxSect;
    for (int i = x * maxSect; i < max; i++) {
        diff[i] = current[i] - previous[i];
    }
}

__global__ void kernel2(int *current, int *previous, int *diff, int maxSect, int current2) {
    int x = threadIdx.x;
    //printf("CALLED %d %d %d\n",x,  x * maxSect ,x * maxSect +  maxSect);
    int max = x * maxSect + maxSect;
    //printf("%d\n", x * maxSect);
    for (int i = x * maxSect; i < max; i++) {
        diff[i] = current[i] - previous[i];
    }
}

float test1(int * frame1, int * frame2, int * h_buffer){
    int *d_diff, *d_current, *d_previous;

    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const int total = 3 * 1080 * 1920;
    cudaMalloc((void **)&d_diff, total * sizeof *d_diff);
    cudaMalloc((void **)&d_current, total * sizeof *d_current);
    cudaMalloc((void **)&d_previous, total * sizeof *d_previous);
    int maxAtTime = total / prop.maxThreadsPerBlock;

    clock_t start = clock();
    // cudaMemset((void *)d_diff, 0, total * sizeof *d_diff); // optional
    cudaMemcpy(d_current, frame1, total * sizeof *frame1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_previous, frame2, total * sizeof *frame2, cudaMemcpyHostToDevice);
    kernel1<<<3, prop.maxThreadsPerBlock>>>(d_current, d_previous, d_diff, maxAtTime / 3);
    cudaMemcpy(h_buffer, d_diff, total * sizeof *h_buffer, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    return (float(clock() - start) / CLOCKS_PER_SEC);
}

float test2(int * frame1, int * frame2, int * h_buffer){
    int *d_diff, *d_current, *d_previous;

    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const int total = 3 * 1080 * 1920;
    cudaMalloc((void **)&d_diff, total * sizeof *d_diff);
    cudaMalloc((void **)&d_current, total * sizeof *d_current);
    cudaMalloc((void **)&d_previous, total * sizeof *d_previous);

    int n_streams = 5;
    cudaStream_t stream[n_streams];
    for (int i = 0; i < n_streams; i++) cudaStreamCreate(&stream[i]);

    clock_t start = clock();
    int maxAtTime = (total/n_streams) / prop.maxThreadsPerBlock;
    for (int i = 0; i < n_streams; i++) {
        int icr = i*(total/n_streams);
        cudaMemcpyAsync(d_current+icr, frame1+icr, (total/n_streams)* sizeof(int), cudaMemcpyHostToDevice, stream[i]);
        cudaMemcpyAsync(d_previous+icr, frame2+icr, (total/n_streams)* sizeof(int), cudaMemcpyHostToDevice, stream[i]);
        kernel2<<<1, prop.maxThreadsPerBlock, 0, stream[i]>>>(d_current+icr, d_previous+icr, d_diff+icr, maxAtTime, icr);
        cudaMemcpyAsync(h_buffer+icr, d_diff+icr, (total/n_streams)* sizeof(int) , cudaMemcpyDeviceToHost, stream[i]);
    }
    cudaDeviceSynchronize();

    for (int i=0; i<n_streams; ++i) cudaStreamDestroy(stream[i]);

    return (float(clock() - start) / CLOCKS_PER_SEC);
}

int main(void) {
    int N_TEST = 20;
    int * frame1 = generateImage(1920, 1080);
    int * frame2 = generateImage(1920, 1080);
    int * difference = generateImage(1920, 1080);
    float acc = 0;


    // TEST1
    for (int i = 0; i < N_TEST; i++){
        acc += test1(frame1, frame2, difference);
    }
    printf("* Time test1: %fs    FPS: %f\n", acc/N_TEST, 1.0/(acc/N_TEST));
    if(checkDifference(frame1, frame2, difference, 1920, 1080) == -1) {
        printf("Error test1\n");
    }

    // TEST2
    difference = generateImage(1920, 1080);
    acc = 0;
    for (int i = 0; i < N_TEST; i++){
        acc += test2(frame1, frame2, difference);
    }
    printf("* Time test2: %fs    FPS: %f\n", acc/N_TEST, 1.0/(acc/N_TEST));
    if(checkDifference(frame1, frame2, difference, 1920, 1080) == -1) {
        printf("Error test2\n");
    }




    // hello from GPU
    /*printf("Hello World from CPU!\n");
    dim3 threadsPerBlock(BLOCK,BLOCK);
    dim3 numBlocks(4096/(BLOCK*BLOCK), 1);
    int n_streams = (H*W)/((BLOCK*BLOCK)*(4096/(BLOCK*BLOCK))) + 1;
    printf("N streams: %d\nN thread blocks: %d\nN blocks: %d\n", n_streams, BLOCK*BLOCK, 4096/(BLOCK*BLOCK));
   
    cudaStream_t stream[n_streams];
    for (int i = 0; i < n_streams; i++) cudaStreamCreate(&stream[i]);

    for (int i = 0 ; i < n_streams; i++){
        //printf("-> %d\n", i);
        helloFromGPU <<<numBlocks,threadsPerBlock, 0, stream[i]>>>(i*n_streams);
        cudaDeviceSynchronize();
    }
    cudaDeviceSynchronize();

    for (int i=0; i<n_streams; ++i) cudaStreamDestroy(stream[i]);
    cudaDeviceReset();*/
    return 0;
}