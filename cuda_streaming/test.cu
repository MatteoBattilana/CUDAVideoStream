
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

#define NSTREAMS 1
#define GPU

using namespace cv;

struct ctxs {
    VideoCapture *cap;
    int cap_w_fd;
    Mat *sampleMat;
    int show_r_fd;
    int ptr_w_fd;
    int ptr_r_fd;
    int proc_w_fd;
    // unsigned int **phpos;
    // int **pxs;
};

#ifdef GPU
__global__ void kernel(uint8_t *current, uint8_t *previous, uint8_t *diff, int maxSect, unsigned int *pos, int *xs) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int df, npos;

    int max = x * maxSect + maxSect;
    for (int i = x * maxSect; i < max; i++) {

        df = current[i] - previous[i];
        if (df > 0) {
            npos = atomicInc(pos, 6220801);
            // printf("npos %d\n", npos);
            diff[npos] = df;
            xs[npos] = i;
        }
        // diff[i] = current[i] - previous[i];

        // xs[i] = 0;
        // if (true) {
        //     npos = atomicInc(pos, 6220801);
        //     diff[npos].x = i;
        //     diff[npos].diff = df;
        //     printf("%u] %d %d\n", npos, i, df);
        // }
    }
}
#endif

void *th_cap_hdl(void *args) {
    int fifosize;
    Mat *pframe;

    struct ctxs *pctx = (struct ctxs *)args;

    while(1) {
        // Mat *frame = new Mat(pctx->sampleMat->rows, pctx->sampleMat->cols, pctx->sampleMat->type());
        read(pctx->ptr_r_fd, &pframe, sizeof pframe);
        *pctx->cap >> *pframe;

        write(pctx->cap_w_fd, &pframe, sizeof pframe);
    }

    return NULL;
}

void *th_show_hdl(void *args) {
    Mat *pframe;
    bool skip = true;

    struct ctxs *pctx = (struct ctxs *)args;
    int tot = 3 * pctx->sampleMat->cols * pctx->sampleMat->rows;

    while(1) {
        read(pctx->show_r_fd, &pframe, sizeof pframe);

        if (skip ^= 1) {
            skip = true;
        }
        

        // write(pctx->proc_w_fd, *pctx->phpos, sizeof **pctx->phpos);
        // write(pctx->proc_w_fd, *pctx->pxs, **pctx->phpos);
        // write(pctx->proc_w_fd, pframe->data, **pctx->phpos);
        // write(pctx->proc_w_fd, pframe->data, tot);
        // imshow("hi", *pframe);
        // if (waitKey(10) == 27) {
        //     break;  // stop capturing by pressing ESC
        // }

        write(pctx->ptr_w_fd, &pframe, sizeof pframe);

    }

    return NULL;
}

int main() {

    VideoCapture cap;
    if (!cap.open(0, CAP_V4L2)) return 1;
    auto codec = cv::VideoWriter::fourcc('M','J','P','G');
    cap.set(cv::CAP_PROP_FOURCC, codec);

    cap.set(3, 1920);
    cap.set(4, 1080);

    printf("A\n");
    Mat base;
    cap >> base;
    printf("B\n");

    int cap_pipe[2];
    int show_pipe[2];
    int ptr_pipe[2];
    int fork_pipe[2];
    pipe(cap_pipe);
    pipe(show_pipe);
    pipe(ptr_pipe);
    pipe(fork_pipe);

    printf("forking\n");
    if (!fork()) {
        int tot = 3 * base.cols * base.rows;
        uint8_t *buffer = new uint8_t[tot];
        Mat frame(base.rows, base.cols, base.type());

        int btot = 0;

        while (btot < tot) {
            btot += read(fork_pipe[0], buffer + btot, 1024);
        }

        memcpy(frame.data, buffer, tot);

        printf("forked\n");

        namedWindow("hi", WINDOW_NORMAL);
        resizeWindow("hi", Size(653, 373));

        while(1) {
            // btot = 0;
            // while (btot < tot) {
            //     btot += read(fork_pipe[0], buffer + btot, 1024);
            // }
            read(fork_pipe[0], buffer, tot);

#if defined(CPU) || defined(GPU)
            // for (int i = 0; i < tot; i++) {
            //     frame.data[i] += buffer[i];
            // }
            // memcpy(frame.data, buffer, tot);
#endif

            imshow("hi", frame);
            if (waitKey(10) == 27) {
                break;  // stop capturing by pressing ESC
            }
        }
    }

    printf("copying\n");
    for (int i = 0; i < 3 * base.rows; i++) {
        write(fork_pipe[1], base.data + base.cols * i, base.cols);
    }

    pthread_mutex_t fifosize_mtx;
    pthread_mutex_init(&fifosize_mtx, NULL);

    int *h_xs;
    unsigned int *h_pos;
    struct ctxs ctx = { 
        .cap = &cap,
        .cap_w_fd = cap_pipe[1], 
        .sampleMat = &base ,
        .show_r_fd = show_pipe[0],
        .ptr_w_fd = ptr_pipe[1],
        .ptr_r_fd = ptr_pipe[0],
        .proc_w_fd = fork_pipe[1]
    };

    pthread_t th_cap;
    pthread_t th_show;
    pthread_create(&th_cap, NULL, th_cap_hdl, &ctx);
    pthread_create(&th_show, NULL, th_show_hdl, &ctx);

    Mat *pframe;
    for (int i = 0; i < 10; i++) {

#ifdef GPU
        uint8_t *h_frame;
        cudaMallocHost((void **)&h_frame, 3 * ctx.sampleMat->rows * ctx.sampleMat->cols * sizeof *h_frame);
        pframe = new Mat(ctx.sampleMat->rows, ctx.sampleMat->cols, ctx.sampleMat->type(), h_frame);
#else
        pframe = new Mat(ctx.sampleMat->rows, ctx.sampleMat->cols, ctx.sampleMat->type());
#endif

        write(ctx.ptr_w_fd, &pframe, sizeof pframe);
    }

    int total = 3 * ctx.sampleMat->rows * ctx.sampleMat->cols;

#ifdef CPU
    uint8_t *h_diff = new uint8_t[total];
#elif defined(GPU)
    struct cudaDeviceProp prop;
    uint8_t *d_current, *d_previous;
    uint8_t *d_diff;
    int *d_xs;
    unsigned int *d_pos;
    cudaStream_t streams[4];

    cudaGetDeviceProperties(&prop, 0);
    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);
    cudaStreamCreate(&streams[2]);
    cudaStreamCreate(&streams[3]);

    cudaMalloc((void **)&d_diff, total * sizeof *d_diff);
    cudaMalloc((void **)&d_xs, total * sizeof *d_xs);
    cudaMalloc((void **)&d_current, total * sizeof *d_current);
    cudaMalloc((void **)&d_previous, total * sizeof *d_previous);

    cudaMalloc((void **)&d_pos, sizeof *d_pos);
    // cudaMemset((void *)d_pos, 0, sizeof *d_pos);

    uint8_t *h_diff;
    cudaMallocHost((void **)&h_diff, total * sizeof *h_diff);
    cudaMallocHost((void **)&h_pos, sizeof *h_pos);
    cudaMallocHost((void **)&h_xs, sizeof *h_xs);

    int maxAtTime = total / prop.maxThreadsPerBlock;
    cudaMemcpy(d_current, ctx.sampleMat->data, total * sizeof *ctx.sampleMat->data, cudaMemcpyHostToDevice);

    int tot4 = total / 4;
    int max4 = maxAtTime / 4;
    uint8_t *dcurr4_0 = d_current;
    uint8_t *dcurr4_1 = d_current + total/4;
    uint8_t *dcurr4_2 = d_current + total/2;
    uint8_t *dcurr4_3 = d_current + 3*total/4;
    uint8_t *dprev4_0 = d_previous;
    uint8_t *dprev4_1 = d_previous + total/4;
    uint8_t *dprev4_2 = d_previous + total/2;
    uint8_t *dprev4_3 = d_previous + 3*total/4;
    uint8_t *ddiff_0 = d_diff;
    uint8_t *ddiff_1 = d_diff + total/4;
    uint8_t *ddiff_2 = d_diff + total/2;
    uint8_t *ddiff_3 = d_diff + 3*total/4;
    // uint8_t *pframe_0 = pframe->data;
    // uint8_t *pframe_1 = pframe->data + total/4;
    // uint8_t *pframe_2 = pframe->data + total/2;
    // uint8_t *pframe_3 = pframe->data + 3*total/4;

#endif

    Mat previous = ctx.sampleMat->clone();
    int streamidx = 0;
    while (1) {
        auto begin = std::chrono::high_resolution_clock::now();

        auto begin2 = std::chrono::high_resolution_clock::now();
        read(cap_pipe[0], &pframe, sizeof pframe);
        auto end2 = std::chrono::high_resolution_clock::now();

        auto begin3 = std::chrono::high_resolution_clock::now();
#ifdef CPU
        Mat pvs = pframe->clone();

        for (int i = 0; i < total; i++) {
            pframe->data[i] = pframe->data[i] - previous.data[i];
        }

        previous = pvs;
#elif defined(GPU)

        uint8_t *d_prev = dcurr4_0;
        dcurr4_0 = dprev4_0;
        dprev4_0 = d_prev;

        // d_prev = d_current;
        // d_current = d_previous;
        // d_previous = d_prev;

        // d_prev = dcurr4_1;
        // dcurr4_1 = dprev4_1;
        // dprev4_1 = d_prev;

        // d_prev = dcurr4_2;
        // dcurr4_2 = dprev4_2;
        // dprev4_2 = d_prev;

        // d_prev = dcurr4_3;
        // dcurr4_3 = dprev4_3;
        // dprev4_3 = d_prev;

        cudaMemsetAsync(d_pos, 0, sizeof *d_pos);
        cudaMemcpyAsync(dcurr4_0, pframe->data, tot4, cudaMemcpyHostToDevice);
        kernel<<<1, prop.maxThreadsPerBlock, 0>>>(dcurr4_0, dprev4_0, ddiff_0, max4, d_pos, d_xs);
        cudaMemcpyAsync(pframe->data, ddiff_0, tot4, cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(h_pos, d_pos, sizeof *d_pos, cudaMemcpyDeviceToHost);
        // cudaMemcpyAsync(h_xs, d_xs, *h_pos * sizeof *d_xs, cudaMemcpyDeviceToHost);

        // cudaMemcpyAsync(dcurr4_1, pframe_1, tot4, cudaMemcpyHostToDevice, streams[1]);
        // kernel<<<1, prop.maxThreadsPerBlock, 0, streams[1]>>>(dcurr4_1, dprev4_1, ddiff_1, max4, d_pos);
        // cudaMemcpyAsync(pframe_1, ddiff_1, tot4, cudaMemcpyDeviceToHost, streams[1]);

        // cudaMemcpyAsync(dcurr4_2, pframe_2, tot4, cudaMemcpyHostToDevice, streams[2]);
        // kernel<<<1, prop.maxThreadsPerBlock, 0, streams[2]>>>(dcurr4_2, dprev4_2, ddiff_2, max4, d_pos);
        // cudaMemcpyAsync(pframe_2, ddiff_2, tot4, cudaMemcpyDeviceToHost, streams[2]);

        // cudaMemcpyAsync(dcurr4_3, pframe_3, tot4, cudaMemcpyHostToDevice, streams[3]);
        // kernel<<<1, prop.maxThreadsPerBlock, 0, streams[3]>>>(dcurr4_3, dprev4_3, ddiff_3, max4, d_pos);
        // cudaMemcpyAsync(pframe_3, ddiff_3, tot4, cudaMemcpyDeviceToHost, streams[3]);

        cudaDeviceSynchronize();

#endif
        auto end3 = std::chrono::high_resolution_clock::now();

        write(show_pipe[1], &pframe, sizeof pframe);

        auto end = std::chrono::high_resolution_clock::now();
        auto elaps = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        auto elaps2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - begin2);
        auto elaps3 = std::chrono::duration_cast<std::chrono::nanoseconds>(end3 - begin3);

        printf("\rFPS: %5.0f\tFOR: %5.2f ms\tREAD: %5.2f\tPOS: %d\n", 1 / ((float)elaps.count() * 1e-9), (float)elaps3.count() * 1e-6, (float)elaps2.count() * 1e-6, *h_pos);
        fflush(stdout);
    }


    return 0;
}