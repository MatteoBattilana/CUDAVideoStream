
#include <stdio.h>
#include <iomanip>
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

#define CHARS_STR "0123456789BFPSW :"
#define LR_THRESHOLDS 20
#define NSTREAMS 1
#define GPU
#define KERNEL2_NEGFEED_OPT

using namespace cv;

typedef long4 chunk_t;

struct mat_ready {
    Mat *pframe;
    int *h_xs;
    unsigned int h_pos;
};

struct cb_args {
    unsigned int *d_pos;
    int show_w_fd;
    struct mat_ready *pready;
};

struct ctxs {
    VideoCapture *cap;
    int cap_w_fd;
    Mat *sampleMat;
    int show_r_fd;
    int ptr_w_fd;
    int ptr_r_fd;
};

#ifdef GPU

// __constant__ uint8_t c_matrix[(sizeof(CHARS_STR) - 1) * 32 * 28];

/*
Optimizations done:
 - removing memsetCuda(pos, 0) and initializing pos at the beginning of the kernel
*/

__global__ void kernel(uint8_t *current, uint8_t *previous, uint8_t *diff, int maxSect, unsigned int *pos, int *xs) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int npos;
    int df;
    
    if (!x) *pos = 0;
    __syncthreads();

    int start = x * maxSect;
    int max = start + maxSect;

    #pragma unroll
    for (int i = start; i < max; i++) {

        df = current[i] - previous[i];
        if (df < -LR_THRESHOLDS || df > LR_THRESHOLDS) {
            npos = atomicInc(pos, 6220801);
            diff[npos] = df;
            xs[npos] = i;
        } else {
            current[i] -= df;
        }

    }

    // __syncthreads();

    // if (x == 0) {
    //     *pos = s_pos[0];
    // }
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
    chunk_t cc;

    int start = thid * N;
    int max = start + N;

    for (int i = start; i < max; i++) {

        int reali = i * sizeof cc;
        int x = offset + reali % matrixWidth;
        int y = reali / matrixWidth;

        int idx = (y * currWidth + x) / sizeof cc;
        cc = ((chunk_t *)current)[idx];

        #pragma unroll
        for (int j = 0; j < sizeof cc; j++) {
            ((uint8_t *)&cc)[j] = matrix[reali + j];
        }

        ((chunk_t *)current)[idx] = cc;
    } 

}

#endif

void *th_cap_hdl(void *args) {

    struct mat_ready *pready;
    struct ctxs *pctx = (struct ctxs *)args;

    while(1) {
        read(pctx->ptr_r_fd, &pready, sizeof pready);
        *pctx->cap >> *(pready->pframe);

        write(pctx->cap_w_fd, &pready, sizeof pready);
    }

    return NULL;
}

void *th_show_hdl(void *args) {
    struct epoll_event ev, events[10];
    struct addrinfo *result, *rp;
    int sfd, epollfd, nfds, sfd2;
    struct mat_ready *pready;

    getaddrinfo("127.0.0.1", "2734", NULL, &result);
    for (rp = result; rp != NULL; rp = rp->ai_next) {
        if ((sfd = socket(AF_INET, SOCK_STREAM, 0)) == -1) {
            continue;
        }

        if (bind(sfd, rp->ai_addr, rp->ai_addrlen) != -1) {
            break;
        }

        close(sfd);
        perror("MOH!");
    }

    epollfd = epoll_create1(0);
    ev.events = EPOLLIN;
    ev.data.fd = sfd;
    epoll_ctl(epollfd, 1, sfd, &ev);

    if (listen(sfd, 10) < 0) {
        perror("OH!");
        exit(errno);
    }

    nfds = epoll_wait(epollfd, events, 10, -1);

    for (int i = 0; i < nfds; i++) {
        if (events[i].data.fd == sfd) {
            sfd2 = accept(sfd, NULL, NULL);
            if (sfd2 < 0) perror("ACCEPT");
        }
    }

    struct ctxs *pctx = (struct ctxs *)args;
    int tot = 3 * pctx->sampleMat->cols * pctx->sampleMat->rows;

    // sending the base frame at the beginning
    write(sfd2, pctx->sampleMat->data, tot);

    while(1) {
        read(pctx->show_r_fd, &pready, sizeof pready);

        write(sfd2, &pready->h_pos, sizeof pready->h_pos);
        write(sfd2, pready->h_xs, pready->h_pos * sizeof *pready->h_xs);
        write(sfd2, pready->pframe->data, pready->h_pos);

        write(pctx->ptr_w_fd, &pready, sizeof pready);
    }

    return NULL;
}

void sigpipe_hdl(int sig) {
    exit(1);
}

/*void cb(cudaStream_t stream, cudaError_t error, void *uData) {

    struct cb_args *pargs = (struct cb_args *)uData;

    // cudaMemcpy(&pargs->pready->h_pos, pargs->d_pos, sizeof *pargs->d_pos, cudaMemcpyDeviceToHost); 
    write(pargs->show_w_fd, &pargs->pready, sizeof pargs->pready);
}*/

int main() {

    VideoCapture cap;
    if (!cap.open(0, CAP_V4L2)) return 1;
    auto codec = cv::VideoWriter::fourcc('M','J','P','G');
    cap.set(cv::CAP_PROP_FOURCC, codec);

    cap.set(3, 1920);
    cap.set(4, 1080);

    Mat base;
    cap >> base;

    auto txtsz = cv::getTextSize("A", cv::FONT_HERSHEY_PLAIN, 3, 2, 0);
    std::cout << "Character pixel size: " << txtsz << std::endl;
    int fullArea = 3 * txtsz.area();

    uint8_t *charsPx = new uint8_t[(sizeof(CHARS_STR) - 1) * fullArea];
    memset(charsPx, 0x0, (sizeof(CHARS_STR) - 1) * fullArea);

    for (int i = 0; i < (sizeof(CHARS_STR) - 1); i++) {
        Mat pxBaseMat(txtsz.height, txtsz.width, base.type(), charsPx + i * fullArea);
        cv::putText(pxBaseMat, std::string(CHARS_STR).substr(i, 1), cv::Point(0, txtsz.height+1), cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    }

#ifdef GPU
    uint8_t *d_charsPx;
    int totcpy = fullArea * sizeof *d_charsPx * (sizeof(CHARS_STR) - 1);
    cudaMalloc((void **)&d_charsPx, totcpy);
    cudaMemcpy(d_charsPx, charsPx, totcpy, cudaMemcpyHostToDevice);
    // cudaMemcpyToSymbol(c_matrix, charsPx, totcpy);
#endif

    int cap_pipe[2];
    int show_pipe[2];
    int ptr_pipe[2];
    pipe(cap_pipe);
    pipe(show_pipe);
    pipe(ptr_pipe);

    signal(SIGPIPE, sigpipe_hdl);

    struct ctxs ctx = { 
        .cap = &cap,
        .cap_w_fd = cap_pipe[1], 
        .sampleMat = &base ,
        .show_r_fd = show_pipe[0],
        .ptr_w_fd = ptr_pipe[1],
        .ptr_r_fd = ptr_pipe[0]
    };

    pthread_t th_cap;
    pthread_t th_show;
    pthread_create(&th_cap, NULL, th_cap_hdl, &ctx);
    pthread_create(&th_show, NULL, th_show_hdl, &ctx);

    struct mat_ready *pready;
    for (int i = 0; i < 6; i++) {

        pready = new struct mat_ready;

#ifdef GPU
        uint8_t *h_frame;
        cudaMallocHost((void **)&h_frame, 3 * ctx.sampleMat->rows * ctx.sampleMat->cols * sizeof *h_frame + sizeof(chunk_t));
        cudaMallocHost((void **)&pready->h_xs, 3 * ctx.sampleMat->rows * ctx.sampleMat->cols * sizeof *pready->h_xs + sizeof(chunk_t));
        pready->pframe = new Mat(ctx.sampleMat->rows, ctx.sampleMat->cols, ctx.sampleMat->type(), h_frame);
#else
        pready->pframe = new Mat(ctx.sampleMat->rows, ctx.sampleMat->cols, ctx.sampleMat->type());
        pready->h_xs = new int[3 * ctx.sampleMat->rows * ctx.sampleMat->cols];
#endif

        pready->h_pos = 0;
        write(ctx.ptr_w_fd, &pready, sizeof pready);
    }

    int total = 3 * ctx.sampleMat->rows * ctx.sampleMat->cols;

#ifdef GPU
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

	int nMaxThreads = prop.maxThreadsPerBlock;
    int maxAtTime = total / nMaxThreads;
    cudaMemcpy(d_current, ctx.sampleMat->data, total * sizeof *ctx.sampleMat->data, cudaMemcpyHostToDevice);

    int max4 = ceil(1.0 * maxAtTime / sizeof(chunk_t));

    // struct cb_args *pargs = new struct cb_args;
    // pargs->d_pos = d_pos;
    // pargs->show_w_fd = show_pipe[1];

    int nThreadsToUse = nMaxThreads;
    int eachThreadDoes = 1;
    for (int i = nMaxThreads; i > 0; i--) {
        float frac = fullArea / (i * 1.0);
        int frac_int = fullArea / i;
        if ((int)ceil(frac) == frac_int && frac_int % sizeof(chunk_t) == 0) {
            nThreadsToUse = i;
            eachThreadDoes = frac_int / sizeof(chunk_t);
            break;
        }
    }

    char *d_text;
    cudaMalloc((void **)&d_text, 1920 * sizeof *d_text);

#endif

    Mat previous = ctx.sampleMat->clone();
    std::string overImageText;

    auto begin0 = std::chrono::high_resolution_clock::now();
    while (1) {
        auto begin = std::chrono::high_resolution_clock::now();

        auto begin2 = std::chrono::high_resolution_clock::now();
        read(cap_pipe[0], &pready, sizeof pready);
        auto end2 = std::chrono::high_resolution_clock::now();

        auto begin3 = std::chrono::high_resolution_clock::now();
#ifdef CPU

        const std::string welcomestr = "WELCOME";
        for (int offset = 0, j = 0; j < welcomestr.length(); j++, offset += txtsz.width*3) {
            int idx;
            for (int i = 0; i < (sizeof(CHARS_STR) - 1); i++) {
                if (CHARS_STR[i] == welcomestr.at(j)) {
                    idx = i;
                }
            }

            for (int i = 0; i < 3 * txtsz.area(); i++) {
                int x = offset + i % (txtsz.width * 3);
                int y = 10 + i / (txtsz.width * 3);
                pready->pframe->data[y * 3 * pready->pframe->cols + x] = charsPx[idx * fullArea + i];
            }
        }


        Mat pvs = pready->pframe->clone();

        pready->h_pos = 0;
        for (int i = 0; i < total; i++) {
            int df = pready->pframe->data[i] - previous.data[i];
            if (df < -LR_THRESHOLDS || df > LR_THRESHOLDS) {
                pready->pframe->data[pready->h_pos] = df;
                pready->h_xs[pready->h_pos] = i;
                pready->h_pos++;
            } else {
                pvs.data[i] -= df;
            }
        }

        previous = pvs;

#elif defined(GPU)

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
        uint8_t *d_prev = d_current;
        d_current = d_previous;
        d_previous = d_prev;

        // cudaMemsetAsync(d_pos, 0, sizeof *d_pos);

        // Copy in the current pointer and run 
        cudaMemcpyAsync(d_current, pready->pframe->data, total, cudaMemcpyHostToDevice);

        for (int offset = 0, j = 0; j < overImageText.length(); j++, offset += txtsz.width*3) {
            int idx;
            for (int i = 0; i < (sizeof(CHARS_STR) - 1); i++) {
                if (CHARS_STR[i] == overImageText.at(j)) {
                    idx = i;
                    break;
                }
            }

            kernel2_char<<<1, nThreadsToUse>>>(d_current, d_charsPx + idx * fullArea, eachThreadDoes, offset, 3 * txtsz.width, 3 * pready->pframe->cols);
        }

        // kernel<<<1, nMaxThreads, 0>>>(d_current, d_previous, d_diff, maxAtTime, d_pos, d_xs);
        kernel2<<<1, nMaxThreads, 0>>>(d_current, d_previous, d_diff, max4, d_pos, d_xs);

        cudaMemcpyAsync(&pready->h_pos, d_pos, sizeof *d_pos, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        cudaMemcpyAsync(pready->pframe->data, d_diff, pready->h_pos, cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(pready->h_xs, d_xs, pready->h_pos * sizeof *d_xs, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

#endif
        auto end3 = std::chrono::high_resolution_clock::now();

        write(show_pipe[1], &pready, sizeof pready);

        auto end = std::chrono::high_resolution_clock::now();

        auto end0 = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::nanoseconds>(end0 - begin0).count() > 1e9) {
            begin0 = end0;

            auto elaps = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
            auto elaps2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - begin2);
            auto elaps3 = std::chrono::duration_cast<std::chrono::nanoseconds>(end3 - begin3);

            float unit = 1 / ((float)elaps.count() * 1e-9);
            int bw = static_cast<int>((pready->h_pos<<4)*unit*1e-3);

            std::stringstream strstream;
            strstream << std::setfill(' ') << std::setw(5) << static_cast<int>(unit);

            printf("\rFPS: %s\tFOR: %5.2f ms\tREAD: %9.2f\tPOS: %7d\t BW: %5d kbps", strstream.str().c_str(), (float)elaps3.count() * 1e-6, (float)elaps2.count() * 1e-6, pready->h_pos, bw);

            strstream = std::stringstream();
            strstream << "FPS: " << static_cast<int>(unit) << " BW: " << bw << " kbps";
            overImageText = strstream.str();

            fflush(stdout);
        }

    }

    return 0;
}