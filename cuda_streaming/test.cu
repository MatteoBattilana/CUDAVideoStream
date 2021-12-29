
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

// 1 for heat map, 2 for red-black, 0 nothing
#define NOISE_FILTER
#define K 3
#define TILE_SIZE 10
#define BLOCK_SIZE (TILE_SIZE + K - 1)

// Noise visualizer: 1 heatmap, 2 red-black
#define NOISE_VISUALIZER 2

#define CHARS_STR "0123456789BFPSW :"
#define LR_THRESHOLDS 20
#define NSTREAMS 1
#define GPU
#define KERNEL2_NEGFEED_OPT

using namespace cv;

typedef long4 chunk_t;

struct mat_show {
    Mat *nframe;
};

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
    int noise_w_fd;
    int noise_r_fd;
};

#ifdef GPU

__constant__ float dev_k[K*K];

__global__ void convolution_kernel(uint8_t *image, uint8_t *R)
{
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
// TODO: try chunk_t so N/sizeof(chunk_t)
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
            ((uint8_t *)&cc)[j] = matrix[i * sizeof cc + j];
        }

        ((chunk_t *)current)[idx] = cc;
    } 

}

__global__ void kernel3_char(uint8_t *current, uint8_t *matrix, char *text, int textLen, int N, int matrixWidth, int currWidth, int fullArea) {
    int thid = threadIdx.x + blockIdx.x * blockDim.x;
    chunk_t cc;
    extern __shared__ uint8_t s_matrix[];

    printf("%s\n", text);

    if (!thid) {
        for (int i = 0; i < (sizeof(CHARS_STR) - 1) * fullArea; i++) {
            s_matrix[i] = matrix[i];
        }
    }

    __syncthreads();

    int start = thid * N;
    int max = start + N;

    for (int offset = 0, j = 0; j < textLen; j++, offset += matrixWidth) {
        int idx;
        for (int i = 0; i < (sizeof(CHARS_STR) - 1); i++) {
            if (CHARS_STR[i] == text[j]) {
                idx = i;
                break;
            }
        }

        for (int i = start; i < max; i++) {

            int reali = i * sizeof cc;
            int x = offset + reali % matrixWidth;
            int y = reali / matrixWidth;

            int idx = (y * currWidth + x) / sizeof cc;
            cc = ((chunk_t *)current)[idx];

            #pragma unroll
            for (int j = 0; j < sizeof cc; j++) {
                ((uint8_t *)&cc)[j] = matrix[i * sizeof cc + j];
            }

            ((chunk_t *)current)[idx] = cc;
        } 
    }

}

#endif

void *th_noise_hdl(void *args) {

    struct mat_show *show_ready;
    struct ctxs *pctx = (struct ctxs *)args;

    while(1) {
        read(pctx->noise_r_fd, &show_ready, sizeof show_ready);
        namedWindow("Noise Visualizer", WINDOW_GUI_NORMAL);
        imshow("Noise Visualizer", *(show_ready->nframe));
        if (waitKey(10) == 27) {
            break;  // stop capturing by pressing ESC
        }
    }

    return NULL;
}

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

void computeGaussianKernel(float* k, float sigma){
    float sum = 0;
    for (int i = 0; i < K; i++){
        for (int j = 0; j < K; j++){
            float x = i - (K - 1) / 2.0;
            float y = j - (K - 1) / 2.0;
            k[i*K+j] = (1.0/(2.0*M_PI*sigma*sigma)) * exp(-((x*x + y*y)/(2.0*sigma*sigma)));
            sum += k[i*K+j];
        }
    }

    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            k[i*K+j] /= sum;
        }
    }
}

int main() {

    VideoCapture cap;
    if (!cap.open(0, CAP_V4L2)) return 1;
    auto codec = cv::VideoWriter::fourcc('M','J','P','G');
    cap.set(cv::CAP_PROP_FOURCC, codec);

    // VideoCapture cap("v4l2src device=/dev/video0 ! video/x-raw, format=YUY2, width=640, height=480, framerate=30/1 ! nvvidconv ! video/x-raw(memory:NVMM) ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink", cv::CAP_GSTREAMER);
    // VideoCapture cap("v4l2src device=%s ! video/x-raw, width=640, height=480, format=(string)YUY2, \
    //             framerate=30/1 ! videoconvert ! video/x-raw, format=BGR ! appsink", cv::CAP_GSTREAMER);
    // VideoCapture cap;
    // if (!cap.open("v4l2src device=/dev/video0 io-mode=2 ! image/jpeg, width=1920, height=1080 ! nvjpegdec ! video/x-raw ! videoconvert ! video/x-raw, format=BGR ! appsink", CAP_GSTREAMER)) return 1;


    cap.set(3, 1920);
    cap.set(4, 1080);

    Mat base;
    cap >> base;

    //448
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
    int noise_pipe[2];
    pipe(cap_pipe);
    pipe(show_pipe);
    pipe(ptr_pipe);
    pipe(noise_pipe);

    signal(SIGPIPE, sigpipe_hdl);

    struct ctxs ctx = { 
        .cap = &cap,
        .cap_w_fd = cap_pipe[1], 
        .sampleMat = &base ,
        .show_r_fd = show_pipe[0],
        .ptr_w_fd = ptr_pipe[1],
        .ptr_r_fd = ptr_pipe[0],
        .noise_w_fd = noise_pipe[1],
        .noise_r_fd = noise_pipe[0]
    };

    pthread_t th_cap;
    pthread_t th_show;
    pthread_t th_noise;
    pthread_create(&th_cap, NULL, th_cap_hdl, &ctx);
    pthread_create(&th_show, NULL, th_show_hdl, &ctx);
    pthread_create(&th_noise, NULL, th_noise_hdl, &ctx);

    float* k = (float*)malloc(K*K*sizeof(float));
    computeGaussianKernel(k, (1+K*K)/6.0);

    struct mat_ready *pready;
    struct mat_show *show_ready;
    for (int i = 0; i < 6; i++) {

        pready = new struct mat_ready;
        // frame for the noise
        show_ready = new struct mat_show;

#ifdef GPU
        uint8_t *h_frame, *n_frame, *o_frame;
        cudaMallocHost((void **)&h_frame, 3 * ctx.sampleMat->rows * ctx.sampleMat->cols * sizeof *h_frame + sizeof(chunk_t));
        cudaMallocHost((void **)&n_frame, 3 * ctx.sampleMat->rows * ctx.sampleMat->cols * sizeof *n_frame + sizeof(chunk_t));
        cudaMallocHost((void **)&o_frame, 3 * ctx.sampleMat->rows * ctx.sampleMat->cols * sizeof *o_frame + sizeof(chunk_t));
        cudaMallocHost((void **)&pready->h_xs, 3 * ctx.sampleMat->rows * ctx.sampleMat->cols * sizeof *pready->h_xs + sizeof(chunk_t));
        pready->pframe = new Mat(ctx.sampleMat->rows, ctx.sampleMat->cols, ctx.sampleMat->type(), h_frame);
        show_ready->nframe = new Mat(ctx.sampleMat->rows, ctx.sampleMat->cols, ctx.sampleMat->type(), n_frame);
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
    uint8_t *d_current, *d_previous, *d_filtered;
    uint8_t *d_diff;
    uint8_t *d_noise_visualization;
    int *d_xs;
    unsigned int *d_pos;

    cudaGetDeviceProperties(&prop, 0);

    cudaMemcpyToSymbol(dev_k, k, K*K * sizeof(float));
    cudaMalloc((void **)&d_diff, total * sizeof *d_diff);
    cudaMalloc((void **)&d_xs, total * sizeof *d_xs);
    cudaMalloc((void **)&d_current, total * sizeof *d_current);
    cudaMalloc((void **)&d_previous, total * sizeof *d_previous);
    cudaMalloc((void **)&d_filtered, total * sizeof *d_filtered);
    cudaMalloc((void **)&d_noise_visualization, total * sizeof *d_noise_visualization);

    cudaMalloc((void **)&d_pos, sizeof *d_pos);

	int nMaxThreads = prop.maxThreadsPerBlock;
    int maxAtTime = total / nMaxThreads;
    cudaMemcpy(d_current, ctx.sampleMat->data, total * sizeof *ctx.sampleMat->data, cudaMemcpyHostToDevice);

    int max4 = ceil(1.0 * maxAtTime / sizeof(chunk_t));
    dim3 blockSize, gridSize;
    blockSize.x = BLOCK_SIZE, blockSize.y = BLOCK_SIZE, blockSize.z = 1;
    gridSize.x = ceil((float)1920/TILE_SIZE),
    gridSize.y = ceil((float)1080/TILE_SIZE),
    gridSize.z = 1;

    // struct cb_args *pargs = new struct cb_args;
    // pargs->d_pos = d_pos;
    // pargs->show_w_fd = show_pipe[1];

    // TODO::: try other way 
    // smaller i means less number of threads but higher memory transfer / s maybe (?)
    // report: discuss what is better

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

        #ifdef NOISE_FILTER
        cudaMemcpyAsync(d_filtered, pready->pframe->data, total, cudaMemcpyHostToDevice);
        convolution_kernel<<<gridSize, blockSize>>>(d_filtered, d_current);
        #else
        cudaMemcpyAsync(d_current, pready->pframe->data, total, cudaMemcpyHostToDevice);
        #endif

        // Noise visualization
        #ifdef NOISE_VISUALIZER
            #if NOISE_VISUALIZER == 1         
            heat_map<<<1, nMaxThreads, 0>>>(d_current, d_previous, max4, d_noise_visualization);
            #elif NOISE_VISUALIZER == 2
            red_black_map<<<1, nMaxThreads, 0>>>(d_current, d_previous, max4, d_noise_visualization);
            #endif
        #endif
        // cudaMemcpy(d_text, overImageText.c_str(), overImageText.length(), cudaMemcpyHostToDevice);
        // kernel3_char<<<1, nThreadsToUse, (sizeof(CHARS_STR) - 1) * fullArea>>>(d_current, d_charsPx, d_text, overImageText.length(), eachThreadDoes, 3 * txtsz.width, 3 * pready->pframe->cols, fullArea);
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

        #if defined(NOISE_VISUALIZER) && NOISE_VISUALIZER != 0
        // Copy noise frame into the Mat
        cudaMemcpyAsync(show_ready->nframe->data, d_noise_visualization, total, cudaMemcpyDeviceToHost);
       
        write(noise_pipe[1], &show_ready, sizeof show_ready);
        #endif

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