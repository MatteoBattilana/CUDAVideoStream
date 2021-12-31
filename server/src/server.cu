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
#include "../include/defs.hpp"
#include "../include/kernels.cuh"

// 1 for heat map, 2 for red-black, 0 nothing
#define NOISE_FILTER
#define K 3
#define TILE_SIZE 10
#define BLOCK_SIZE (TILE_SIZE + K - 1)

// Noise visualizer: 1 heatmap, 2 red-black, 3 red-black overlap
#define NOISE_VISUALIZER 1

#define CHARS_STR "0123456789BFPSWbkps :"
#define LR_THRESHOLDS 20
#define NSTREAMS 1
#define GPU
#define KERNEL2_NEGFEED_OPT


#ifdef GPU


#endif

void *th_noise_hdl(void *args) {

    struct diff::mat_show *show_ready;
    struct diff::ctxs *pctx = (struct diff::ctxs *)args;

    while(1) {
        read(pctx->noise_r_fd, &show_ready, sizeof show_ready);
        cv::namedWindow("Noise Visualizer", cv::WINDOW_GUI_NORMAL);
        cv::imshow("Noise Visualizer", *(show_ready->nframe));
        if (cv::waitKey(10) == 27) {
            exit(1);  // stop capturing by pressing ESC
        }
    }

    return NULL;
}

void *th_cap_hdl(void *args) {

    struct diff::mat_ready *pready;
    struct diff::ctxs *pctx = (struct diff::ctxs *)args;

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
    struct diff::mat_ready *pready;

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

    struct diff::ctxs *pctx = (struct diff::ctxs *)args;
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

    cv::VideoCapture cap;
    if (!cap.open(0, cv::CAP_V4L2)) return 1;
    auto codec = cv::VideoWriter::fourcc('M','J','P','G');
    cap.set(cv::CAP_PROP_FOURCC, codec);

    cap.set(3, 1920);
    cap.set(4, 1080);

    cv::Mat base;
    cap >> base;

    auto txtsz = cv::getTextSize("A", cv::FONT_HERSHEY_PLAIN, 3, 2, 0);
    std::cout << "Character pixel size: " << txtsz << std::endl;
    int fullArea = 3 * txtsz.area();

    uint8_t *charsPx = new uint8_t[(sizeof(CHARS_STR) - 1) * fullArea];
    memset(charsPx, 0x00, (sizeof(CHARS_STR) - 1) * fullArea);

    for (int i = 0; i < (sizeof(CHARS_STR) - 1); i++) {
        cv::Mat pxBaseMat(txtsz.height, txtsz.width, base.type(), charsPx + i * fullArea);
        cv::putText(pxBaseMat, std::string(CHARS_STR).substr(i, 1), cv::Point(0, txtsz.height+1), cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    }

#ifdef GPU

    // uint8_t *c_matrixPtr;
    // auto x = cudaMemcpyToSymbol(c_matrix, charsPx, totcpy);
    // cudaGetSymbolAddress((void **)&c_matrixPtr, c_matrix);

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

    struct diff::ctxs ctx = { 
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

    struct diff::mat_ready *pready;
    struct diff::mat_show *show_ready;
    for (int i = 0; i < 6; i++) {

        pready = new struct diff::mat_ready;
        // frame for the noise
        show_ready = new struct diff::mat_show;

#ifdef GPU
        uint8_t *h_frame, *n_frame, *o_frame;
        diff::cuda::CUDACore::alloc_arrays(&h_frame, &n_frame, &o_frame, &pready->h_xs, ctx.sampleMat->rows, ctx.sampleMat->cols);
        pready->pframe = new cv::Mat(ctx.sampleMat->rows, ctx.sampleMat->cols, ctx.sampleMat->type(), h_frame);
        show_ready->nframe = new cv::Mat(ctx.sampleMat->rows, ctx.sampleMat->cols, ctx.sampleMat->type(), n_frame);
#else
        pready->pframe = new cv::Mat(ctx.sampleMat->rows, ctx.sampleMat->cols, ctx.sampleMat->type());
        pready->h_xs = new int[3 * ctx.sampleMat->rows * ctx.sampleMat->cols];
#endif

        pready->h_pos = 0;
        write(ctx.ptr_w_fd, &pready, sizeof pready);
    }

    int total = 3 * ctx.sampleMat->rows * ctx.sampleMat->cols;

#ifdef GPU

    // struct cb_args *pargs = new struct cb_args;
    // pargs->d_pos = d_pos;
    // pargs->show_w_fd = show_pipe[1];

    diff::utils::matsz matSz(txtsz.height, txtsz.width);
    diff::utils::matsz frameSz(ctx.sampleMat->rows, ctx.sampleMat->cols);
    diff::cuda::CUDACore cudaCore(charsPx, matSz, k, total, ctx.sampleMat->data, frameSz);

#endif

    cv::Mat previous = ctx.sampleMat->clone();
    std::string overImageText;

    std::cout << "Ready to rock!" << std::endl;

    auto begin0 = std::chrono::high_resolution_clock::now();
    while (1) {
        auto begin = std::chrono::high_resolution_clock::now();

        auto begin2 = std::chrono::high_resolution_clock::now();
        read(cap_pipe[0], &pready, sizeof pready);
        auto end2 = std::chrono::high_resolution_clock::now();

        auto begin3 = std::chrono::high_resolution_clock::now();
#ifdef CPU

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

        cudaCore.exec_core(pready->pframe->data, show_ready->nframe->data, overImageText, &pready->h_pos, pready->h_xs);
        write(noise_pipe[1], &show_ready, sizeof show_ready);

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