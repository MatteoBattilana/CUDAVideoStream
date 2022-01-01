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
#include <exception>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "../include/threads.hpp"
#include "../include/defs.hpp"
#include "../include/common.h"
#include "../include/kernels.cuh"

#define CCAP(__pcap) ((cv::VideoCapture *)__pcap)
#define CBASE(__pbase) ((cv::Mat *)__pbase)
#define CSHOWNDATA(__pshowready) ((struct mat_show *)__pshowready)

using namespace diff::threads;
using namespace diff::threads::defs;

static void *th_noise_hdl(void *args);
static void *th_cap_hdl(void *args);
static void *th_show_hdl(void *args);

ThreadsCore::ThreadsCore() {

    this->pcap = new cv::VideoCapture;
    if (!CCAP(pcap)->open(0, cv::CAP_V4L2)) throw new std::exception();
    auto codec = cv::VideoWriter::fourcc('M','J','P','G');
    CCAP(pcap)->set(cv::CAP_PROP_FOURCC, codec);

    CCAP(pcap)->set(3, 1920);
    CCAP(pcap)->set(4, 1080);

    cv::Mat base;
    *CCAP(pcap) >> base;
    this->pbase = new cv::Mat(base);

    auto txtsz = cv::getTextSize("A", cv::FONT_HERSHEY_PLAIN, 3, 2, 0);
    std::cout << "Character pixel size: " << txtsz << std::endl;
    int fullArea = 3 * txtsz.area();

    charsPx = new uint8_t[(sizeof(CHARS_STR) - 1) * fullArea];
    memset(charsPx, 0x00, (sizeof(CHARS_STR) - 1) * fullArea);

    for (int i = 0; i < (sizeof(CHARS_STR) - 1); i++) {
        cv::Mat pxBaseMat(txtsz.height, txtsz.width, base.type(), charsPx + i * fullArea);
        cv::putText(pxBaseMat, std::string(CHARS_STR).substr(i, 1), cv::Point(0, txtsz.height+1), cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    }

    this->frameSz = diff::utils::matsz(base.rows, base.cols);
    this->charSz = diff::utils::matsz(txtsz.height, txtsz.width);

    pipe(cap_pipe);
    pipe(show_pipe);
    pipe(ptr_pipe);
    pipe(noise_pipe);

    struct ctxs ctx = {
        .cap = CCAP(pcap),
        .cap_w_fd = cap_pipe[1],
        .sampleMat = CBASE(pbase),
        .show_r_fd = show_pipe[0],
        .ptr_w_fd = ptr_pipe[1],
        .ptr_r_fd = ptr_pipe[0],
        .noise_w_fd = noise_pipe[1],
        .noise_r_fd = noise_pipe[0]
    };

    this->pctx = new struct ctxs(ctx);

    pthread_create(&th_cap, NULL, th_cap_hdl, this->pctx);
    pthread_create(&th_show, NULL, th_show_hdl, this->pctx);
    pthread_create(&th_noise, NULL, th_noise_hdl, this->pctx);

    struct mat_ready *pready;
    struct mat_show *show_ready;
    for (int i = 0; i < 6; i++) {

        pready = new struct mat_ready;
        // frame for the noise
        show_ready = new struct mat_show;
        this->pshowready = show_ready;

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

}

diff::utils::matsz diff::threads::ThreadsCore::getFrameSize() {
    return this->frameSz;
}

diff::utils::matsz diff::threads::ThreadsCore::getCharSize() {
    return this->charSz;
}

uint8_t *diff::threads::ThreadsCore::getCharsPx() {
    return this->charsPx;
}

uint8_t *diff::threads::ThreadsCore::getShowReadyNData() {
    return CSHOWNDATA(this->pshowready)->nframe->data;
}

uint8_t *diff::threads::ThreadsCore::getBaseFrameData() {
    return CBASE(this->pbase)->data;
}

void diff::threads::ThreadsCore::writeNoise() {
    write(noise_pipe[1], &this->pshowready, sizeof *CSHOWNDATA(this->pshowready));
}

void ThreadsCore::readCap(struct preadymin& minready) {
    struct mat_ready *pready;
    read(cap_pipe[0], &pready, sizeof pready);

    minready.data = pready->pframe->data;
    minready.h_pos = &pready->h_pos;
    minready.h_xs = pready->h_xs;
    minready.__ptr = pready;
}

void ThreadsCore::writeShow(struct preadymin& minready) {
    struct mat_ready *pready = (struct mat_ready *)minready.__ptr;
    write(show_pipe[1], &pready, sizeof pready);
}

static void *th_noise_hdl(void *args) {

#ifdef NOISE_VISUALIZER
    struct diff::threads::mat_show *show_ready;
    struct diff::threads::ctxs *pctx = (struct diff::threads::ctxs *)args;

    while(1) {
        read(pctx->noise_r_fd, &show_ready, sizeof show_ready);
        cv::namedWindow("Noise Visualizer", cv::WINDOW_GUI_NORMAL);
        cv::imshow("Noise Visualizer", *(show_ready->nframe));
        if (cv::waitKey(10) == 27) {
            exit(1);  // stop capturing by pressing ESC
        }
    }
#endif

    return NULL;
}

static void *th_cap_hdl(void *args) {

    struct diff::threads::defs::mat_ready *pready;
    struct diff::threads::defs::ctxs *pctx = (struct diff::threads::defs::ctxs *)args;

    read(pctx->ptr_r_fd, &pready, sizeof pready);
    while(1) {
        *pctx->cap >> *(pready->pframe);
        write(pctx->cap_w_fd, &pready, sizeof pready);
        read(pctx->ptr_r_fd, &pready, sizeof pready);
    }

    return NULL;
}

static void *th_show_hdl(void *args) {
    struct epoll_event ev, events[10];
    struct addrinfo *result, *rp;
    int sfd, epollfd, nfds, sfd2;
    struct diff::threads::defs::mat_ready *pready;

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

    struct diff::threads::defs::ctxs *pctx = (struct diff::threads::defs::ctxs *)args;
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

