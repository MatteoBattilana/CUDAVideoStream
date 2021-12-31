#include <stdio.h>
#include <stdint.h>
#include <signal.h>
#include <string>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include "../include/kernels.cuh"
#include "../include/common.h"
#include "../include/threads.hpp"

void sigpipe_hdl(int sig) {
    exit(1);
}

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

    signal(SIGPIPE, sigpipe_hdl);

    float* k = (float*)malloc(K*K*sizeof(float));
    computeGaussianKernel(k, (1+K*K)/6.0);

    diff::threads::ThreadsCore threadsCore;
    int total = 3 * threadsCore.getFrameSize().height * threadsCore.getFrameSize().width;

#ifdef GPU

    // struct cb_args *pargs = new struct cb_args;
    // pargs->d_pos = d_pos;
    // pargs->show_w_fd = show_pipe[1];

    diff::utils::matsz charsSz = threadsCore.getCharSize();
    diff::utils::matsz frameSz = threadsCore.getFrameSize();
    diff::cuda::CUDACore cudaCore(threadsCore.getCharsPx(), charsSz, k, total, threadsCore.getBaseFrameData(), frameSz);

#endif

    // cv::Mat previous = ctx.sampleMat->clone();
    std::string overImageText;
    struct diff::threads::preadymin ready;

    std::cout << "Ready to rock!" << std::endl;

    auto begin0 = std::chrono::high_resolution_clock::now();
    while (1) {
        auto begin = std::chrono::high_resolution_clock::now();

        auto begin2 = std::chrono::high_resolution_clock::now();
        threadsCore.readCap(ready);
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

        cudaCore.exec_core(ready.data, threadsCore.getShowReadyNData(), overImageText, ready.h_pos, ready.h_xs);
        threadsCore.writeNoise();

#endif
        auto end3 = std::chrono::high_resolution_clock::now();

        threadsCore.writeShow(ready);

        auto end = std::chrono::high_resolution_clock::now();

        auto end0 = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::nanoseconds>(end0 - begin0).count() > 1e9) {
            begin0 = end0;

            auto elaps = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
            auto elaps2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - begin2);
            auto elaps3 = std::chrono::duration_cast<std::chrono::nanoseconds>(end3 - begin3);

            float unit = 1 / ((float)elaps.count() * 1e-9);
            int bw = static_cast<int>(((*ready.h_pos)<<4)*unit*1e-3);

            std::stringstream strstream;
            strstream << std::setfill(' ') << std::setw(5) << static_cast<int>(unit);

            printf("\rFPS: %s\tFOR: %5.2f ms\tREAD: %9.2f\tPOS: %7d\t BW: %5d kbps", strstream.str().c_str(), (float)elaps3.count() * 1e-6, (float)elaps2.count() * 1e-6, *ready.h_pos, bw);

            strstream = std::stringstream();
            strstream << "FPS: " << static_cast<int>(unit) << " BW: " << bw << " kbps";
            overImageText = strstream.str();

            fflush(stdout);
        }

    }

    return 0;
}