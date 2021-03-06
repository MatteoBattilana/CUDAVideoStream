#include "../include/common.h"
#include "../include/kernels.cuh"
#include "../include/threads.hpp"
#include "../include/utils.hpp"
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <signal.h>
#include <sstream>
#include <stdint.h>
#include <stdio.h>
#include <string>

void sigpipe_hdl(int sig) {
    exit(1);
}

void computeGaussianKernel(float *k, float sigma) {
    float sum = 0;
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            float x = i - (K - 1) / 2.0;
            float y = j - (K - 1) / 2.0;
            k[i * K + j] = (1.0 / (2.0 * M_PI * sigma * sigma)) * exp(-((x * x + y * y) / (2.0 * sigma * sigma)));
            sum += k[i * K + j];
        }
    }

    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            k[i * K + j] /= sum;
        }
    }
}

int main() {

    signal(SIGPIPE, sigpipe_hdl);

    float *k = (float *)malloc(K * K * sizeof(float));
    computeGaussianKernel(k, (K * K) / 6.0);

    diff::threads::ThreadsCore threadsCore;
    int total = 3 * threadsCore.getFrameSize().height * threadsCore.getFrameSize().width;

    diff::utils::matsz frameSz = threadsCore.getFrameSize();

#ifdef GPU

    diff::utils::matsz charsSz = threadsCore.getCharSize();
    diff::cuda::CUDACore cudaCore(threadsCore.getCharsPx(), charsSz, k, total, threadsCore.getBaseFrameData(), frameSz);

#elif defined(CPU)

    uint8_t *pvs_data = new uint8_t[3 * frameSz.area()];
    uint8_t *pvs = new uint8_t[3 * frameSz.area()];
    std::memcpy(pvs_data, threadsCore.getBaseFrameData(), 3 * frameSz.area());

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

        std::memcpy(pvs_data, ready.data, 3 * frameSz.area());

        // *ready.h_pos = 0;
        // for (int i = 0; i < total; i++) {
        //     int df = ready.data[i] - pvs_data[i];
        //     if (df < -LR_THRESHOLDS || df > LR_THRESHOLDS) {
        //         ready.data[*ready.h_pos] = df;
        //         ready.h_xs[*ready.h_pos] = i;
        //         (*ready.h_pos)++;
        //     } else {
        //         pvs_data[i] -= df;
        //     }
        // }

        // diff::utils::swap(pvs, pvs_data);

        for (int i = 0; i < total; i = i + 3) {
            int sum = ready.data[i] + ready.data[i + 1] + ready.data[i + 2];
            ready.data[i] = sum / 3;
            ready.data[i + 1] = sum / 3;
            ready.data[i + 2] = sum / 3;
        }

        int histogram[256] = {0};
        for (int i = 0; i < total; i = i + 3) {
            histogram[ready.data[i]]++;
        }

        int max = -1, sec_max = -1;
        int index_max = -1, index_sec_max = -1;
        for (int i = 0; i < 256; i++) {
            if (histogram[i] >= max) {
                index_sec_max = index_max;
                index_max = i;
                max = histogram[i];
                sec_max = max;
            } else if (histogram[i] > sec_max && histogram[i] < max) {
                sec_max = histogram[i];
                index_sec_max = i;
            }
        }
        int threshold = (index_max + index_sec_max) / 2;
        if (threshold < 50) {
            threshold = 50;
        }
        if (threshold > 200) {
            threshold = 200;
        }

        for (int i = 0; i < total; i++) {
            if (ready.data[i] > threshold) {
                ready.data[i] = 255;
            } else {
                ready.data[i] = 0;
            }
        }

#elif defined(GPU)

        cudaCore.exec_core(ready.data, threadsCore.getShowReadyNData(), overImageText, ready.h_pos, ready.h_xs);

#endif

        threadsCore.writeNoise();
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
            int bw = static_cast<int>(((*ready.h_pos) << 4) * unit * 1e-3);

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