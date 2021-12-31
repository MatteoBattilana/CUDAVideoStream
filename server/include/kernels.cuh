#ifndef KERNEL_CUH_
#define KERNEL_CUH_

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <stdio.h>
#include "../include/utils.hpp"
#include <string>

namespace diff {

    namespace cuda {

        typedef long4 chunk_t;

        class CUDACore {

        private:
            uint8_t *d_charsPx;
            struct cudaDeviceProp prop;
            uint8_t *d_current, *d_previous, *d_filtered;
            uint8_t *d_diff;
            uint8_t *d_noise_visualization;
            int *d_xs;
            unsigned int *d_pos;
            int nMaxThreads;
            int maxAtTime;
            int max4;
            int total;
            dim3 blockSize, gridSize;
            int nThreadsToUse;
            int eachThreadDoes;
            diff::utils::matsz charsSz;
            diff::utils::matsz frameSz;
            int fullArea;

        public:
            CUDACore(uint8_t *charsPx, diff::utils::matsz& charsSz, float *k, int total, uint8_t *sampleMatData, diff::utils::matsz& frameSz);
            static void alloc_arrays(uint8_t **h_frame, uint8_t **n_frame, uint8_t **o_frame, int **h_xs, int r, int c);
            void exec_core(uint8_t *frameData, uint8_t *showReadyNData, std::string& text, unsigned int *h_pos, int *h_xs);

        };

    }

}

#endif