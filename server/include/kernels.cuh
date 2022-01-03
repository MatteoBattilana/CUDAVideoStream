#ifndef KERNEL_CUH_
#define KERNEL_CUH_

#include <stdint.h>
#include <stdio.h>
#include <string>
#include "../include/utils.hpp"

namespace diff {

    namespace cuda {

        class CUDACore {

        private:
            uint8_t *d_charsPx;
            uint8_t *d_current, *d_previous, *d_filtered, *d_prevmod;
            uint8_t *d_diff;
            uint8_t *d_noise_visualization;
            int *d_xs;
            unsigned int *d_pos;
            int nMaxThreads;
            int maxAtTime;
            int max4;
            int total;
            void *pblockSize, *pgridSize;
            int nThreadsToUse;
            int eachThreadDoes;
            diff::utils::matsz charsSz;
            diff::utils::matsz frameSz;
            int fullArea;

            void applyOverlay(std::string& text, uint8_t *d_frame);

        public:
            CUDACore(uint8_t *charsPx, diff::utils::matsz& charsSz, float *k, int total, uint8_t *sampleMatData, diff::utils::matsz& frameSz);
            static void alloc_arrays(uint8_t **h_frame, uint8_t **n_frame, uint8_t **o_frame, int **h_xs, int r, int c);
            void exec_core(uint8_t *frameData, uint8_t *showReadyNData, std::string& text, unsigned int *h_pos, int *h_xs);
            size_t chunkt_size();

        };

    }

}

#endif