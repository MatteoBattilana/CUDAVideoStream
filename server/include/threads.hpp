#ifndef THREADS_HPP_
#define THREADS_HPP_

#include <pthread.h>
#include "utils.hpp"

namespace diff {

    namespace threads {

        struct preadymin {
            uint8_t *data;
            unsigned int *h_pos;
            int *h_xs;
            void *__ptr;
        };

        class ThreadsCore {

        private:
            int cap_pipe[2];
            int show_pipe[2];
            int ptr_pipe[2];
            int noise_pipe[2];
            void *pcap;
            void *pctx;
            void *pbase;
            void *pshowready;
            pthread_t th_cap;
            pthread_t th_show;
            pthread_t th_noise;
            diff::utils::matsz frameSz;
            diff::utils::matsz charSz;
            uint8_t *charsPx;

        public:
            ThreadsCore();
            diff::utils::matsz getFrameSize();
            diff::utils::matsz getCharSize();
            uint8_t *getCharsPx();
            uint8_t *getBaseFrameData();
            uint8_t *getShowReadyNData();
            void readCap(struct preadymin& minready);
            void writeNoise();
            void writeShow(struct preadymin& minready);

        };

    }

}


#endif