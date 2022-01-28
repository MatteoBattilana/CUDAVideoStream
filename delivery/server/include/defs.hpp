#ifndef DEFS_H_
#define DEFS_H_

#include "opencv2/opencv.hpp"

using namespace cv;

namespace diff {
    namespace threads {

        namespace defs {

            struct mat_show {
                Mat *nframe;
            };

            struct mat_ready {
                Mat *pframe;
                int *h_xs;
                unsigned int h_pos;
            };

            // struct cb_args {
            //     unsigned int *d_pos;
            //     int show_w_fd;
            //     struct mat_ready *pready;
            // };

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

        }

    }

}

#endif