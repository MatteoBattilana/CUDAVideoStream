#include <stdio.h>
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
#include <chrono>

#define NSTREAMS 1

using namespace cv;

struct args_s {
    int fd;
    VideoCapture *cap;
    Mat *frame;
    int total;
};

struct px_df {
    int x;
    uint8_t diff;
};

struct ctx {
    uint8_t *current;
    uint8_t *previous;
    uint8_t *diff;
    int maxSect;
    unsigned int *pos;
};

static bool do_cont = true;

__global__ void kernel(uint8_t *current, uint8_t *previous, uint8_t *diff, int maxSect, unsigned int *pos) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    // int df, npos;

    // printf("%d .. %d\n", x * maxSect, x * maxSect + maxSect);
    // printf("%d\n", x);

    int max = x * maxSect + maxSect;
    for (int i = x * maxSect; i < max; i++) {
        diff[i] = current[i] - previous[i];
        // xs[i] = 0;
        // if (true) {
        //     npos = atomicInc(pos, 6220801);
        //     diff[npos].x = i;
        //     diff[npos].diff = df;
        //     printf("%u] %d %d\n", npos, i, df);
        // }
    }
}

void sighdl(int sig) {
	do_cont = false;
}

// void *th_hdl(void *args_ptr) {

//     struct args_s *args = (struct args_s *)args_ptr;

//     do {
//         *args->cap >> *args->frame;
//         write(args->fd, args->frame->data, args->total * sizeof *args->frame->data);
//     } while(!args->frame->empty());

// }

int main() {
    struct cudaDeviceProp prop;
    struct epoll_event ev, events[10];
    struct addrinfo *result, *rp;
    int i;
    int sfd, epollfd, nfds, sfd2;
    uint8_t *d_current, *d_previous;
    uint8_t *d_diff;
    int *d_xs;
    unsigned int *d_pos, h_pos;
    cudaStream_t streams[NSTREAMS];
    int startx;
    int starty;
    int maxn;

    cudaGetDeviceProperties(&prop, 0);

    for (i = 0; i < NSTREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

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
        return -1;
    }

	signal(SIGINT, sighdl);

    VideoCapture cap;
    if (!cap.open(0, CAP_V4L)) return 1;

    auto codec = cv::VideoWriter::fourcc('M','J','P','G');
    cap.set(cv::CAP_PROP_FOURCC, codec);

    cap.set(3, 1920);
    cap.set(4, 1080);

    Mat base;
    cap >> base;

    printf("suck it- c %d, r %d\n", base.cols, base.rows);


    const int total = 3 * base.cols * base.rows;
    // const int total = 3 * 1920 * 1080;
    // const int total = sz;
    cudaMalloc((void **)&d_diff, total * sizeof *d_diff);
    cudaMalloc((void **)&d_xs, total * sizeof *d_xs);
    cudaMalloc((void **)&d_current, total * sizeof *d_current);
    cudaMalloc((void **)&d_previous, total * sizeof *d_previous);

    cudaMalloc((void **)&d_pos, sizeof *d_pos);
    cudaMemset((void *)d_pos, 0, sizeof *d_pos);
   
    // uint8_t *h_buffer = new uint8_t[total];
    uint8_t *h_buffer;
    cudaMallocHost((void **)&h_buffer, total * sizeof *h_buffer);

    int *h_xs;
    cudaMallocHost((void **)&h_xs, total * sizeof *h_xs);

    // uint8_t *h_frame;
    // cudaMallocHost((void **)&h_frame, total);
    Mat frame(base.rows, base.cols, base.type(), h_buffer);

    int maxAtTime = total / prop.maxThreadsPerBlock;

    startx = 400;
    starty = 400;
    maxn = 400;


    // int pipe_fd[2];
    // pipe(pipe_fd);
    // pthread_t thid;
    // struct args_s args = {.fd = pipe_fd[1], .cap = &cap, .frame = &frame, .total = total};
    // pthread_create(&thid, NULL, th_hdl, (void *)&args);

    while (do_cont) {
        printf("wait..\n");
        nfds = epoll_wait(epollfd, events, 10, -1);
        printf("yes!\n");

        for (i = 0; i < nfds; i++) {
            if (events[i].data.fd == sfd) {
                sfd2 = accept(sfd, NULL, NULL);
                perror("AH!");

                ev.data.fd = sfd2;
                epoll_ctl(epollfd, 1, sfd2, &ev);

                int fps = 0;
                bool started = false;
                bool skip = false;
                int part = 0;
                int offset;
                while (1) {

                    auto begin = std::chrono::high_resolution_clock::now();

					if (!do_cont) {
						printf("\n closing!\n");
						close(sfd2);
						close(sfd);
						exit(0);
					}

                    auto begin2 = std::chrono::high_resolution_clock::now();
                    // read(pipe_fd[0], frame.data, total * sizeof *frame.data);

                    cap >> frame;
                    // cvtColor(frame.data, frame.data, YUY)
                    if (frame.empty()) {
                        break;  // end of video stream
                    }
                    // grab_frame(fd, sz, h_frame);
                    
                    auto end2 = std::chrono::high_resolution_clock::now();

                    auto begin3 = std::chrono::high_resolution_clock::now();
                    if (started) {

                        uint8_t *d_prev = d_current;
                        d_current = d_previous;
                        d_previous = d_prev;

                        for (int i = 0; i < maxn; i++) {
                            cudaMemcpyAsync(d_current, &frame.data[(i + starty) * 1920 + startx], maxn, cudaMemcpyHostToDevice, streams[0]);
                        }

                        int todo = maxn/prop.maxThreadsPerBlock;

                        kernel<<<1, prop.maxThreadsPerBlock, 0, streams[0]>>>(d_current, d_previous, d_diff, maxAtTime, d_pos);
                        cudaMemcpyAsync(h_buffer, d_diff, total * sizeof *d_diff, cudaMemcpyDeviceToHost, streams[0]);
                        // cudaMemcpyAsync(h_xs, d_xs, total * sizeof *d_xs, cudaMemcpyDeviceToHost, streams[0]);

                    } else {
                        cudaMemcpy(d_current, frame.data, total * sizeof *frame.data, cudaMemcpyHostToDevice);

                        // for (int k = 0; k < total; k++) {
                        //     h_buffer[k] = frame.data[k];
                        // }

                        memcpy(h_buffer, frame.data, total * sizeof *frame.data);
                    }

                    cudaDeviceSynchronize();
                    auto end3 = std::chrono::high_resolution_clock::now();

                    auto begin4 = std::chrono::high_resolution_clock::now();
                    // cudaMemcpy((void *)h_pos, (void *)d_pos, sizeof *d_pos, cudaMemcpyDeviceToHost);
                    // printf("pos: %u\n", h_pos);
                    write(sfd2, h_buffer, total * sizeof *h_buffer);
                    // write(sfd2, h_xs, total * sizeof *h_xs);
                    auto end4 = std::chrono::high_resolution_clock::now();

                    auto begin5 = std::chrono::high_resolution_clock::now();
                    started = true;
                    auto end5 = std::chrono::high_resolution_clock::now();

                    auto end = std::chrono::high_resolution_clock::now();
                    auto elaps = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
                    auto elaps2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - begin2);
                    auto elaps3 = std::chrono::duration_cast<std::chrono::nanoseconds>(end3 - begin3);
                    auto elaps4 = std::chrono::duration_cast<std::chrono::nanoseconds>(end4 - begin4);
                    auto elaps5 = std::chrono::duration_cast<std::chrono::nanoseconds>(end5 - begin5);

					printf("\rFPS: %5.0f\tFOR: %6.2f ms\tWR: %6.2f ms\tCAP: %6.2f ms\tCL: %6.2f", 1 / ((float)elaps.count() * 1e-9), elaps3.count() * 1e-6, elaps4.count() * 1e-6, elaps2.count() * 1e-6, elaps5.count() * 1e-6);
					fflush(stdout);
                }
            } else {
                close(events[i].data.fd);
            }
        }
    }

    // int max = 100;
    // while(max--) {
    //     clock_t start = clock();
    //     memset(h_buffer, 0, total * sizeof *h_buffer);

    //     cudaMemcpy(d_buffer, h_buffer, total * sizeof *h_buffer, cudaMemcpyHostToDevice);
    //     kernel<<<3, prop.maxThreadsPerBlock>>>(d_buffer, maxAtTime / 3);
    //     cudaMemcpy(h_buffer, d_buffer, total * sizeof *h_buffer, cudaMemcpyDeviceToHost);

    //     cudaDeviceSynchronize();
    //     fprintf(stderr, "\rexec time: %.2f ms", (float(clock()) - start) / CLOCKS_PER_SEC * 1e3);

    //     for (int i = 0; i < total * sizeof *h_buffer; i++) {
    //         if (h_buffer[i] != 1) {
    //             fprintf(stderr, "\nwtf!!\n");
    //             exit(1);
    //         }
    //     }
    // }

    // fprintf(stderr, "\n");

    return 0;
}