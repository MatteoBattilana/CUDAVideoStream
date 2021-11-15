#include <stdio.h>
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

using namespace cv;

static bool do_cont = true;

__global__ void kernel(uint8_t *current, uint8_t *previous, uint8_t *diff, int maxSect) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;

    // printf("%d .. %d\n", x * maxSect, x * maxSect + maxSect);
    // printf("%d\n", x);

    int max = x * maxSect + maxSect;
    for (int i = x * maxSect; i < max; i++) {
        diff[i] = current[i] - previous[i];
    }
}

void sighdl(int sig) {
	do_cont = false;
}

int main() {
    struct cudaDeviceProp prop;
    struct epoll_event ev, events[10];
    struct addrinfo *result, *rp;
    int i, j, max;
    int sfd, epollfd, nfds, sfd2;
    uint8_t *d_diff, *d_current, *d_previous;

    cudaGetDeviceProperties(&prop, 0);

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
    if (!cap.open("video.mp4")) return 1;

    Mat previous, frame;
    cap >> frame;

    const int total = 3 * frame.cols * frame.rows;
    cudaMalloc((void **)&d_diff, total * sizeof *d_diff);
    cudaMalloc((void **)&d_current, total * sizeof *d_current);
    cudaMalloc((void **)&d_previous, total * sizeof *d_previous);
    uint8_t *h_buffer = new uint8_t[total];
    int maxAtTime = total / prop.maxThreadsPerBlock;

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

                max = 0;
                int fps = 0;
                clock_t start = clock();
                while (1) {

					if (!do_cont) {
						printf("\n closing!\n");
						close(sfd2);
						close(sfd);
						exit(0);
					}

					cap >> frame;
					if (frame.empty()) {
                        break;  // end of video stream
                    }

					imshow("this is you, smile! :)", frame);
					if (waitKey(10) == 27) {
						break;  // stop capturing by pressing ESC
                    }

					clock_t start2 = clock();
                    if (!previous.empty()) {
                        // cudaMemset((void *)d_diff, 0, total * sizeof *d_diff); // optional
                        cudaMemcpy(d_current, frame.data, total * sizeof *frame.data, cudaMemcpyHostToDevice);
                        cudaMemcpy(d_previous, previous.data, total * sizeof *frame.data, cudaMemcpyHostToDevice);
                        kernel<<<3, prop.maxThreadsPerBlock>>>(d_current, d_previous, d_diff, maxAtTime / 3);
                        cudaMemcpy(h_buffer, d_diff, total * sizeof *h_buffer, cudaMemcpyDeviceToHost);
                    } else {
                        memcpy(h_buffer, frame.data, total * sizeof *frame.data);
                    }

                    cudaDeviceSynchronize();
                    clock_t end2 = clock();

					clock_t start3 = clock();
                    write(sfd2, h_buffer, total * sizeof *h_buffer);
                    clock_t end3 = clock();

					previous = frame.clone();

					printf("\rFPS: %5.2f\tFOR: %6.2f ms\tWR: %6.2f ms", ++fps / (float(clock() - start) / CLOCKS_PER_SEC), (float(end2) - start2) / CLOCKS_PER_SEC * 1e3,  (float(end3) - start3) / CLOCKS_PER_SEC * 1e3);
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