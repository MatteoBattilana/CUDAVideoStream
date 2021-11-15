#include <fcntl.h>
#include <math.h>
#include <netdb.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include <signal.h>

#include "opencv2/opencv.hpp"

using namespace cv;

static bool do_cont = true;

void sighdl(int sig) {
	do_cont = false;
}

int main(int argc, char** argv) {
    struct epoll_event ev, events[10];
    struct addrinfo *result, *rp;
    int i, j, max;
    int sfd, epollfd, nfds, sfd2;

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

    uint8_t* buffer = new uint8_t[3 * 1080 * 1920];
    uint8_t* cumulative = new uint8_t[3 * 1080 * 1920];
    VideoCapture cap;
    // open the default camera, use something different from 0 otherwise;
    // Check VideoCapture documentation.
    if (!cap.open("video2.mp4")) return 0;
    Mat previous;

    Mat frame;
    cap >> frame;

    printf("%d %d\n", frame.rows, frame.cols);
    if (frame.empty()) return 1;

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
					fps++;
					if (frame.empty()) break;  // end of video stream
					imshow("this is you, smile! :)", frame);
					if (waitKey(10) == 27)
						break;  // stop capturing by pressing ESC
					int diff_pixel_count = 0;
					// compute difference
					int pos = 0;
					int pos2 = 0;
					clock_t start2 = clock();
					for (int i = 0; i < frame.rows; i++) {
						for (int j = 0; j < frame.cols; j++) {
							unsigned char* p = frame.ptr(i, j);  // Y first, X after
							unsigned char* p2 = previous.ptr(i, j);
							// printf("%d\n", p[0]);
							if (!previous.empty()) {
								uint8_t r = p[0] - p2[0];
								uint8_t g = p[1] - p2[1];
								uint8_t b = p[2] - p2[2];
								cumulative[pos2] += r;
								cumulative[pos2 + 1] += g;
								cumulative[pos2 + 2] += b;

								if (abs(cumulative[pos2]) + abs(cumulative[pos2 + 1]) + abs(cumulative[pos2 + 2]) > 25) {
									diff_pixel_count++;
									buffer[pos++] = cumulative[pos2];
									buffer[pos++] = cumulative[pos2 + 1];
									buffer[pos++] = cumulative[pos2 + 2];
									cumulative[pos2] = 0;
									cumulative[pos2 + 1] = 0;
									cumulative[pos2 + 2] = 0;
								} else {
									buffer[pos++] = 0;
									buffer[pos++] = 0;
									buffer[pos++] = 0;
								}
								pos2 += 3;

							} else {
								// Send new frame and not the difference
								buffer[pos++] = p[0];
								buffer[pos++] = p[1];
								buffer[pos++] = p[2];
							}
						}
					}

					clock_t start3 = clock();
					write(sfd2, buffer, 3 * 1080 * 1920 * sizeof *buffer);
					// write(sfd2, frame.data, 3*1080*1920*sizeof
					// *frame.data);
					printf("\rFPS: %.2f\tFOR: %.2f ms\tWR: %.2f ms", fps / (float(clock() - start) / CLOCKS_PER_SEC), (float(clock()) - start2) / CLOCKS_PER_SEC * 1e3,  (float(clock()) - start3) / CLOCKS_PER_SEC * 1e3);
					fflush(stdout);
					previous = frame.clone();

                    // sleep(10);
                }
            } else {
                close(events[i].data.fd);
            }
        }
    }

    freeaddrinfo(result);

    // diff = malloc(bmp1.dip.heigth * bmp1.dip.width * sizeof *diff);
    // diffsign = malloc(bmp1.dip.heigth * bmp1.dip.width * sizeof *diffsign);

    // for (i = 0, max = bmp1.dip.heigth * bmp1.dip.width; i < max; i++) {
    //     diff[i].red = abs(bmp2.pixels[i].red - bmp1.pixels[i].red);
    //     diff[i].green = abs(bmp2.pixels[i].green - bmp1.pixels[i].green);
    //     diff[i].blue = abs(bmp2.pixels[i].blue - bmp1.pixels[i].blue);

    //     diffsign[i].channels.red = bmp2.pixels[i].red < bmp1.pixels[i].red;
    //     diffsign[i].channels.green = bmp2.pixels[i].green <
    //     bmp1.pixels[i].green; diffsign[i].channels.blue = bmp2.pixels[i].blue
    //     < bmp1.pixels[i].blue;

    //     // if (diff[i].red || diff[i].green || diff[i].blue) {
    //     //     printf("\tbmp1 [%04d][%04d] r: %03d g: %03d b: %03d\n", i /
    //     bmp1.dip.width, j % bmp1.dip.width, bmp1.pixels[i].red,
    //     bmp1.pixels[i].green, bmp1.pixels[i].blue);
    //     //     printf("\tbmp2 [%04d][%04d] r: %03d g: %03d b: %03d\n", i /
    //     bmp1.dip.width, j % bmp1.dip.width, bmp2.pixels[i].red,
    //     bmp2.pixels[i].green, bmp2.pixels[i].blue);
    //     //     printf("\tdiff [%04d][%04d] r: %03d g: %03d b: %03d\n", i /
    //     bmp1.dip.width, j % bmp1.dip.width, diff[i].red, diff[i].green,
    //     diff[i].blue);
    //     //     printf("\tsign [%04d][%04d] r: %03d g: %03d b: %03d\n", i /
    //     bmp1.dip.width, j % bmp1.dip.width, diffsign[i].channels.red,
    //     diffsign[i].channels.green, diffsign[i].channels.blue);
    //     //     printf("\n");
    //     // }
    // }

    // for (i = 0; i < bmp1.dip.heigth; i++) {
    //     for (j = 0; j < bmp1.dip.width; j++) {
    //         diff[i * bmp1.dip.width + j].red = abs(bmp2.pixels[i *
    //         bmp1.dip.width + j].red - bmp1.pixels[i * bmp1.dip.width +
    //         j].red); diff[i * bmp1.dip.width + j].green = abs(bmp2.pixels[i *
    //         bmp1.dip.width + j].green - bmp1.pixels[i * bmp1.dip.width +
    //         j].green); diff[i * bmp1.dip.width + j].blue = abs(bmp2.pixels[i
    //         * bmp1.dip.width + j].blue - bmp1.pixels[i * bmp1.dip.width +
    //         j].blue);

    //         diffsign[i * bmp1.dip.width + j].channels.red = bmp2.pixels[i *
    //         bmp1.dip.width + j].red < bmp1.pixels[i * bmp1.dip.width +
    //         j].red; diffsign[i * bmp1.dip.width + j].channels.green =
    //         bmp2.pixels[i * bmp1.dip.width + j].green < bmp1.pixels[i *
    //         bmp1.dip.width + j].green; diffsign[i * bmp1.dip.width +
    //         j].channels.blue = bmp2.pixels[i * bmp1.dip.width + j].blue <
    //         bmp1.pixels[i * bmp1.dip.width + j].blue;

    //         if (diff[i * bmp1.dip.width + j].red || diff[i * bmp1.dip.width +
    //         j].green || diff[i * bmp1.dip.width + j].blue) {
    //             printf("\tbmp1 [%04d][%04d] r: %03d g: %03d b: %03d\n", i, j,
    //             bmp1.pixels[i * bmp1.dip.width + j].red, bmp1.pixels[i *
    //             bmp1.dip.width + j].green, bmp1.pixels[i * bmp1.dip.width +
    //             j].blue); printf("\tbmp2 [%04d][%04d] r: %03d g: %03d b:
    //             %03d\n", i, j, bmp2.pixels[i * bmp1.dip.width + j].red,
    //             bmp2.pixels[i * bmp1.dip.width + j].green, bmp2.pixels[i *
    //             bmp1.dip.width + j].blue); printf("\tdiff [%04d][%04d] r:
    //             %03d g: %03d b: %03d\n", i, j, diff[i * bmp1.dip.width +
    //             j].red, diff[i * bmp1.dip.width + j].green, diff[i *
    //             bmp1.dip.width + j].blue); printf("\tsign [%04d][%04d] r:
    //             %03d g: %03d b: %03d\n", i, j, diffsign[i * bmp1.dip.width +
    //             j].channels.red, diffsign[i * bmp1.dip.width +
    //             j].channels.green, diffsign[i * bmp1.dip.width +
    //             j].channels.blue); printf("\n");
    //         }

    //     }
    // }

    // free(diff);
    // free(diffsign);
    // free(bmp1.pixels);
    // free(bmp2.pixels);
    return 0;
}
