#include <fcntl.h>
#include <netdb.h>
#include <stdio.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define cols 1920
#define rows 1080

struct px_df {
    int x;
    uint8_t diff;
};

int main() {
    struct addrinfo *result, *rp;
    int sfd;
    uint8_t* buffer = new uint8_t[3 * rows * cols];
    int* xs = new int[3 * rows * cols];

    // struct addrinfo hints = {
    //     .ai_family = AF_INET,
    //     .ai_socktype = SOCK_DGRAM
    // };

    getaddrinfo("127.0.0.1", "2734", NULL, &result);
    for (rp = result; rp != NULL; rp = rp->ai_next) {
        if ((sfd = socket(AF_INET, SOCK_STREAM, 0)) == -1) {
            continue;
        }
        if (connect(sfd, rp->ai_addr, rp->ai_addrlen) != -1) {
            // fcntl(sfd, F_SETFL, fcntl(sfd, F_GETFL, 0) | O_NONBLOCK);
            break;
        }

        close(sfd);
    }

    int iteration = -1;
    Mat frame2 = imread("image.bmp", IMREAD_ANYCOLOR);
    while (1) {
        int btot = 0;
        while (btot < (3 * rows * cols * sizeof *buffer)) {
            btot += read(sfd, buffer + btot, ((3 * rows * cols * sizeof *buffer)) - btot);
        }

        // btot = 0;
        // while (btot < (3 * rows * cols * sizeof *xs)) {
        //     btot += read(sfd, xs + btot, ((3 * rows * cols * sizeof *xs)) - btot);
        // }

        int buffer_id = 0;
        if (iteration == -1) {
            // vector<struct px_df> v(buffer, buffer + 3 * rows * cols * sizeof *buffer);

            // frame2 = Mat(rows, cols, frame2.type(), &v[0]).clone();
            frame2 = Mat(rows, cols, frame2.type());
            for (int k = 0; k < 3 * rows * cols; k++) {
                frame2.data[k] = buffer[k];
            }
            // for (int i = 0; i < 3 * rows * cols; i++) {
            //     printf("%d ", frame2.data[0]);
            // }
            // exit(0);

            // printf("\n");
        } else {

            int total = 3 * rows * cols;
            for (int k = 0; k < total; k++) {
                frame2.data[k] += buffer[buffer_id++];
            }

            // for (int i = offset / frame2.cols; i < frame2.rows; i++) {
            //     for (int j = offset % frame2.cols; j < frame2.cols; j++) {
            //         unsigned char* p = frame2.ptr(i, j);  // Y first, X after
            //         p[0] += buffer[buffer_id++];
            //         p[1] += buffer[buffer_id++];
            //         p[2] += buffer[buffer_id++];
            //     }
            // }
        }

        namedWindow("hi", WINDOW_GUI_NORMAL);
        imshow("hi", frame2);
        if (waitKey(10) == 27) break;  // stop capturing by pressing ESC
        iteration++;
        iteration %= 2;
    }
    freeaddrinfo(result);
    return 0;
}