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
    unsigned int pos;

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

    printf("reading base\n");
    int btot = 0;
    while (btot < (3 * rows * cols * sizeof *buffer)) {
        btot += read(sfd, buffer + btot, ((3 * rows * cols * sizeof *buffer)) - btot);
    }

    frame2 = Mat(rows, cols, frame2.type());
    for (int k = 0; k < 3 * rows * cols; k++) {
        frame2.data[k] = buffer[k];
    }


    while (1) {

        // printf("recving.\n");
        read(sfd, &pos, sizeof pos);

        btot = 0;
        while (btot < pos * sizeof *xs) {
            btot += read(sfd, (uint8_t *)xs + btot, pos * sizeof *xs - btot);
        }

        btot = 0;
        while (btot < pos * sizeof *buffer) {
            btot += read(sfd, (uint8_t *)buffer + btot, pos * sizeof *buffer - btot);
        }

        int total = 3 * rows * cols;
        for (int i = 0; i < pos; i++) {
            // if (xs[i] != 0) {
            //     printf("RECV xs[%d]=%d, b[%d]=%d\n", i, xs[i], i, buffer[i]);
            // }
            frame2.data[xs[i]] += buffer[i];
        }
        

        namedWindow("hi", WINDOW_GUI_NORMAL);
        imshow("hi", frame2);
        if (waitKey(10) == 27) break;  // stop capturing by pressing ESC
        // iteration++;
        // iteration %= 2;
    }
    freeaddrinfo(result);
    return 0;
}