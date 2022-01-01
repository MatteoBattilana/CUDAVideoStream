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

    namedWindow("hi", WINDOW_GUI_NORMAL);
    while (1) {

        read(sfd, &pos, sizeof pos);

        btot = 0;
        while (btot < pos * sizeof *xs) {
            btot += read(sfd, (uint8_t *)xs + btot, pos * sizeof *xs - btot);
        }

        btot = 0;
        while (btot < pos * sizeof *buffer) {
            btot += read(sfd, (uint8_t *)buffer + btot, pos * sizeof *buffer - btot);
        }

        for (int i = 0; i < pos; i++) {
            frame2.data[xs[i]] += buffer[i];
        }
        
        imshow("hi", frame2);
        if (waitKey(1) == 27) break;  // stop capturing by pressing ESC
    }

    freeaddrinfo(result);
    return 0;
}
