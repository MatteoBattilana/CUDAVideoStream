#include <stdio.h>
#include <opencv4/opencv2/opencv.hpp>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <netdb.h>
#include <fcntl.h>

using namespace std;
using namespace cv;

int main() {
    struct addrinfo *result, *rp;
    int sfd;
    uint8_t * buffer = new uint8_t[3*1280*720];

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

    Mat frame = imread("Untitled-2.bmp", IMREAD_ANYCOLOR);
    int btot = 0;
    while (btot < 3 * 1280 * 720 * sizeof *buffer) {
        btot += read(sfd, buffer + btot, 3 * 1280 * 720 * sizeof *buffer);
    }

    printf("read\n");


    vector<uint8_t> v(buffer, buffer + 3 * 1280 * 720 * sizeof *buffer);
    Mat frame2 = Mat(720, 1280, frame.type(), &v[0]).clone();
    imshow("hi", frame2);
    waitKey();

    freeaddrinfo(result);
    return 0;
}