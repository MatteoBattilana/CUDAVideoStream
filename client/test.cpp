#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdint.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <netdb.h>
#include <sys/epoll.h>
#include "opencv2/opencv.hpp"
#include <time.h>
#include <math.h>

using namespace std;
using namespace cv;

int main() {

    VideoCapture cap("video2.mp4");

    Mat frame;

    while(true) {

        cap >> frame;
        imshow("this is you, smile! :)", frame);
        if (waitKey(10) == 27)
            break; // stop capturing by pressing ESC

    }

    return 0;
}