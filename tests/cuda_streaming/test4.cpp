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
#include <time.h>
#include <stdint.h>
#include <pthread.h>
#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

// __global__ void kernel(uint8_t *a, uint8_t *b) {

// }

int main() {

    auto txtsz = cv::getTextSize("A", cv::FONT_ITALIC, 3, 2, 0);
    std::cout << txtsz << std::endl;
    cv::Mat img(txtsz, CV_64FC3, cv::Scalar(0)); 
    cv::putText(img, "A", cv::Point(0, txtsz.height), cv::FONT_ITALIC, 3, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    cv::imshow("hi", img);
    while (cv::waitKey(10));

    return 0;
}