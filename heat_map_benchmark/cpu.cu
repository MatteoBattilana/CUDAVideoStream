#include <stdio.h>
#include "opencv2/opencv.hpp"
using namespace cv;
#include <cmath>
#include <chrono>
#include <time.h>

#define SINE 
#define H 1080
#define W 1920
#define C 3

struct HeatElemt {
    int r;
    int g;
    int b;
};

HeatElemt getHeatPixel(int diff){
    struct HeatElemt h;
    #ifdef SINE
        float diff1 = diff/(255.0*2.0);
        h.r = min(max(sin(M_PI*diff1 - M_PI/2.0)*255.0, 0.0),255.0);
        h.g = min(max(sin(M_PI*diff1)*255.0, 0.0),255.0);
        h.b = min(max(sin(M_PI*diff1 + M_PI/2.0)*255.0, 0.0),255.0);
    #else
        h.r = diff > 30 ? 255: 0;
        h.g = 0;
        h.b = 0;
    #endif

    return h;
}

int main(void)
{
    Mat image1, image2;
    VideoCapture cap;
    if (!cap.open("/dev/video0")) return 1;
    auto codec = cv::VideoWriter::fourcc('M','J','P','G');
    cap.set(cv::CAP_PROP_FOURCC, codec);
    cap.set(3, W);
    cap.set(4, H);
    cap >> image1;

    while (1) {
        cap >> image2;
        if (image2.empty()) {
            break;  // end of video stream
        }

        namedWindow("Original", WINDOW_GUI_NORMAL);
        imshow("Original", image1);
        if (waitKey(10) == 27) {
            break;  // stop capturing by pressing ESC
        }

        // Generate heat map
        auto start = std::chrono::high_resolution_clock::now();
        for (int y = 0; y < H; y++){
            for (int x = 0; x < W; x++){
                Vec3b & intensity = image1.at<Vec3b>(y, x);

                Vec3b a = image1.at<Vec3b>(y, x);
                Vec3b b = image2.at<Vec3b>(y, x);
                HeatElemt elem = getHeatPixel(abs(a.val[0] - b.val[0]) + abs(a.val[1] - b.val[1]) + abs(a.val[2] - b.val[2]));
                
                intensity.val[0] = elem.b;
                intensity.val[1] = elem.g;
                intensity.val[2] = elem.r;
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto elaps = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        printf("\rHeatmap time generation: %.3f ms", (float)elaps.count() * 1e-6);
        fflush(stdout);

        namedWindow("HeatMap", WINDOW_GUI_NORMAL);
        imshow("HeatMap", image1);
        if (waitKey(10) == 27) {
            break;  // stop capturing by pressing ESC
        }
        image1 = image2.clone();
    }
    waitKey();
    waitKey();
    waitKey();
}
