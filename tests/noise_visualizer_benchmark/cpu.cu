#include <stdio.h>
#include "opencv2/opencv.hpp"
using namespace cv;
#include <cmath>
#include <chrono>
#include <time.h>

#define H 1080
#define W 1920
#define C 3
#define LR_THRESHOLDS 20

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
                
                if(abs(a.val[0] - b.val[0]) > LR_THRESHOLDS || abs(a.val[1] - b.val[1]) > LR_THRESHOLDS || abs(a.val[2] - b.val[2]) > LR_THRESHOLDS)
                {
                    intensity.val[0] = 0;
                    intensity.val[1] = 0;
                    intensity.val[2] = 255;
                } else {
                    intensity.val[0] = 0;
                    intensity.val[1] = 0;
                    intensity.val[2] = 0;
                }
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
