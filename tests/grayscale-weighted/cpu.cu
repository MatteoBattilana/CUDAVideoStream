#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define H 1080
#define W 1920

int main(int argc, char const *argv[]) {

    // VideoCapture cap(video_file);
    VideoCapture cap(0, CAP_V4L2);

    if (!cap.isOpened())
        cerr << "Error opening video stream\n";

    auto codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    cap.set(cv::CAP_PROP_FOURCC, codec);
    cap.set(3, W);
    cap.set(4, H);

    Mat frame, output;
    output.create(H, W, CV_8UC1);
    int sum = 0;
    while (1) {
        cap >> frame;
        if (frame.empty())
            return 0;

        imshow("input", frame);

        if (waitKey(10) == 27) {
            break; // stop capturing by pressing ESC
        }

        auto start = std::chrono::high_resolution_clock::now();
        for (int row = 0; row < H; row++) {
            for (int col = 0; col < W; col++) {
                output.at<uchar>(row, col) = 0.114 * frame.at<Vec3b>(row, col)[0] + 0.587 * frame.at<Vec3b>(row, col)[1] + 0.299 * frame.at<Vec3b>(row, col)[2];
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto elaps = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        printf("\rHeatmap time generation: %.3f ms", (float)elaps.count() * 1e-6);
        fflush(stdout);
        imshow("output", output);
        if (waitKey(10) == 27) {
            break; // stop capturing by pressing ESC
        }
    }

    cap.release();
    return 0;
}