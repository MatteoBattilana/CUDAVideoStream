#include "opencv2/opencv.hpp"
#include <netdb.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

using namespace cv;

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN3(a, b, c) MIN((a), MIN((b), (c)))
#define MAX3(a, b, c) MAX((a), MAX((b), (c)))

uchar HueConversion(float blue, float green, float red, float delta, float maximum) {
    uchar h;
    if (red == maximum) {
        h = 60 * (green - blue) / delta;
    }
    if (green == maximum) {
        h = 60 * (blue - red) / delta + 120;
    }
    if (blue == maximum) {
        h = 60 * (red - green) / delta + 240;
    }
    if (h < 0) {
        h += 360;
    }
    return h;
}

void rgbToHSV(Mat frame) {
    Vec3b hsv;
    for (int rows = 0; rows < frame.rows; rows++) {
        for (int cols = 0; cols < frame.cols; cols++) {
            float blue = frame.at<Vec3b>(rows, cols)[0] / 255.0;  // blue
            float green = frame.at<Vec3b>(rows, cols)[1] / 255.0; // green
            float red = frame.at<Vec3b>(rows, cols)[2] / 255.0;   // red

            float maximum = MAX3(red, green, blue);
            float minimum = MIN3(red, green, blue);

            float delta = maximum - minimum;
            uchar h = HueConversion(blue, green, red, delta, maximum);
            hsv[0] = h / 2;
            uchar s = (delta / maximum) * 255;
            hsv[1] = s;
            float v = (maximum)*255;
            hsv[2] = v;

            frame.at<Vec3b>(rows, cols) = hsv;
        }
    }
}

int main() {
    uint8_t *d_diff, *d_current, *d_previous;
    int r_threshold, g_threshold, b_threshold;
    VideoCapture cap;
    if (!cap.open(0, CAP_V4L2))
        return 1;

    Mat previous, frame, output;
    cap >> frame;

    int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    const int total = 3 * frame_width * frame_height;

    namedWindow("Trackbar");
    createTrackbar("R", "Trackbar", &r_threshold, 255);
    createTrackbar("G", "Trackbar", &g_threshold, 255);
    createTrackbar("B", "Trackbar", &b_threshold, 255);

    int cnt = 0;
    previous = frame.clone();
    while (cap.isOpened()) {
        cnt++;
        imshow("Previous", previous);
        cap >> frame;
        if (frame.empty()) {
            break; // end of video stream
        }

        // imshow("Input", frame);
        if (waitKey(10) == 27) {
            break; // stop capturing by pressing ESC
        }

        cv::Mat diffImage;
        // rgbToHSV(frame);
        if (!previous.empty()) {
            // rgbToHSV(previous);
            cv::absdiff(previous, frame, diffImage);
        } else {
            diffImage = frame.clone();
        }

        Mat clone_of_diff = diffImage.clone();
        imshow("Diff Image", clone_of_diff);

        cv::Mat foregroundMask = cv::Mat::zeros(diffImage.rows, diffImage.cols, CV_8UC1);

        float threshold = 70.0f;
        float dist;

        for (int j = 0; j < diffImage.rows; ++j) {
            for (int i = 0; i < diffImage.cols; ++i) {
                cv::Vec3b pix = diffImage.at<cv::Vec3b>(j, i);

                dist = (pix[0] * pix[0] + pix[1] * pix[1] + pix[2] * pix[2]);
                dist = sqrt(dist);

                if (dist > threshold) {
                    foregroundMask.at<unsigned char>(j, i) = 255;
                }
            }
        }

        int curr_sum;
        Point start_x, end_x, start_y, end_y;
        Point x1(0, 0), x2(10, 10);
        uint8_t row_sum[diffImage.rows];
        for (int j = 0; j < foregroundMask.rows; ++j) {
            curr_sum = 0;
            for (int i = 0; i < foregroundMask.cols; ++i) {
                curr_sum += foregroundMask.at<unsigned char>(j, i);
            }
            row_sum[j] = curr_sum / foregroundMask.rows;
        }

        uint8_t col_sum[diffImage.cols];
        for (int j = 0; j < foregroundMask.cols; ++j) {
            curr_sum = 0;
            for (int i = 0; i < diffImage.rows; ++i) {
                curr_sum += foregroundMask.at<unsigned char>(i, j);
            }
            col_sum[j] = curr_sum / diffImage.cols;
        }

        Mat frame_with_lines = frame.clone();
        int heigth = 0;
        for (int i = 0; i < diffImage.rows; i++) {
            if (row_sum[i] > 35) {
                if (heigth == 0) {
                    start_y.x = 0;
                    start_y.y = i;
                }

                heigth++;

                // cv::Point pt1(0, i);
                // cv::Point pt2(diffImage.cols, i);
                // line(frame_with_lines, pt1, pt2, (255, 255, 255));
            } else {
                end_y.x = diffImage.cols;
                end_y.y = i;
                if (heigth > 15) {
                    rectangle(frame_with_lines, start_y, end_y, (255, 255, 255), 3);
                }
                heigth = 0;
            }
        }

        int length = 0;
        for (int i = 0; i < diffImage.cols; i++) {
            if (col_sum[i] > 35) {
                if (length == 0) {
                    start_x.x = i;
                    start_x.y = 0;
                }
                length++;
                // cv::Point pt1(i, 0);
                // cv::Point pt2(i, diffImage.rows);
                // line(frame_with_lines, pt1, pt2, (255, 255, 255));
            } else {
                end_x.x = i;
                end_x.y = diffImage.rows;
                if (length > 15) {
                    rectangle(frame_with_lines, start_x, end_x, (255, 255, 255), 3);
                }
                length = 0;
            }
        }
        rectangle(frame_with_lines, x1, x2, (255, 255, 255), 3);

        Point top_left(start_x.x, start_y.y);
        Point bottom_right(end_x.x, end_y.y);

        imshow("Diff Image Black and White", foregroundMask);
        imshow("Input", frame_with_lines);
        if (cnt % 2 == 0)
            previous = frame.clone();
    }
    return 0;
}
