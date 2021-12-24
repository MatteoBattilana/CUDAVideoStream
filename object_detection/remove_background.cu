#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <sstream>

using namespace cv;
using namespace std;
const char *params = "{ help h         |           | Print usage }"
                     "{ input          | vtest.avi | Path to a video or a sequence of image }"
                     "{ algo           | KNN      | Background subtraction method (KNN, MOG2) }";
int main(int argc, char *argv[]) {
    CommandLineParser parser(argc, argv, params);
    parser.about("This program shows how to use background subtraction methods provided by "
                 " OpenCV. You can process both videos and images.\n");
    if (parser.has("help")) {
        // print help information
        parser.printMessage();
    }
    // create Background Subtractor objects
    Ptr<BackgroundSubtractor> pBackSub;
    if (parser.get<String>("algo") == "MOG2")
        pBackSub = createBackgroundSubtractorMOG2();
    else
        pBackSub = createBackgroundSubtractorKNN();
    VideoCapture capture(samples::findFile(parser.get<String>("input")));
    // VideoCapture capture(0, CAP_V4L2);
    if (!capture.isOpened()) {
        // error in opening the video input
        cerr << "Unable to open: " << parser.get<String>("input") << endl;
        return 0;
    }
    Mat frame, diffImage;
    while (true) {
        capture >> frame;
        if (frame.empty())
            break;
        // update the background model
        pBackSub->apply(frame, diffImage);
        // get the frame number and write it on the current frame
        // rectangle(frame, cv::Point(10, 2), cv::Point(100, 20),
        //           cv::Scalar(255, 255, 255), -1);
        // stringstream ss;
        // ss << capture.get(CAP_PROP_POS_FRAMES);
        // string frameNumberString = ss.str();
        // putText(frame, frameNumberString.c_str(), cv::Point(15, 15),
        //         FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        // show the current frame and the fg masks
        imshow("Frame", frame);

        int curr_sum;
        uint8_t row_sum[diffImage.rows];
        for (int j = 0; j < diffImage.rows; ++j) {
            curr_sum = 0;
            for (int i = 0; i < diffImage.cols; ++i) {
                curr_sum += diffImage.at<unsigned char>(j, i);
            }
            row_sum[j] = curr_sum / diffImage.rows;
        }

        uint8_t col_sum[diffImage.cols];
        for (int j = 0; j < diffImage.cols; ++j) {
            curr_sum = 0;
            for (int i = 0; i < diffImage.rows; ++i) {
                curr_sum += diffImage.at<unsigned char>(i, j);
            }
            col_sum[j] = curr_sum / diffImage.cols;
        }

        Mat frame_with_lines = frame.clone();
        for (int i = 0; i < diffImage.rows; i++) {
            if (row_sum[i] > 10) {
                cv::Point pt1(0, i);
                cv::Point pt2(diffImage.cols, i);
                line(frame_with_lines, pt1, pt2, (255, 255, 255));
            }
        }

        for (int i = 0; i < diffImage.cols; i++) {
            if (col_sum[i] > 10) {
                cv::Point pt1(i, 0);
                cv::Point pt2(i, diffImage.rows);
                line(frame_with_lines, pt1, pt2, (255, 255, 255));
            }
        }

        imshow("Frame With Lines", frame_with_lines);
        imshow("Diff Image", diffImage);

        // get the input from the keyboard
        int keyboard = waitKey(30);
        if (keyboard == 'q' || keyboard == 27)
            break;
    }
    return 0;
}