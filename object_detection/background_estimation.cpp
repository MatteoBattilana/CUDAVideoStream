#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>

using namespace std;
using namespace cv;

int computeMedian(vector<int> elements) {
    // nth_element(elements.begin(), elements.begin() + elements.size() / 2, elements.end());
    sort(elements.begin(), elements.end());

    return elements[elements.size() / 2];
}

Mat compute_median(std::vector<cv::Mat> vec) {
    // Note: Expects the image to be CV_8UC3
    Mat medianImg(vec[0].rows, vec[0].cols, CV_8UC3, cv::Scalar(0, 0, 0));

    for (int row = 0; row < vec[0].rows; row++) {
        for (int col = 0; col < vec[0].cols; col++) {
            vector<int> elements_B;
            vector<int> elements_G;
            vector<int> elements_R;

            for (int imgNumber = 0; imgNumber < vec.size(); imgNumber++) {
                int B = vec[imgNumber].at<cv::Vec3b>(row, col)[0];
                int G = vec[imgNumber].at<cv::Vec3b>(row, col)[1];
                int R = vec[imgNumber].at<cv::Vec3b>(row, col)[2];

                elements_B.push_back(B);
                elements_G.push_back(G);
                elements_R.push_back(R);
            }

            medianImg.at<cv::Vec3b>(row, col)[0] = computeMedian(elements_B);
            medianImg.at<cv::Vec3b>(row, col)[1] = computeMedian(elements_G);
            medianImg.at<cv::Vec3b>(row, col)[2] = computeMedian(elements_R);
        }
    }
    return medianImg;
}

int main(int argc, char const *argv[]) {
    string video_file;
    // Read video file
    // uchar data[12] = {255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0};
    // uchar data[12] = {0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0};
    uchar data[12] = {0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255};
    Mat src = Mat(2, 2, CV_8UC3, data);
    for (int i = 0; i < 11; i++)
    {
        printf("%d\n", src.data[i]);
    }
    
    imshow("prova", src);
    waitKey(0);
    if (argc > 1) {
        video_file = argv[1];
    } else {
        video_file = "video.mp4";
    }

    VideoCapture cap(video_file);
    // VideoCapture cap(0, CAP_V4L2);
    double prova = cap.get(CAP_PROP_FRAME_COUNT);

    if (!cap.isOpened())
        cerr << "Error opening video file\n";

    // Randomly select 25 frames
    default_random_engine generator;
    uniform_int_distribution<int> distribution(0, cap.get(CAP_PROP_FRAME_COUNT));

    vector<Mat> frames;
    Mat frame;

    for (int i = 0; i < 25; i++) {
        int fid = distribution(generator);
        cap.set(CAP_PROP_POS_FRAMES, fid);
        Mat frame;
        cap >> frame;
        if (frame.empty())
            continue;
        frames.push_back(frame);
    }

    // Calculate the median along the time axis
    Mat medianFrame = compute_median(frames);

    // Display median frame
    imshow("frame", medianFrame);
    waitKey(0);

    //  Reset frame number to 0
    cap.set(CAP_PROP_POS_FRAMES, 0);

    // Convert background to grayscale
    Mat grayMedianFrame, original_frame;
    cvtColor(medianFrame, grayMedianFrame, COLOR_BGR2GRAY);

    // Loop over all frames
    while (1) {
        // Read frame
        cap >> frame;
        original_frame = frame.clone();
        if (frame.empty())
            break;

        // Convert current frame to grayscale
        cvtColor(frame, frame, COLOR_BGR2GRAY);

        // Calculate absolute difference of current frame and the median frame
        Mat dframe;
        absdiff(frame, grayMedianFrame, dframe);

        // Threshold to binarize
        threshold(dframe, dframe, 30, 255, THRESH_BINARY);

        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(dframe, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);
        // draw contours on the original image
        Mat image_copy = original_frame.clone();
        // drawContours(image_copy, contours, -1, Scalar(0, 255, 0), 2);

        for (size_t i = 0; i < contours.size(); i++) {
            int area = contourArea(contours[i]);
            if (area < 10) {
                drawContours(image_copy, contours, i, Scalar(0, 255, 0), 2);
            }
        }

        imshow("None approximation", image_copy);
        // Display Image
        imshow("frame", dframe);

        waitKey(20);
    }

    cap.release();
    return 0;
}