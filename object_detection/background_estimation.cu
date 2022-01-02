#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>
#include <string>

using namespace std;
using namespace cv;

#define LR_THRESHOLDS 20

typedef long4 chunk_t;

__global__ void background(uint8_t *current, uint8_t *previous, int maxSect, uint8_t *noise_visualization) {

    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int start = x * maxSect;
    int max = start + maxSect;
    chunk_t cc, pc;
    uint8_t redColor = 0;
    int df;
    int size = sizeof(chunk_t);

    for (int i = start; i < max; i++) {
        cc = ((chunk_t *)current)[i];
        pc = ((chunk_t *)previous)[i];

        for (int j = 0; j < size; j++) {
            df = ((uint8_t *)&cc)[j] - ((uint8_t *)&pc)[j];

            if ((df < -LR_THRESHOLDS || df > LR_THRESHOLDS)) {
                redColor = 255;
            }

            if (((i * size) + j) % 3 == 2) {
                noise_visualization[(i * size) + j] = redColor;
                redColor = 0;
            }
        }
    }
}

int main(int argc, char const *argv[]) {
    string video_file;
    video_file = "video.mp4";

    VideoCapture cap(video_file);
    // VideoCapture cap(0, CAP_V4L2);
    double prova = cap.get(CAP_PROP_FRAME_COUNT);

    if (!cap.isOpened())
        cerr << "Error opening video file\n";

    // Randomly select 25 frames
    default_random_engine generator;
    uniform_int_distribution<int> distribution(0, cap.get(CAP_PROP_FRAME_COUNT));

    vector<Mat> frames;
    Mat frame, median_frame;

    for (int i = 0; i < 25; i++) {
        int fid = distribution(generator);
        cap.set(CAP_PROP_POS_FRAMES, fid);
        Mat frame;
        cap >> frame;
        if (frame.empty())
            continue;
        frames.push_back(frame);
    }

    uint8_t *frames_unrolled;
    imshow("frames[0]", frames[0]);
    printf("%d\n", frames.size());
    waitKey(0);
    int k = 0;
    int trash;
    frames_unrolled = frames.data()->data;
    ofstream out("out.txt");
    streambuf *coutbuf = std::cout.rdbuf(); // save old buf
    cout.rdbuf(out.rdbuf());                // redirect std::cout to out.txt!
    cout << "M = " << endl
         << " " << frames[0] << endl
         << endl;
    cout << endl;
    cout << "M = " << endl
         << " " << frames[1] << endl
         << endl;
    scanf("%d", &trash);

    for (int i = 0; i < 10; i++) {
        printf("%d\n", frames_unrolled[i]);
        scanf("%d", &trash);
    }
    for (int i = 691190; i < 693000; i++) {
        printf("%d\n", frames_unrolled[i]);
        scanf("%d", &trash);
    }

    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int total_bytes = 25 * 3 * frames[0].rows * frames[0].cols;
    int total = 3 * frames[0].rows * frames[0].cols;
    int nMaxThreads = prop.maxThreadsPerBlock;
    int maxAtTime = total / nMaxThreads;

    // Loop over all frames

    cap.release();
    return 0;
}