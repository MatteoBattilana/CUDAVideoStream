#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

typedef long4 chunk_t;

#define H 1080
#define W 1920

__global__ void grayscale(uint8_t *color, uint8_t *grayscale, int maxSect) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int start = x * maxSect;
    int max = start + maxSect;
    int sum = 0;

    for (int i = start; i < max; i++) {
        sum = color[i*3] + color[i*3 + 1] + color[i*3 + 2];
        grayscale[i] = sum/3;
    }
}

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
    unsigned int *d_pos;
    output.create(H, W, CV_8UC1);
    // output.create(H, W, CV_8UC3);
    int sum = 0;

    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int total_color = 3 * W * H;
    int total_grayscale = W * H;
    int nMaxThreads = prop.maxThreadsPerBlock;
    int maxAtTime = total_grayscale / nMaxThreads;
    uint8_t *d_color, *d_grayscale;
    cudaMalloc((void **)&d_color, total_color * sizeof *d_color);
    cudaMalloc((void **)&d_grayscale, total_grayscale * sizeof *d_grayscale);
    cudaMalloc((void **)&d_pos, sizeof *d_pos);

    while (1) {
        cap >> frame;
        if (frame.empty())
            return 0;

        imshow("input", frame);

        if (waitKey(10) == 27) {
            break; // stop capturing by pressing ESC
        }

        auto start = std::chrono::high_resolution_clock::now();

        cudaMemset(d_pos, 0, sizeof *d_pos);
        cudaMemcpy(d_color, frame.data, total_color, cudaMemcpyHostToDevice);

        grayscale<<<1, nMaxThreads>>>(d_color, d_grayscale, maxAtTime);

        cudaMemcpy(output.data, d_grayscale, total_grayscale, cudaMemcpyDeviceToHost);

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