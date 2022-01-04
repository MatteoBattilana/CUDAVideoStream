#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define NAIVE

#define H 1080
#define W 1920

__global__ void compute_max(int *histogram, uint8_t *indexes_max) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    __shared__ int shared_histogram[256];
    __shared__ int shared_indexes[256];

    shared_histogram[tid] = histogram[tid];
    shared_indexes[tid] = tid;
    __syncthreads();

    for (unsigned int offset = blockDim.x / 2; offset > 1; offset >>= 1) {
        // Exercise 1.1) reduce two values per loop and write these back to shared memory
        if (threadIdx.x < offset) {
            if (shared_histogram[threadIdx.x] < shared_histogram[threadIdx.x + offset]) {
                shared_histogram[threadIdx.x] = shared_histogram[threadIdx.x + offset];
                shared_indexes[threadIdx.x] = shared_indexes[threadIdx.x + offset];
            }
        }
        // sync threads required to ensure all threads have finished writing
        __syncthreads();
    }
    if(tid == 0){
        indexes_max[tid] = shared_indexes[tid];
    }
    if(tid == 1){
        indexes_max[tid] = shared_indexes[tid];
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

    Mat frame, bw, binarize;
    frame.create(H, W, CV_8UC3);
    bw.create(H, W, CV_8UC1);
    binarize.create(H, W, CV_8UC1);
    int sum = 0;

    // GPU

    int total_grayscale = W * H;
    int *d_histogram;
    uint8_t *d_indexes;
    uint8_t *d_grayscale;

    cudaMalloc((void **)&d_histogram, 256 * sizeof(int));
    cudaMalloc((void **)&d_indexes, 2 * sizeof(uint8_t));

    //

    while (1) {
        cap >> frame;
        if (frame.empty())
            return 0;

        imshow("input", frame);

        if (waitKey(10) == 27) {
            break; // stop capturing by pressing ESC
        }

        // generate black & white image

        for (int row = 0; row < H; row++) {
            for (int col = 0; col < W; col++) {
                bw.at<uchar>(row, col) = 0.114 * frame.at<Vec3b>(row, col)[0] + 0.587 * frame.at<Vec3b>(row, col)[1] + 0.299 * frame.at<Vec3b>(row, col)[2];
            }
        }

        imshow("bw", bw);
        // waitKey(0);
        if (waitKey(10) == 27) {
            break; // stop capturing by pressing ESC
        }

        // find the histogram of the occurency of the values from 0 to 255
        // Naive implementation is a for loop from 0 to 255 and then a loop inside on the matrix
        int histogram[256] = {0};
        uint8_t h_indexes[2] = {0};

#ifdef NAIVE

        // CPU
        auto start = std::chrono::high_resolution_clock::now();
        for (int row = 0; row < H; row++) {
            for (int col = 0; col < W; col++) {
                int index = bw.at<uchar>(row, col);
                histogram[index]++;
            }
        }
        //

        // for (int i = 0; i < 256; i++) {
        //     printf("%d \n", histogram[i]);
        // }
        // int trash;
        // scanf("%d", &trash);

        // CPU

        // int max = -1, sec_max = -1;
        int index_max = -1, index_sec_max = -1;
        // for (int i = 0; i < 256; i++) {
        //     if (histogram[i] >= max) {
        //         index_sec_max = index_max;
        //         index_max = i;
        //         max = histogram[i];
        //         sec_max = max;
        //     } else if (histogram[i] > sec_max && histogram[i] < max) {
        //         sec_max = histogram[i];
        //         index_sec_max = i;
        //     }
        // }

        //

        // GPU
        cudaMemcpy(d_histogram, histogram, 256 * sizeof(int), cudaMemcpyHostToDevice);

        compute_max<<<1, 256>>>(d_histogram, d_indexes);

        cudaMemcpy(h_indexes, d_indexes, 2 * sizeof(uint8_t), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        //
        for (int i = 0; i < 2; i++) {
            printf("%d \n", h_indexes[i]);
        }
        index_max = h_indexes[0];
        index_sec_max = h_indexes[1];
        printf("\n%d %d\n", index_max, index_sec_max);
        int threshold = (index_max + index_sec_max) / 2;

        if (threshold < 50) {
            threshold = 50;
        }
        if(threshold > 200){
            threshold = 200;
        }

        // scanf("%d", &trash);

        for (int row = 0; row < H; row++) {
            for (int col = 0; col < W; col++) {
                if (bw.at<uchar>(row, col) > threshold) {
                    binarize.at<uchar>(row, col) = 255;
                } else {
                    binarize.at<uchar>(row, col) = 0;
                }
            }
        }

        imshow("binarize", binarize);

        auto end = std::chrono::high_resolution_clock::now();
        auto elaps = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        printf("\rHeatmap time generation: %.3f ms", (float)elaps.count() * 1e-6);
        fflush(stdout);
#endif
    }

    cap.release();
    return 0;
}