#ifndef COMM_H_
#define COMM_H_

// 1 for heat map, 2 for red-black, 0 nothing
// #define NOISE_FILTER
#define K 3
#define TILE_SIZE 10
#define BLOCK_SIZE (TILE_SIZE + K - 1)

// Noise visualizer: 1 heatmap, 2 red-black, 3 red-black overlap, 4 grayscale-kernel
#define NOISE_VISUALIZER 5

#define CHARS_STR "0123456789BFPSWbkps :"
#define LR_THRESHOLDS 20
#define GPU
#define KERNEL2_NEGFEED_OPT

#define SERVER_IMSHOW

#endif