#ifndef COMM_H_
#define COMM_H_

// 1 for heat map, 2 for red-black, 0 nothing
#define NOISE_FILTER
#define K 3
#define TILE_SIZE 10
#define BLOCK_SIZE (TILE_SIZE + K - 1)

// Noise visualizer: 1 heatmap, 2 red-black, 3 red-black overlap
#define NOISE_VISUALIZER 3

#define CHARS_STR "0123456789BFPSWbkps :"
#define LR_THRESHOLDS 20
#define GPU
#define KERNEL2_NEGFEED_OPT

#endif