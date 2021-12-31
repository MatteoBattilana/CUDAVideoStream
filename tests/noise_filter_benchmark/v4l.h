#ifndef V4L_H_
#define V4L_H_

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int set_format(int fd);
int request_buffer(int fd, int count);
u_int8_t* query_buffer(int fd, int *size);
int start_streaming(int fd);
void grab_frame(int camera, int size, u_int8_t* buffer);

int init_mmap(int fd, uint8_t **buffer, int *sz);
int capture_image(int fd, uint8_t *buffer, uint8_t *frame);


#endif