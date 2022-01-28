#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>    /* For O_RDWR */
#include <sys/ioctl.h>
#include <unistd.h>   /* For open(), creat() */
#include <sys/mman.h>

#include <linux/videodev2.h>

int set_format(int fd) {
    struct v4l2_format format = {0};
    format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    format.fmt.pix.width = 640;
    format.fmt.pix.height = 480;
    format.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;  // V4L2_PIX_FMT_YUYV , V4L2_PIX_FMT_SRGGB10
    format.fmt.pix.field = V4L2_FIELD_NONE;
    int res = ioctl(fd, VIDIOC_S_FMT, &format);
    if(res == -1) {
        perror("Could not set format");
        exit(1);
    }
    else
        printf("set_format: ioctl returns : %d\n", res);
    return res;
}

int request_buffer(int fd, int count) {
    struct v4l2_requestbuffers req = {0};
    req.count = count;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    int res = ioctl(fd, VIDIOC_REQBUFS, &req);
    if (res == -1)
    {
        perror("Requesting Buffer");
        exit(1);
    }
    else
        printf("request_buffer: ioctl returns : %d\n", res);
    return 0;
}

int size;
u_int8_t* query_buffer(int fd) {
    struct v4l2_buffer buf = {0};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = 0;
    int res = ioctl(fd, VIDIOC_QUERYBUF, &buf);
    if(res == -1) {
        perror("Could not query buffer");
        exit(2);
    }
    else
        printf("query_buffer: ioctl returns : %d\n", res);
    
    u_int8_t* buffer = (u_int8_t*)mmap (NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset);
    size = buf.length;
    // return buf.length;
    return buffer;
}

int start_streaming(int fd) {
    unsigned int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    int res = ioctl(fd, VIDIOC_STREAMON, &type);
    if(res == -1){
        perror("VIDIOC_STREAMON");
        exit(1);
    }
    else
        printf("start_streaming: ioctl returns : %d\n", res);

    return res;
}

int queue_buffer(int fd) {
    struct v4l2_buffer bufd = {0};
    bufd.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    bufd.memory = V4L2_MEMORY_MMAP;
    bufd.index = 0;
    int res = ioctl(fd, VIDIOC_QBUF, &bufd);
    if(res == -1)
    {
        perror("Queue Buffer");
        return 1;
    }
    else
        printf("queue_buffer: ioctl returns : %d\n", res);
    return bufd.bytesused;
}

void grab_frame(int camera, int size, u_int8_t* buffer) {
    queue_buffer(camera);
    //Wait for io operation
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(camera, &fds);
    struct timeval tv = {0};
    tv.tv_sec = 2; //set timeout to 2 second
    int r = select(camera+1, &fds, NULL, NULL, &tv);
    if(-1 == r){
        perror("Waiting for Frame");
        exit(1);
    }
    int file = open("output_raw_img", O_WRONLY | O_CREAT, 0644);
    printf("grab_frame fd : %d\n", file);
    write(file, buffer, size); //size is obtained from the query_buffer function
    // dequeue_buffer(camera);
}

int main() {    
    int fd = open("/dev/video2", O_RDWR);
    printf("fd : %d\n\n", fd);

    set_format(fd);
    int count = 1;
    request_buffer(fd, count);

    u_int8_t* buffer = query_buffer(fd);

    start_streaming(fd);
    grab_frame(fd, size, buffer);
    return 0;
}
