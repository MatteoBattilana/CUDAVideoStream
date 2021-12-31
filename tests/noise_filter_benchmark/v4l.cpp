#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <fcntl.h>    /* For O_RDWR */
#include <sys/ioctl.h>
#include <unistd.h>   /* For open(), creat() */
#include <sys/mman.h>

#include <linux/videodev2.h>
#include "v4l.h"

int set_format(int fd) {
    struct v4l2_format format = {0};
    format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    format.fmt.pix.width = 1920;
    format.fmt.pix.height = 1080;
    format.fmt.pix.pixelformat = V4L2_PIX_FMT_BGR24;  // V4L2_PIX_FMT_YUYV , V4L2_PIX_FMT_SRGGB10
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

u_int8_t* query_buffer(int fd, int *size) {
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
    *size = buf.length;
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
    // int file = open("output_raw_img", O_WRONLY | O_CREAT, 0644);
    printf("grab_frame fd : %d\n", camera);
    // write(file, buffer, size); //size is obtained from the query_buffer function
    // dequeue_buffer(camera);
}

static int xioctl(int fd, int request, void *arg) {
        int r;

        do r = ioctl (fd, request, arg);
        while (-1 == r && EINTR == errno);

        return r;
}

int print_caps(int fd) {
        struct v4l2_capability caps = {};
        if (-1 == xioctl(fd, VIDIOC_QUERYCAP, &caps))
        {
                perror("Querying Capabilities");
                return 1;
        }

        printf( "Driver Caps:\n"
                "  Driver: \"%s\"\n"
                "  Card: \"%s\"\n"
                "  Bus: \"%s\"\n"
                "  Version: %d.%d\n"
                "  Capabilities: %08x\n",
                caps.driver,
                caps.card,
                caps.bus_info,
                (caps.version>>16)&&0xff,
                (caps.version>>24)&&0xff,
                caps.capabilities);


        struct v4l2_cropcap cropcap = {0};
        cropcap.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (-1 == xioctl (fd, VIDIOC_CROPCAP, &cropcap))
        {
                perror("Querying Cropping Capabilities");
                return 1;
        }

        printf( "Camera Cropping:\n"
                "  Bounds: %dx%d+%d+%d\n"
                "  Default: %dx%d+%d+%d\n"
                "  Aspect: %d/%d\n",
                cropcap.bounds.width, cropcap.bounds.height, cropcap.bounds.left, cropcap.bounds.top,
                cropcap.defrect.width, cropcap.defrect.height, cropcap.defrect.left, cropcap.defrect.top,
                cropcap.pixelaspect.numerator, cropcap.pixelaspect.denominator);

        int support_grbg10 = 0;

        struct v4l2_fmtdesc fmtdesc = {0};
        fmtdesc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        char fourcc[5] = {0};
        char c, e;
        printf("  FMT : CE Desc\n--------------------\n");
        while (0 == xioctl(fd, VIDIOC_ENUM_FMT, &fmtdesc))
        {
                strncpy(fourcc, (char *)&fmtdesc.pixelformat, 4);
                if (fmtdesc.pixelformat == V4L2_PIX_FMT_YUYV)
                    support_grbg10 = 1;
                c = fmtdesc.flags & 1? 'C' : ' ';
                e = fmtdesc.flags & 2? 'E' : ' ';
                printf("  %s: %c%c %s\n", fourcc, c, e, fmtdesc.description);
                fmtdesc.index++;
        }

        if (!support_grbg10)
        {
            printf("Doesn't support GRBG10.\n");
        }

        struct v4l2_format fmt = {0};
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.fmt.pix.width = 1920;
        fmt.fmt.pix.height = 1080;
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
        fmt.fmt.pix.field = V4L2_FIELD_NONE;

        if (-1 == xioctl(fd, VIDIOC_S_FMT, &fmt))
        {
            perror("Setting Pixel Format");
            return 1;
        }

        strncpy(fourcc, (char *)&fmt.fmt.pix.pixelformat, 4);
        printf( "Selected Camera Mode:\n"
                "  Width: %d\n"
                "  Height: %d\n"
                "  PixFmt: %s\n"
                "  Field: %d\n",
                fmt.fmt.pix.width,
                fmt.fmt.pix.height,
                fourcc,
                fmt.fmt.pix.field);
        return 0;
}

int init_mmap(int fd, uint8_t **buffer, int *sz) {

    print_caps(fd);

    struct v4l2_requestbuffers req = {0};
    req.count = 1;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;

    if (-1 == xioctl(fd, VIDIOC_REQBUFS, &req)) {
        perror("Requesting Buffer");
        return 1;
    }

    struct v4l2_buffer buf = {0};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = 0;
    if(-1 == xioctl(fd, VIDIOC_QUERYBUF, &buf)) {
        perror("Querying Buffer");
        return 1;
    }

    *buffer = (uint8_t *)mmap (NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset);
    *sz = buf.bytesused;
    printf("Length: %d\nAddress: %p\n", buf.length, *buffer);
    printf("Image Length: %d\n", buf.bytesused);

    return 0;
}

int capture_image(int fd, uint8_t *buffer, uint8_t *frame) {
    struct v4l2_buffer buf = {0};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = 0;
    if(-1 == xioctl(fd, VIDIOC_QBUF, &buf)) {
        perror("Query Buffer");
        return 1;
    }

    if(-1 == xioctl(fd, VIDIOC_STREAMON, &buf.type)) {
        perror("Start Capture");
        return 1;
    }

    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(fd, &fds);
    struct timeval tv = {0};
    tv.tv_sec = 2;
    int r = select(fd+1, &fds, NULL, NULL, &tv);
    if(-1 == r)
    {
        perror("Waiting for Frame");
        return 1;
    }

    if(-1 == xioctl(fd, VIDIOC_DQBUF, &buf))
    {
        perror("Retrieving Frame");
        return 1;
    }

    // int outfd = open("out.img", O_RDWR);
    memcpy(frame, buffer, buf.bytesused);;
    // write(outfd, buffer, buf.bytesused);
    // close(outfd);

    return 0;
}


// int main() {    
//     int fd = open("/dev/video2", O_RDWR);
//     printf("fd : %d\n\n", fd);

//     set_format(fd);
//     int count = 1;
//     request_buffer(fd, count);

//     u_int8_t* buffer = query_buffer(fd);

//     start_streaming(fd);
//     grab_frame(fd, size, buffer);
//     return 0;
// }
