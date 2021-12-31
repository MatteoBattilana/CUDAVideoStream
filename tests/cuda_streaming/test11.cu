
#include <stdio.h>
#include <fcntl.h>    /* For O_RDWR */
#include <sys/ioctl.h>
#include <unistd.h>   /* For open(), creat() */
#include <netdb.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <unistd.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <stdint.h>
#include <pthread.h>
#include <math.h>
#include <stdlib.h>

__global__ void kernel2(uint8_t *current, uint8_t *previous, int maxSect, uint8_t* d_heat_pixels) {
    int df;
    int cc, pc;
    for (int i = 0; i < 10; i++) {
        cc = ((int *)current)[i];
        pc = ((int *)previous)[i];

        int pixelDiff = 0;
        for (int j = 0; j < 4; j++) {
            df = ((uint8_t *)&cc)[j] - ((uint8_t *)&pc)[j];
            pixelDiff += fabs(df);

            int pos = i*4+j;
            if( pos%3==2){
                float diff1 = pixelDiff/(255*3.0);
                int r = 255 - 1;//fmin(fmax(sin(M_PI*diff1 - M_PI/2.0)*255.0, 0.0),255.0);
                int g = 255 - 2; //fmin(fmax(sin(M_PI*diff1)*255.0, 0.0),255.0);
                int b = 255 - 3; //fmin(fmax(sin(M_PI*diff1 + M_PI/2.0)*255.0, 0.0),255.0);
                int  v = (int)(d_heat_pixels[pos] | r << 16 | g << 8 | b );
                printf("%d %d \n", pos , d_heat_pixels[pos]);
                *((int*)(d_heat_pixels+pos-2)) = v;
                // d_heat_pixels[pos-2] = b;
                // d_heat_pixels[pos-1] = g;
                // d_heat_pixels[pos] = r;
            }
        }
    }
}

int main() {

    uint8_t *d_current, *d_previous;
    uint8_t *d_diff;
    uint8_t *d_heat_pixels;

    uint8_t a[9] = {10, 11, 2, 1, 0, 11, 22, 14, 44};
    uint8_t b[9] = {22, 44, 55, 1, 1, 3, 23, 11, 22};
    uint8_t d[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

    cudaMalloc((void **)&d_current, 9 * sizeof *d_current);
    cudaMalloc((void **)&d_previous, 9 * sizeof *d_previous);
    cudaMalloc((void **)&d_heat_pixels, 9 * sizeof *d_heat_pixels);
    kernel2(a, b, 3, d);
    printf("\n");
    printf("\n");
    for(int i = 0; i < 10; i ++){
        printf("%d ", d[i]);
    }
    printf("\n");
    return 0;
}