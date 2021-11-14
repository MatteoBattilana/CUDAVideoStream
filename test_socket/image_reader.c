#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdint.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <netdb.h>
#include <sys/epoll.h>

struct rgb_s {
    uint8_t red;
    uint8_t green;
    uint8_t blue;
};

struct bmp_s {
    struct header_s {
        uint8_t id[2];
        uint32_t size;
        uint16_t _reserved0;
        uint16_t _reserved1;
        uint32_t offset;
    } header;

    struct dip_s {
        uint16_t width;
        uint16_t heigth;
    } dip;

    struct rgb_s *pixels;
};

typedef union {
    uint8_t byte;
    struct {
        unsigned red: 1;
        unsigned green: 1;
        unsigned blue: 1;
        unsigned _reserved: 5;
    } channels;
} diffsign_t;

void read_bmp(struct bmp_s *bmp, const char *bmpfile) {
    int fd, i;
    struct stat sb;
    uint8_t *file;
    uint8_t *bmap;

    lstat(bmpfile, &sb);
    file = malloc(sb.st_size * sizeof *file);
    fd = open(bmpfile, O_RDONLY);

    if (read(fd, file, sb.st_size * sizeof *file) != sb.st_size) {
        fprintf(stderr, "WTF!\n");
    }

    memcpy(bmp->header.id, file + 0x00, 2 * sizeof *bmp->header.id);
    memcpy(&bmp->header.size, file + 0x02, sizeof bmp->header.size);
    memcpy(&bmp->header.offset, file + 0x0a, sizeof bmp->header.offset);
    memcpy(&bmp->dip.width, file + 0x12, sizeof bmp->dip.width);
    memcpy(&bmp->dip.heigth, file + 0x16, sizeof bmp->dip.heigth);
    
    bmap = malloc(bmp->header.size * sizeof *bmap);
    memcpy(bmap, file + bmp->header.offset, bmp->header.size * sizeof *bmap);

    bmp->pixels = malloc((bmp->header.size / 3) * sizeof *bmp->pixels);
    for (i = 0; i < bmp->dip.heigth * bmp->dip.width; i++) {
        bmp->pixels[i].red = bmap[3 * i + 2];
        bmp->pixels[i].green = bmap[3 * i + 1];
        bmp->pixels[i].blue = bmap[3 * i + 0];
    }

    free(bmap);
    free(file);
    close(fd);
}

int main(int argc, char **argv) {
    struct epoll_event ev, events[10];
    struct addrinfo *result, *rp;
    struct bmp_s bmp1, bmp2;
    int i, j, max;
    struct rgb_s *diff;
    diffsign_t *diffsign;
    int sfd, epollfd, nfds, sfd2;

    if (argc < 3) {
        fprintf(stderr, "ERROR: no image specified!\n");
    }

    getaddrinfo("127.0.0.1", "2734", NULL, &result);
    for (rp = result; rp != NULL; rp = rp->ai_next) {
        if ((sfd = socket(AF_INET, SOCK_STREAM, 0)) == -1) {
            continue;
        }

        if (bind(sfd, rp->ai_addr, rp->ai_addrlen) != -1) {
            break;
        }

        close(sfd);
        perror("MOH!");
    }

    epollfd = epoll_create1(0);
    ev.events = EPOLLIN;
    ev.data.fd = sfd;
    epoll_ctl(epollfd, 1, sfd, &ev);

    read_bmp(&bmp1, argv[1]);
    read_bmp(&bmp2, argv[2]);

    if (listen(sfd, 10) < 0) {
        perror("OH!");
    }

    while (1) {
        printf("wait..\n");
        nfds = epoll_wait(epollfd, events, 10, -1);
        printf("yes!\n");

        for (i = 0; i < nfds; i++) {
            if (events[i].data.fd == sfd) {
                sfd2 = accept(sfd, NULL, NULL);
                perror("AH!");

                ev.data.fd = sfd2;
                epoll_ctl(epollfd, 1, sfd2, &ev);
                printf("stored!\n");

                max = 0;
                while(1) {

                    for (j = bmp1.dip.heigth * bmp1.dip.width; j >= 0; j--) {
                        write(sfd2, &bmp1.pixels[j].blue, sizeof bmp1.pixels[j].blue);
                        write(sfd2, &bmp1.pixels[j].green, sizeof bmp1.pixels[j].green);
                        write(sfd2, &bmp1.pixels[j].red, sizeof bmp1.pixels[j].red);
                        max += 3;
                    }

                    printf("so.. %d\n", max);

                    // sleep(10);
                    break;
                }
            } else {
                close(events[i].data.fd);
            }
        }


    }

    freeaddrinfo(result);

    // diff = malloc(bmp1.dip.heigth * bmp1.dip.width * sizeof *diff);
    // diffsign = malloc(bmp1.dip.heigth * bmp1.dip.width * sizeof *diffsign);



    // for (i = 0, max = bmp1.dip.heigth * bmp1.dip.width; i < max; i++) {
    //     diff[i].red = abs(bmp2.pixels[i].red - bmp1.pixels[i].red);
    //     diff[i].green = abs(bmp2.pixels[i].green - bmp1.pixels[i].green);
    //     diff[i].blue = abs(bmp2.pixels[i].blue - bmp1.pixels[i].blue);

    //     diffsign[i].channels.red = bmp2.pixels[i].red < bmp1.pixels[i].red;
    //     diffsign[i].channels.green = bmp2.pixels[i].green < bmp1.pixels[i].green;
    //     diffsign[i].channels.blue = bmp2.pixels[i].blue < bmp1.pixels[i].blue;

    //     // if (diff[i].red || diff[i].green || diff[i].blue) {
    //     //     printf("\tbmp1 [%04d][%04d] r: %03d g: %03d b: %03d\n", i / bmp1.dip.width, j % bmp1.dip.width, bmp1.pixels[i].red, bmp1.pixels[i].green, bmp1.pixels[i].blue);
    //     //     printf("\tbmp2 [%04d][%04d] r: %03d g: %03d b: %03d\n", i / bmp1.dip.width, j % bmp1.dip.width, bmp2.pixels[i].red, bmp2.pixels[i].green, bmp2.pixels[i].blue);
    //     //     printf("\tdiff [%04d][%04d] r: %03d g: %03d b: %03d\n", i / bmp1.dip.width, j % bmp1.dip.width, diff[i].red, diff[i].green, diff[i].blue);
    //     //     printf("\tsign [%04d][%04d] r: %03d g: %03d b: %03d\n", i / bmp1.dip.width, j % bmp1.dip.width, diffsign[i].channels.red, diffsign[i].channels.green, diffsign[i].channels.blue);
    //     //     printf("\n");
    //     // }
    // }

    // for (i = 0; i < bmp1.dip.heigth; i++) {
    //     for (j = 0; j < bmp1.dip.width; j++) {
    //         diff[i * bmp1.dip.width + j].red = abs(bmp2.pixels[i * bmp1.dip.width + j].red - bmp1.pixels[i * bmp1.dip.width + j].red);
    //         diff[i * bmp1.dip.width + j].green = abs(bmp2.pixels[i * bmp1.dip.width + j].green - bmp1.pixels[i * bmp1.dip.width + j].green);
    //         diff[i * bmp1.dip.width + j].blue = abs(bmp2.pixels[i * bmp1.dip.width + j].blue - bmp1.pixels[i * bmp1.dip.width + j].blue);

    //         diffsign[i * bmp1.dip.width + j].channels.red = bmp2.pixels[i * bmp1.dip.width + j].red < bmp1.pixels[i * bmp1.dip.width + j].red;
    //         diffsign[i * bmp1.dip.width + j].channels.green = bmp2.pixels[i * bmp1.dip.width + j].green < bmp1.pixels[i * bmp1.dip.width + j].green;
    //         diffsign[i * bmp1.dip.width + j].channels.blue = bmp2.pixels[i * bmp1.dip.width + j].blue < bmp1.pixels[i * bmp1.dip.width + j].blue;

    //         if (diff[i * bmp1.dip.width + j].red || diff[i * bmp1.dip.width + j].green || diff[i * bmp1.dip.width + j].blue) {
    //             printf("\tbmp1 [%04d][%04d] r: %03d g: %03d b: %03d\n", i, j, bmp1.pixels[i * bmp1.dip.width + j].red, bmp1.pixels[i * bmp1.dip.width + j].green, bmp1.pixels[i * bmp1.dip.width + j].blue);
    //             printf("\tbmp2 [%04d][%04d] r: %03d g: %03d b: %03d\n", i, j, bmp2.pixels[i * bmp1.dip.width + j].red, bmp2.pixels[i * bmp1.dip.width + j].green, bmp2.pixels[i * bmp1.dip.width + j].blue);
    //             printf("\tdiff [%04d][%04d] r: %03d g: %03d b: %03d\n", i, j, diff[i * bmp1.dip.width + j].red, diff[i * bmp1.dip.width + j].green, diff[i * bmp1.dip.width + j].blue);
    //             printf("\tsign [%04d][%04d] r: %03d g: %03d b: %03d\n", i, j, diffsign[i * bmp1.dip.width + j].channels.red, diffsign[i * bmp1.dip.width + j].channels.green, diffsign[i * bmp1.dip.width + j].channels.blue);
    //             printf("\n");
    //         }

    //     }
    // }


    // free(diff);
    // free(diffsign);
    // free(bmp1.pixels);
    // free(bmp2.pixels);
    return 0;

}
