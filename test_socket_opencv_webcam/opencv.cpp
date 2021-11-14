#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <netdb.h>
#include <fcntl.h>

using namespace std;
using namespace cv;

int main() {
  struct addrinfo * result, * rp;
  int sfd;
  uint8_t * buffer = new uint8_t[3 * 480 * 640];

  // struct addrinfo hints = {
  //     .ai_family = AF_INET,
  //     .ai_socktype = SOCK_DGRAM
  // };

  getaddrinfo("127.0.0.1", "2734", NULL, & result);
  for (rp = result; rp != NULL; rp = rp -> ai_next) {
    if ((sfd = socket(AF_INET, SOCK_STREAM, 0)) == -1) {
      continue;
    }
    if (connect(sfd, rp -> ai_addr, rp -> ai_addrlen) != -1) {
      // fcntl(sfd, F_SETFL, fcntl(sfd, F_GETFL, 0) | O_NONBLOCK);
      break;
    }

    close(sfd);
  }

  int iteration = 0;
  Mat frame2;
  Mat frame = imread("Untitled-2.bmp", IMREAD_ANYCOLOR);
  while (1) {
    int btot = 0;
    while (btot < 3 * 480 * 640 * sizeof * buffer) {
      btot += read(sfd, buffer + btot, 3 * 480 * 640 * sizeof * buffer);
    }

    printf("read\n");
    int buffer_id = 0;
    if(iteration == 0){
      vector < uint8_t > v(buffer, buffer + 3 * 480 * 640 * sizeof * buffer);
      frame2 = Mat(480, 640, frame.type(), & v[0]).clone();
    } else {
      for (int i = 0; i < frame2.rows; i++) {
                for (int j = 0; j < frame2.cols; j++) {
                  unsigned char * p = frame2.ptr(i, j); // Y first, X after
                  p[0] += buffer[buffer_id++];
                  p[1] += buffer[buffer_id++];
                  p[2] += buffer[buffer_id++];
                }
          }
    }
    imshow("hi", frame2);
    if (waitKey(10) == 27) break; // stop capturing by pressing ESC 
    iteration++;
  }
  freeaddrinfo(result);
  return 0;
}