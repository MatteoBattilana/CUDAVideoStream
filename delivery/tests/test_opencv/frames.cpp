#include "opencv2/opencv.hpp"

#include <time.h>

#include <math.h>

using namespace cv;

using namespace std;

int main(int argc, char ** argv) {
  VideoCapture cap;
  // open the default camera, use something different from 0 otherwise;
  // Check VideoCapture documentation.
  if (!cap.open(2))
    return 0;
  Mat previous;

  Mat frame;
  cap >> frame;
  if (frame.empty()) return 1;
  int r[frame.rows][frame.cols];
  int g[frame.rows][frame.cols];
  int b[frame.rows][frame.cols];
  for (;;)

  {

    cap >> frame;
    if (frame.empty()) break; // end of video stream
    imshow("this is you, smile! :)", frame);
    if (waitKey(10) == 27) break; // stop capturing by pressing ESC 
    if (!previous.empty()) {
	int diff_pixel_count = 0;
      // compute difference
      for (int i = 0; i < frame.rows; i++) {
        for (int j = 0; j < frame.cols; j++) {
          unsigned char * p = frame.ptr(i, j); // Y first, X after
          unsigned char * p2 = previous.ptr(i, j);
          //printf("%d\n", p[0]);

          r[i][j] = p[0] - p2[0];
          g[i][j] = p[1] - p2[1];
          b[i][j] = p[2] - p2[2];

          if (abs(r[i][j]) + abs(g[i][j]) + abs(b[i][j]) > 70) 
		diff_pixel_count++;
        }
      }
	printf("Pixel changed: %d\n", diff_pixel_count);
    }
    previous = frame.clone();
  }

  return 0;
}
