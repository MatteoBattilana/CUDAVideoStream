# CUDA-Video-Stream

## Webcam test
### `test_v4l2`
You have to modify the device and its resolution by looking at the output of the following command:
`v4l2-ctl -d /dev/video0 --list-formats-ext`
You have to choose the YUY format. An image will be store into the `output_raw_img` file and can be visualized in the [http://rawpixels.net/](http://rawpixels.net/) site.

### `test_opencv`
Change the video source by chaning the number 2 with your webcam id /dev/videoX.
To compile:
```
g++ -std=c++11 frames.cpp -ograb -lv4l2 -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -L/usr/local/lib -lopencv_shape -lopencv_videoio -o frames.out
```
