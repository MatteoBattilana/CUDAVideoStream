# CUDA-Video-Stream

## Webcam test
First: `test_v4l2` folder
You have to modify the device and its resolution by looking at the output of the following command:
`v4l2-ctl -d /dev/video0 --list-formats-ext`
You have to choose the YUY format. An image will be store into the `output_raw_img` file and can be visualized in the [http://rawpixels.net/](http://rawpixels.net/) site.
