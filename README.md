# CUDA-Video-Stream

## Webcam test
First: `test_v4l2` folder

### Compilation
```
g++ test.cpp -lv4l2
```
Moreover you have to install v4l2 
```
sudo apt-get install libv4l-dev v4l-utils
```


You have to modify the device and its resolution by looking at the output of the following command:
`v4l2-ctl -d /dev/video0 --list-formats-ext`
You have to choose the YUY format. An image will be store into the `output_raw_img` file and can be visualized in the [http://rawpixels.net/](http://rawpixels.net/) site.
Use this parameter:
![image](https://user-images.githubusercontent.com/9128612/141523554-c8488fd3-daef-4083-b54d-d9b5dd531c02.png)

