CC =g++
LFAGS += -g -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs
INCLUDES += -I/usr/include/opencv4

all: m1 m2

m1: 
	g++ -std=c++11 image_reader.cpp -ograb -lv4l2 -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -L/usr/local/lib -lopencv_shape -lopencv_videoio -o image_reader
m2: 
	$(CC) opencv.cpp $(LFAGS) $(INCLUDES) -o opencv

