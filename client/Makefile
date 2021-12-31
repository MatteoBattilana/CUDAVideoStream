CC =g++
LFAGS += -g -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio
INCLUDES += -I/usr/include/opencv4

all: client run

client: 
	$(CC) opencv.cpp $(LFAGS) $(INCLUDES) -o opencv

run:
	./opencv
