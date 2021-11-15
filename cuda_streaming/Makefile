CC = /usr/local/cuda/bin/nvcc
LFLAGS += -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio -L/usr/local/lib
CFLAGS += -std=c++11 -g 
INCLUDES += -I/usr/include/opencv4

all: build run

build: 
	$(CC) $(CFLAGS) $(LFLAGS) $(INCLUDES) server.cu -o server

run:
	./server