CC = /usr/local/cuda/bin/nvcc
LFLAGS += -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio -lpthread -L/usr/local/lib
LFLAGS += -lv4l2
CFLAGS += -std=c++11 -g 
INCLUDES += -I/usr/include/opencv4
# INCLUDES += 
# SRCS += server.cu v4l.cpp
SRCS += server.cu

all: build run

build: 
	$(CC) $(CFLAGS) $(LFLAGS) $(INCLUDES) $(SRCS) -o server

run:
	./server

test:
	$(CC) $(CFLAGS) $(LFLAGS) $(INCLUDES) test.cu -o server

test2:
	$(CC) $(CFLAGS) $(LFLAGS) $(INCLUDES) test2.cu -o server