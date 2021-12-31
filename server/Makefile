CC = /usr/local/cuda/bin/nvcc
PROFILER = /usr/local/cuda/bin/nvprof
LFLAGS += -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio -lpthread -L/usr/local/lib
LFLAGS += -lv4l2 
CFLAGS += -std=c++11 -g -lineinfo --use_fast_math
INCLUDES += -I/usr/include/opencv4

all: build run

build: 
	$(CC) $(CFLAGS) $(LFLAGS) $(INCLUDES) server.cu -o server

run:
	./server

prof: 
	sudo $(PROFILER) ./server
