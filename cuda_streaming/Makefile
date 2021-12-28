CC = /usr/local/cuda/bin/nvcc
PROFILER = /usr/local/cuda/bin/nvprof
LFLAGS += -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio -lpthread -L/usr/local/lib
LFLAGS += -lv4l2
CFLAGS += -std=c++11 -g -lineinfo
INCLUDES += -I/usr/include/opencv4
# INCLUDES += 
# SRCS += server.cu v4l.cpp
SRCS += test.cu

all: build run

build: test 
	# $(CC) $(CFLAGS) $(LFLAGS) $(INCLUDES) $(SRCS) -o server

run:
	./server

test:
	$(CC) $(CFLAGS) $(LFLAGS) $(INCLUDES) test.cu -o server

test2:
	$(CC) $(CFLAGS) $(LFLAGS) $(INCLUDES) test2.cu -o server

test3:
	$(CC) $(CFLAGS) $(INCLUDES) test3.cu -o server

test4:
	$(CC) test4.cpp $(CFLAGS) $(LFLAGS) $(INCLUDES) -o server

prof: 
	sudo $(PROFILER) ./server