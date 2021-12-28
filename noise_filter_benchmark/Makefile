CC = /usr/local/cuda/bin/nvcc
PROFILER = /usr/local/cuda/bin/nvprof
LFLAGS += -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio -lpthread -L/usr/local/lib
LFLAGS += -lv4l2 
CFLAGS += -std=c++11 -g -lineinfo --use_fast_math
INCLUDES += -I/usr/include/opencv4

all: build-v1 run

build-v3: 
	$(CC) $(CFLAGS) $(LFLAGS) $(INCLUDES) $(SRCS) v3.cu -o noiseFilter

build-v2: 
	$(CC) $(CFLAGS) $(LFLAGS) $(INCLUDES) $(SRCS) v2.cu -o noiseFilter

build-v1: 
	$(CC) $(CFLAGS) $(LFLAGS) $(INCLUDES) $(SRCS) v1.cu -o noiseFilter

run:
	./noiseFilter

prof:
	sudo $(PROFILER) ./noiseFilter