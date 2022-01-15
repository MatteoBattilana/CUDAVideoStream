CC = /usr/local/cuda/bin/nvcc
LFLAGS += -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio -lopencv_video -L/usr/local/lib
CFLAGS += -std=c++11 -g 
INCLUDES += -I/usr/include/opencv4

all-v1: build-v1 run
all-v2: build-v2 run


build-v1:
	$(CC) $(CFLAGS) $(LFLAGS) $(INCLUDES) gpu-v1.cu -o histogram

build-v2:
	$(CC) $(CFLAGS) $(LFLAGS) $(INCLUDES) gpu-v2.cu -o histogram

run:
	./histogram

prof:
	sudo $(PROFILER) ./histogram