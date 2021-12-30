CC = /usr/local/cuda/bin/nvcc
PROFILER = /usr/local/cuda/bin/nvprof
LFLAGS += -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio -lpthread -L/usr/local/lib
LFLAGS += -lv4l2 
CFLAGS += -std=c++11 -g  --use_fast_math
INCLUDES += -I/usr/include/opencv4 

all: build-v4 run

build-cpu: 
	$(CC) $(CFLAGS) $(LFLAGS) $(INCLUDES) cpu.cu -o heatMap

build-v0: 
	$(CC) $(CFLAGS) $(LFLAGS) $(INCLUDES) $(SRCS) v0.cu -o heatMap
	
build-v1: 
	$(CC) $(CFLAGS) $(LFLAGS) $(INCLUDES) $(SRCS) v1.cu -o heatMap
	
build-v1-assembly: 
	$(CC) $(CFLAGS) $(LFLAGS) $(INCLUDES) $(SRCS) -ptx -o heatMap.ptx v1.cu

build-v2: 
	$(CC) $(CFLAGS) $(LFLAGS) $(INCLUDES) $(SRCS) v2.cu -o heatMap

build-v3: 
	$(CC) $(CFLAGS) $(LFLAGS) $(INCLUDES) $(SRCS) v3.cu -o heatMap

build-v3-assembly: 
	$(CC) $(CFLAGS) $(LFLAGS) $(INCLUDES) $(SRCS) -ptx -o heatMap.ptx v3.cu

build-v4: 
	$(CC) $(CFLAGS) $(LFLAGS) $(INCLUDES) $(SRCS) v4.cu -o heatMap

run:
	./heatMap

prof:
	sudo $(PROFILER) ./heatMap