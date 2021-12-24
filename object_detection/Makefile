CC = /usr/local/cuda/bin/nvcc
LFLAGS += -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio -lopencv_video -L/usr/local/lib
CFLAGS += -std=c++11 -g 
INCLUDES += -I/usr/include/opencv4

all: build run

all2: build2 run2

all3: build3 run3

build: 
	$(CC) $(CFLAGS) $(LFLAGS) $(INCLUDES) object_detection.cpp -o object_detection

build2: 
	$(CC) $(CFLAGS) $(LFLAGS) $(INCLUDES) remove_background.cu -o remove_background

build3: 
	$(CC) $(CFLAGS) $(LFLAGS) $(INCLUDES) detection.cpp -o detection

run:
	./object_detection

run2:
	./remove_background

run3:
	./remove_background