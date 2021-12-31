CC = /usr/local/cuda/bin/nvcc
LFLAGS +=  
CFLAGS += -g
INCLUDES +=

all: build run

build: 
	$(CC) $(CFLAGS) $(LFLAGS) $(INCLUDES) pixel_diff.cu -o pixel_diff

run:
	./pixel_diff 