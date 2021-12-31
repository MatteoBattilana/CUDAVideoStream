CC = /usr/local/cuda/bin/nvcc
LFLAGS +=  
CFLAGS += -g
INCLUDES +=

all: build run

build: 
	$(CC) $(CFLAGS) $(LFLAGS) $(INCLUDES) threads.cu -o threads

run:
	./threads