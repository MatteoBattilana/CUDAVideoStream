#
# LIST OF SUPPRESSABLE WARNINGS: http://www.ssl.berkeley.edu/~jimm/grizzly_docs/SSL/opt/intel/cc/9.0/lib/locale/en_US/mcpcom.msg
#


.PHONY: run prof

CC = /usr/local/cuda/bin/nvcc
PROFILER = /usr/local/cuda/bin/nvprof

LFLAGS += -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio -lpthread -L/usr/local/lib
LFLAGS += -lv4l2 
CFLAGS += -std=c++11 -g -lineinfo --use_fast_math
CFLAGS += -Xcudafe "--diag_suppress=partial_override"
INCLUDES += -I/usr/include/opencv4

EXEC_FILE = server

all: run

$(EXEC_FILE): $(EXEC_FILE).o
	$(info [L] $^)
	@$(CC) $^ $(CFLAGS) $(LFLAGS) $(INCLUDES) -o $@

%.o: %.cu
	$(info [C] $^)
	@$(CC) -c $^ $(CFLAGS) $(LFLAGS) $(INCLUDES)

clean:
	rm -f *.o $(EXEC_FILE)

run: $(EXEC_FILE)
	./$(EXEC_FILE)

prof: $(EXEC_FILE)
	sudo $(PROFILER) ./$(EXEC_FILE)
