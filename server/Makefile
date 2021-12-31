#
# LIST OF SUPPRESSABLE WARNINGS: http://www.ssl.berkeley.edu/~jimm/grizzly_docs/SSL/opt/intel/cc/9.0/lib/locale/en_US/mcpcom.msg
#


.PHONY: run prof build

CCNV = /usr/local/cuda/bin/nvcc
CCPP = g++
PROFILER = /usr/local/cuda/bin/nvprof

NVLFLAGS +=  
NVCFLAGS += -std=c++11 -g -lineinfo --use_fast_math
NVCFLAGS += -Xcudafe "--diag_suppress=partial_override"
NVINCLUDES += 

CVLFLAGS += -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio -lpthread -L/usr/local/lib
CVCFLAGS += -std=c++11 -g 
CVINCLUDES += -I/usr/include/opencv4

EXEC_FILE = server

all: run

$(EXEC_FILE): kernels.o threads.o $(EXEC_FILE).o 
	$(info [L] $^)
	@$(CCNV) $^ $(CVFLAGS) $(CVLFLAGS) $(CVINCLUDES) -o $@

server.o: src/server.cu
	$(info [C] (nvcc) $^)
	@$(CCNV) -c $^ $(NVCFLAGS) $(NVLFLAGS) $(NVINCLUDES) $(CVCFLAGS) $(CVLFLAGS) $(CVINCLUDES)

%.o: src/%.cu
	$(info [C] (nvcc) $^)
	@$(CCNV) -c $^ $(NVCFLAGS) $(NVLFLAGS) $(NVINCLUDES)

%.o: src/%.cpp
	$(info [C] (nvcc) $^)
	@$(CCNV) -c $^ $(CVCFLAGS) $(CVLFLAGS) $(CVINCLUDES)

clean:
	rm -f *.o $(EXEC_FILE)

build: $(EXEC_FILE)

run: $(EXEC_FILE)
	./$(EXEC_FILE)

prof: $(EXEC_FILE)
	sudo $(PROFILER) ./$(EXEC_FILE)
