#
# LIST OF SUPPRESSABLE WARNINGS: http://www.ssl.berkeley.edu/~jimm/grizzly_docs/SSL/opt/intel/cc/9.0/lib/locale/en_US/mcpcom.msg
#


.PHONY: run prof build

CCNV = /usr/local/cuda/bin/nvcc
CCPP = g++
PROFILER = /usr/local/cuda/bin/nvprof

LFLAGS += 
CFLAGS += -std=c++11 -g -MMD
INCLUDES += 

NVLFLAGS +=  
NVCFLAGS += $(CFLAGS) -lineinfo --use_fast_math 
NVCFLAGS += -Xcudafe "--diag_suppress=partial_override"
NVINCLUDES += 

CVLFLAGS += -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio -lpthread -L/usr/local/lib
CVCFLAGS += $(CFLAGS) 
CVINCLUDES += -I/usr/include/opencv4

SOURCES := $(wildcard src/*.c*)
SOURCES := $(shell echo $(SOURCES) | sed -E "s/src\///g")
OBJECTS := $(shell echo $(SOURCES) | sed -E "s/(cpp|cu)/o/g")
EXEC_FILE = server

all: run

$(EXEC_FILE): $(OBJECTS) 
	$(info [L] nvcc $^)
	@$(CCNV) $^ $(NVLFLAGS) $(NVINCLUDES) $(CVLFLAGS) $(CVINCLUDES) -o $@

%.o: src/%.cu
	$(info [C] nvcc $<)
	@$(CCNV) -c $< $(NVCFLAGS) $(NVINCLUDES)

threads.o: src/threads.cpp
	$(info [C]  g++ $<)
	@$(CCPP) -c $< $(CVCFLAGS) $(CVINCLUDES)

%.o: src/%.cpp
	$(info [C]  g++ $<)
	@$(CCPP) -c $< $(CFLAGS) $(INCLUDES)

-include $(OBJECTS:.o=.d)

clean:
	rm -f *.o *.d $(EXEC_FILE)

build: $(EXEC_FILE)

run: $(EXEC_FILE)
	./$(EXEC_FILE)

prof: $(EXEC_FILE)
	sudo $(PROFILER) ./$(EXEC_FILE)
