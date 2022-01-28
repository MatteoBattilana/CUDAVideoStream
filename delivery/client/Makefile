CC = g++

LFLAGS += -g -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio -lpthread
INCLUDES += -I/usr/include/opencv4

EXEC_FILE = opencv

all: run

$(EXEC_FILE): $(EXEC_FILE).o
	$(info [L] $^)
	@$(CC) $^ $(CFLAGS) $(LFLAGS) $(INCLUDES) -o $@

%.o: %.cpp
	$(info [C] $^)
	@$(CC) -c $^ $(CFLAGS) $(LFLAGS) $(INCLUDES) 

clean:
	rm -f *.o $(EXEC_FILE)

run: $(EXEC_FILE)
	./$(EXEC_FILE)
