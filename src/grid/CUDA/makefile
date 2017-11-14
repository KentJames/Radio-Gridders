CC = nvcc
CPP = g++
CC_FLAGS = -I/usr/include 
CFLAGS = 
LFLAGS = 


all: wtowers.out 


clean: 
		rm *.o
		rm wtowers.out

wtowers.out: wtowers.o wtowers_cuda.o
		$(CC) wtowers.o wtowers_cuda.o -o wtowers.out $(LFLAGS)


wtowers.o: wtowers.cpp
		$(CC) $(CFLAGS) $(CC_FLAGS) -c wtowers.cpp

wtowers_cuda.o: wtowers_cuda.cu 
		$(CC) $(CFLAGS) $(CC_FLAGS) -c wtowers.cu


.PHONY: all clean
