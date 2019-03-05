NVCC = nvcc
CC = gcc
CXX = g++

override CXXFLAGS +=-std=c++11 -I../common
override LDFLAGS += -lhdf5 -lm  -L../common -lhdf5_r -lfftw3
override CUDA_FLAGS += 

BINDIR = ./

all: wstack.out


clean:
	rm -f *.o 
	rm -f *.out


wstack.out: wstack.o predict.o
	$(CXX) $^ -o ../../bin/wstack.out $(LDFLAGS)


wstack.o: wstack.cpp
	$(CXX) -c $< -o $@ $(CXXFLAGS)

predict.o: predict.cpp
	$(CXX) -c $< -o $@ $(CXXFLAGS)
