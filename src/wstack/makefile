NVCC = nvcc
CC = gcc
CXX = g++

override CXXFLAGS +=-std=c++11 -I../common
override LDFLAGS += -lhdf5 -lm -lcufft -lcublas -gencode arch=compute_52,code=sm_52 -L../common -lhdf5_cu -lfftw3
override CUDA_FLAGS += 

BINDIR = ./

all: wstack.out


clean:
	rm -f *.o 
	rm -f *.out


wstack.out: wstack.o predict.o
	$(NVCC) $^ -o ../../bin/wstack.out $(LDFLAGS)


wstack.o: wstack.cpp
	$(NVCC) -c $< -o $@ $(CXXFLAGS)

predict.o: predict.cpp
	$(NVCC) -c $< -o $@ $(CXXFLAGS)
