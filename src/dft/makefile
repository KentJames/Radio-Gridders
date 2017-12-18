NVCC = nvcc
CC = gcc
CXX = g++

CFLAGS= -g -gencode arch=compute_35,code=sm_35
override CXXFLAGS=-std=c++11 -gencode arch=compute_35,code=sm_35 -I../common
LDFLAGS=-lhdf5 -lm -gencode arch=compute_35,code=sm_35 -L../common -lhdf5_cu


BINDIR = ./

all: dft.out


clean:
	rm *.o
	rm *.out


dft.out: dft_cuda.o dft.o 
	$(NVCC) $^ -o ../../bin/dft.out $(LDFLAGS)

dft_cuda.o: dft_cuda.cu
	$(NVCC) -c $< -o $@ $(CXXFLAGS)

dft.o: dft.cpp
	$(NVCC) -c $< -o $@ $(CXXFLAGS)

