NVCC = nvcc
CC = gcc
CXX = g++

CFLAGS= -g -gencode arch=compute_35,code=sm_35
CXXFLAGS=-std=c++11 -g -G -gencode arch=compute_35,code=sm_35
LDFLAGS=-lhdf5 -lm -gencode arch=compute_35,code=sm_35


BINDIR = ./

all: dft.out


clean:
	rm *.o
	rm *.out


dft.out: hdf5.o dft_cuda.o dft.o 
	$(NVCC) $^ -o dft.out $(LDFLAGS)

dft_cuda.o: dft_cuda.cu
	$(NVCC) -c $< -o $@ $(CXXFLAGS)

dft.o: dft.cpp
	$(NVCC) -c $< -o $@ $(CXXFLAGS)

hdf5.o: hdf5.c
	$(NVCC) -c $< -o $@ $(CFLAGS)
