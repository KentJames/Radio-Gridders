NVCC = nvcc
CC = gcc
CXX = g++

CFLAGS= -g -gencode arch=compute_35,code=sm_35
CXXFLAGS=-std=c++11 -gencode arch=compute_35,code=sm_35 -I../common
LDFLAGS=-lhdf5 -lm -gencode arch=compute_35,code=sm_35 -L../common -lhdf5_cu


BINDIR = ./

all: wtowers.out


clean:
	rm *.o
	rm *.out


wtowers.out: wtowers_cuda.o wtowers.o 
	$(NVCC) $^ -o wtowers.out $(LDFLAGS)

wtowers_cuda.o: wtowers_cuda.cu
	$(NVCC) -c $< -o $@ $(CXXFLAGS)

wtowers.o: wtowers.cpp
	$(NVCC) -c $< -o $@ $(CXXFLAGS)

