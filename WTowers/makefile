NVCC = nvcc
CC = gcc
CXX = g++

CFLAGS= -g -gencode arch=compute_35,code=sm_35
CXXFLAGS=-std=c++11 -g -G -gencode arch=compute_35,code=sm_35 -I../common
LDFLAGS=-lhdf5 -lm -lcufft -gencode arch=compute_35,code=sm_35 -L../common -lhdf5_cu


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

test:
	./wtowers.out -theta=0.1 -lambda=20480 -image=image.out -wkernel=../../crocodile/data/kernels/vlaa_w20_static.h5 -vis=../../crocodile/data/vis/SKA1_Low_quick.h5 -subgrid=512 -margin=64 -winc=20
