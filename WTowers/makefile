NVCC = nvcc
CC = gcc
CXX = g++

CFLAGS= -g -gencode arch=compute_35,code=sm_35
CXXFLAGS=-std=c++11 -g -G -gencode arch=compute_35,code=sm_35 -I../common
LDFLAGS=-lhdf5 -lm -lcufft -lcublas -gencode arch=compute_35,code=sm_35 -L../common -lhdf5_cu
CUDA_FLAGS=-Xcompiler -fopenmp -Xptxas -v,

BINDIR = ./

all: wtowers.out


clean:
	rm *.o
	rm *.out


wtowers.out: wtowers_cuda.o wtowers.o 
	$(NVCC) $^ -o wtowers.out $(LDFLAGS)

wtowers_cuda.o: wtowers_cuda.cu
	$(NVCC) -c $< -o $@ $(CXXFLAGS) $(CUDA_FLAGS)

wtowers.o: wtowers.cpp
	$(NVCC) -c $< -o $@ $(CXXFLAGS)

test:
	./wtowers.out -theta=0.1 -lambda=20480 -image=image.out -wkernel=../data/crocodile_data/kernels/vla_w20_static_size15.h5 -vis=../data/crocodile_data/vis/vlaa_theta0.1.h5 -subgrid=512 -margin=64 -winc=20

test_wproj:
	./wtowers.out -theta=0.1 -lambda=20480 -image=image.out -wkernel=../data/crocodile_data/kernels/vla_w20_static_size15.h5 -vis=../data/crocodile_data/vis/vlaa_theta0.1.h5 -bl_min=0 -bl_max=20000 -wproj -flat
