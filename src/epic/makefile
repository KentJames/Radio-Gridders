NVCC = nvcc
CC = gcc
CXX = g++

CFLAGS= -gencode arch=compute_52,code=sm_52
override CXXFLAGS+=-std=c++11 -gencode arch=compute_52,code=sm_52 -I../common
LDFLAGS= -L/data4/jkent/bifrost/lib -lm -gencode arch=compute_52,code=sm_52 -L../common
CUDA_FLAGS=-Xcompiler -fopenmp -Xptxas -v,

BINDIR = ./

all: epic.out

clean:
	rm *.o
	rm *.out

epic.out: epic_grid.o 
	$(NVCC) $^ -o ../../bin/epic.out $(LDFLAGS)

epic_grid.o: epic_grid.cu
	$(NVCC) -c $< -o $@ $(CXXFLAGS) $(CUDA_FLAGS)
