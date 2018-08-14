NVCC = nvcc
CC = gcc
CXX = g++

override CXXFLAGS +=-std=c++11 -I../common
override LDFLAGS += -L/data4/jkent/bifrost/lib -lm -gencode arch=compute_52,code=sm_52 -L../common
CUDA_FLAGS +=  -Xcompiler -fopenmp -Xptxas -v,

BINDIR = ./

all: epic.out

clean:
	rm -f *.o 
	rm -f *.out 

epic.out: epic_grid.o 
	$(NVCC) $^ -o ../../bin/epic.out $(LDFLAGS)

epic_grid.o: epic_grid.cu
	$(NVCC) -c $< -o $@ $(CXXFLAGS) $(CUDA_FLAGS)
