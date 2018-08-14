NVCC = nvcc
CC = gcc
CXX = g++

override CXXFLAGS += -std=c++11 -I../common
override LDFLAGS += -L/data4/jkent/bifrost/lib -lhdf5 -lm -L../common -lhdf5_cu
override CUDA_FLAGS += -Xcompiler -fopenmp -Xptxas -v,

BINDIR = ./

all: dft.out


clean:
	rm -f *.o
	rm -f *.out


dft.out: dft_cuda.o dft.o 
	$(NVCC) $^ -o ../../bin/dft.out $(LDFLAGS)

dft_cuda.o: dft_cuda.cu
	$(NVCC) -c $< -o $@ $(CXXFLAGS) $(CUDA_FLAGS)

dft.o: dft.cpp
	$(NVCC) -c $< -o $@ $(CXXFLAGS) 

