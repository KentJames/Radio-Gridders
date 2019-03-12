NVCC = nvcc
CC = clang
CXX = clang++

override CXXFLAGS +=-std=c++14 -I../common
override LDFLAGS += -lhdf5 -lm  -L../common -lhdf5_r -lfftw3
override CUDA_FLAGS += 

BINDIR = ./

all: wstack.out libpredict.a


clean:
	rm -f *.o 
	rm -f *.out


wstack.out: wstack.o libpredict.a
	$(CXX) $^ -o ../../bin/wstack.out $(LDFLAGS) -L ./ -lpredict

wstack.o: wstack.cpp
	$(CXX) -c $< -o $@ $(CXXFLAGS)

predict.o: predict.cpp
	$(CXX) -c $< -o $@ $(CXXFLAGS)

libpredict.a: predict.o
	ar rcs $@ $^
