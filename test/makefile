NVCC = nvcc
CXX = clang++
CC = clang

override CXXFLAGS +=-std=c++14 -I../src/common
override LDFLAGS += -lhdf5 -lm -L../src/common -lhdf5_r -lfftw3

all: test_wstack.out

clean:
	rm -f *.o 
	rm -f *.out

test_wstack.out: test_wstack.o 
	$(CXX) $^ -o test_wstack.out $(LDFLAGS) -L../src/wstack/ -lpredict

test_wstack.o: test_wstack.cpp
	$(CXX) -c $< -o $@  $(CXXFLAGS) -I../src/wstack/

