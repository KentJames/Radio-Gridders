include ../../user.mk

override CXXFLAGS +=-std=c++14 -I../common
override LDFLAGS += -lhdf5 -lm  -L../common -lhdf5_r -lfftw3
override CUDA_FLAGS +=-std=c++14 -I../common

BINDIR = ./

all: ../../bin/wstack.out


clean:
	rm -f *.o 
	rm -f *.out


../../bin/wstack.out: wstack.o libpredict.a
	$(CXX) wstack.o -o $@ $(LDFLAGS) -L ./ -lpredict

wstack.o: wstack.cpp
	$(CXX) -c $^ -o $@ $(CXXFLAGS)

predict.o: predict.cpp
	$(CXX) -c $^ -o $@ $(CXXFLAGS)


ifndef CUDA
libpredict.a: predict.o
	ar rcs $@ $^
else
predict_cu.o: predict.cu
	$(NVCC) -c $^ -o $@ $(CUDA_FLAGS)
libpredict.a: predict.o predict_cu.o
	ar rcs $@ $^
endif
