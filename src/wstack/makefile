include ../../user.mk

ifndef CUDA
override CXXFLAGS += -fopenmp -std=c++14 -I../common
else
override CXXFLAGS +=-DCUDA_ACCELERATION -std=c++14 -I../common -I/usr/local/cuda/include
endif

ifndef CUDA
override LDFLAGS +=  -lm  -L../common -lomp -lhdf5_r -lfftw3 -lhdf5 
else
override LDFLAGS +=  -lm  -L../common -lomp -lhdf5_cu -lfftw3 -lhdf5 -L/usr/local/cuda/lib -lcuda -lcudart -lcufft
endif

override CUDA_FLAGS +=-std=c++14 -I../common

BINDIR = ./

all: wstack.out


clean:
	rm -f *.o 
	rm -f *.out


wstack.out: wstack.o libpredict.a
	$(CXX) wstack.o -o $@  -L ./ -lpredict $(LDFLAGS)
	cp $@ ../../bin/

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
