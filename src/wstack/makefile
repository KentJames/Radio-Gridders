include ../../user.mk

ifndef CUDA
override CXXFLAGS += -fopenmp -std=c++14 -I../common
else
override CXXFLAGS += -g -fopenmp -DCUDA_ACCELERATION -std=c++14 -I../common -I/usr/local/cuda/include 
endif

ifndef CUDA
override LDFLAGS +=  -lm  -L../common -lgomp -lhdf5_r -lfftw3_threads -lfftw3 -lhdf5 
else
override LDFLAGS += -L../common -lgomp -lhdf5_cu -lfftw3_threads -lfftw3 -lm -lhdf5 -L$(CUDA_LIBDIR) -lcuda -lcudart -lcufft 
endif

override CUDA_FLAGS += -std=c++14  -use_fast_math -maxrregcount=64 -I../common -gencode arch=compute_60,code=sm_60

BINDIR = ./

all: wstack.out


clean:
	rm -f *.o 
	rm -f *.out

ifndef CUDA
wstack.out: wstack.o libpredict.a
	$(CXX) wstack.o -o $@  -L ./ -lpredict $(LDFLAGS)
	cp $@ ../../bin/
else
wstack.out: wstack.o libpredict.a
	$(NVCC) wstack.o -o $@  -L ./ -lpredict $(LDFLAGS)
	cp $@ ../../bin/
endif

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
