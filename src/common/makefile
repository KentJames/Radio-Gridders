include ../../user.mk

ifdef CUDA
override NVCCFLAGS += -gencode arch=compute_50,code=sm_50
endif
override CXXFLAGS +=
override LDFLAGS += 

ifndef CUDA
all: libhdf5_r.a 
else
all: libhdf5_cu.a libhdf5_r.a libcommon_kernels.a
endif
clean:
	rm -f *.o
	rm -f *.a


libhdf5_r.a: hdf5.o
	ar rcs $@ $^

hdf5.o: hdf5.c
	$(CC) -c -o $@ $< $(CXXFLAGS)

ifdef CUDA
libhdf5_cu.a: hdf5_cu.o
	ar rcs $@ $^

libcommon_kernels.a: common_kernels.o
	ar rcs $@ $^


hdf5_cu.o: hdf5.c
	$(NVCC) -c -o $@ $<  -DCUDA_ACCELERATION -lhdf5 $(NVCCFLAGS)

common_kernels.o: common_kernels.cu
	$(NVCC) -c -o $@ $<  -lhdf5 $(NVCCFLAGS)
endif



