NVCC = nvcc
CC = gcc
override NVCCFLAGS += -gencode arch=compute_50,code=sm_50
override CXXFLAGS +=
override LDFLAGS += 


all: libhdf5_cu.a libhdf5_r.a

clean:
	rm -f *.o
	rm -f *.a

libhdf5_cu.a: hdf5_cu.o
	ar rcs $@ $^
libhdf5_r.a: hdf5.o
	ar rcs $@ $^
hdf5_cu.o: hdf5.cu 
	$(NVCC) -c -o $@ $<  -lhdf5 $(LDFLAGS) $(NVCCFLAGS)
hdf5.o: hdf5.c
	$(CC) -c -o $@ $<  -lhdf5 $(LDFLAGS) $(CXXFLAGS)
