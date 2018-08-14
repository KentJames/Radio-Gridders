CC = nvcc

override CXXFLAGS += -gencode arch=compute_50,code=sm_50
override LDFLAGS += 


all: libhdf5_cu.a

clean:
	rm -f *.o
	rm -f *.a

libhdf5_cu.a: hdf5_cu.o
	ar rcs $@ $^

hdf5_cu.o: hdf5.c hdf5_h.h
	$(CC) -c -o $@ $<  -lhdf5 $(LDFLAGS) $(CXXFLAGS)
