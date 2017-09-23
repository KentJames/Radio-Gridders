CC = nvcc


all: libhdf5_cu.a

libhdf5_cu.a: hdf5_cu.o
	ar rcs $@ $^

hdf5_cu.o: hdf5.c hdf5_h.h
	$(CC) -c -o $@ $< -lhdf5
