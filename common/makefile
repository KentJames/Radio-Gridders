CC = gcc


all: hdf5.a

hdf5.a: hdf5.o
	ar rcs $@ $^

hdf5.o: hdf5.c hdf5_h.h
	gcc -c -o $@ $< -lhdf5
