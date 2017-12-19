CC = gcc
CC_FLAGS = -Wall -g -O3 -ftree-vectorize -ffast-math -fopenmp
CFLAGS = 
LFLAGS = -g -lgomp -lhdf5 -lfftw3 -lfftw3_threads -lm -lc -ldl

all: gridder.out dft.out


clean: 
		rm *.o
		rm *.out

gridder.out: main.o hdf5.o perf.o grid.o
		$(CC) main.o hdf5.o perf.o grid.o -o ../../bin/grid_CPU.out $(LFLAGS)

dft.out:     dft.o hdf5.o perf.o grid.o 
		$(CC) dft.o hdf5.o perf.o grid.o -o ../../bin/dft_CPU.out $(LFLAGS)


dft.o: 	dft.c 	
		$(CC) $(CFLAGS) $(CC_FLAGS) -c dft.c

main.o: main.c
		$(CC) $(CFLAGS) $(CC_FLAGS) -c main.c

hdf5.o: hdf5.c 
		$(CC) $(CFLAGS) $(CC_FLAGS) -c hdf5.c

perf.o: perf.c 
		$(CC) $(CFLAGS) $(CC_FLAGS) -c perf.c

grid.o: grid.c
		$(CC) $(CFLAGS) $(CC_FLAGS) -c grid.c

wtowers_test: 
	./gridder.out --theta=0.08 --lambda=300000 --wkern=../../data/kernels/SKA1_Low_wkern.h5 --subgrid=100 --margin=20 --winc=10 ../../data/vis/SKA1_Low_vis.h5

wtowers_test_vla: 
	./gridder.out --theta=0.1 --lambda=18000 --image=image_towers.out --wkern=../../data/kernels/vla_w20_static_size15.h5 --subgrid=100 --margin=20 --winc=20 ../../data/vis/vlaa_theta0.1.h5

test:
	./gridder.out --theta=0.1 --lambda=18000 --image=image_proj.out --wkern=../../data/kernels/vlaa_w20_var.h5 ../../data/vis/vlaa_theta0.1.h5

.PHONY: all clean test
