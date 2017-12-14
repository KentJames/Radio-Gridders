#include "hdf5_h.h"
#include "radio.cuh"

#ifndef GRID_H
#define GRID_H

#include <cuComplex.h>
#include "cuda.h"
#include "cuda_runtime_api.h"
#define THREADS_BLOCK 16

cudaError_t image_dft_host(const char* visfile,  cuDoubleComplex *grid_host, cuDoubleComplex *grid_dev,
			   int grid_size, double theta,  double lambda, double bl_min, double bl_max,
			   int blocks, int threads_block);

cudaError_t image_dft_host_flat(const char* visfile,  cuDoubleComplex *grid_host, cuDoubleComplex *grid_dev,
				int grid_size, double theta,  double lambda, double bl_min, double bl_max,
				int blocks, int threads_block);


#endif
