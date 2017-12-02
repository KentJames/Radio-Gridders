#include "hdf5_h.h"

#ifndef GRID_H
#define GRID_H

#include <cuComplex.h>
#include "cuda.h"
#include "cuda_runtime_api.h"
#define THREADS_BLOCK 16

cudaError_t image_dft_host(const char* visfile,  int grid_size,
			   double theta,  double lambda, double bl_min, double bl_max,
			   int blocks, int threads_block);

cudaError_t image_dft_host_flat(const char* visfile,  int grid_size,
			   double theta,  double lambda, double bl_min, double bl_max,
			   int blocks, int threads_block);


#endif
