#include "hdf5_h.h"

#ifndef GRID_H
#define GRID_H

#include <cuComplex.h>
#include "cuda.h"
#include "cuda_runtime_api.h"
#define THREADS_BLOCK 16

#ifdef __CUDACC__

//Our CUDA Prototypes.

__device__ cuDoubleComplex calculate_dft_sum(struct vis_data *vis, double l, double m);

__global__ void image_dft(struct vis_data *vis, cuDoubleComplex *uvgrid, int grid_size,
			  double lambda);


#endif

cudaError_t image_dft_host(const char* visfile,  int grid_size,
			   double theta,  double lambda, double bl_min, double bl_max,
			   int blocks, int threads_block);


#endif
