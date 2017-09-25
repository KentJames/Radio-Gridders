#include "hdf5_h.h"

#ifndef WTOWER_H
#define WTOWER_H

#include <cuComplex.h>
#include "cuda.h"
#include "cuda_runtime_api.h"
#define THREADS_BLOCK 16


#ifdef __CUDACC__

//Our CUDA Prototypes.

__device__ cuDoubleComplex calculate_dft_sum(struct vis_data *vis, double l, double m);

__global__ void image_dft(struct vis_data *vis, cuDoubleComplex *uvgrid, int grid_size,
			  double lambda, int iter, int N);


#endif

cudaError_t wtowers_host(const char* visfile, const char* wkernfile, int grid_size,
			   double theta,  double lambda, double bl_min, double bl_max,
			   int iter);


#endif
