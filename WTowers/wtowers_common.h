#include "hdf5_h.h"

#ifndef WTOWER_H
#define WTOWER_H

#include <cuComplex.h>
#include "cuda.h"
#include "cuda_runtime_api.h"
#define THREADS_BLOCK 16

//Our CUDA Prototypes.

#ifdef __CUDACC__

__host__ __device__ inline cuDoubleComplex cu_cexp_d (cuDoubleComplex z);




//Kernels

__global__ void scatter_grid_kernel(struct vis_data *vis, struct w_kernel_data *wkern,
				    cuDoubleComplex *uvgrid, int max_support, int grid_size);

__global__ void fresnel_pattern_kernel(cuDoubleComplex *subimg, cuDoubleComplex *subgrid,
				       cuDoubleComplex *fresnel, int subgrid_size, int w_plane);



#endif

__host__ inline void fft_shift(cuDoubleComplex *uvgrid, int grid_size);

cudaError_t wtowers_CUDA(const char* visfile, const char* wkernfile, int grid_size,
			   double theta,  double lambda, double bl_min, double bl_max,
			 int subgrid_size, int subgrid_margin, double witer);


#endif
