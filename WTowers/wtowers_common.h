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

__device__ inline void scatter_grid_point(int max_supp, cuDoubleComplex *uvgrid,
					  struct bl_data **bin, int myU, int myV,
					  int subgrid_size, int subgrid_pitch);

//Kernels

__global__ void scatter_grid_kernel(struct vis_data *vis, struct w_kernel_data *wkern,
				    cuDoubleComplex *uvgrid, int max_support, int grid_size);

__global__ void fresnel_pattern_kernel(cuDoubleComplex *subimg, cuDoubleComplex *subgrid,
				       cuDoubleComplex *fresnel, int subgrid_size, int w_plane);

#endif

//Host Functions

__host__ void cudaReallocManaged(void **ptr, int size, int size_original);

__host__ inline double lambda_min(struct bl_data *bl_data, double u);

__host__ inline double lambda_max(struct bl_data *bl_data, double u);

__host__ inline void fft_shift(cuDoubleComplex *uvgrid, int grid_size);

__host__ inline void bin_visibilities(struct vis_data *vis, struct bl_data ***bins,
				      int chunk_count, int wincrement, double theta,
				      int grid_size, int chunk_size);

__host__ cudaError_t wtowers_CUDA(const char* visfile, const char* wkernfile, int grid_size,
			   double theta,  double lambda, double bl_min, double bl_max,
			 int subgrid_size, int subgrid_margin, double witer);

__host__ cudaError_t wprojection_CUDA(const char* visfile, const char* wkernfile, int grid_size,
				      double theta,  double lambda, double bl_min, double bl_max, 
				      int threads_per_block);

__host__ cudaError_t wprojection_CUDA_flat(const char* visfile, const char* wkernfile, int grid_size,
					   double theta,  double lambda, double bl_min, double bl_max,
					   int threads_per_block);

#endif
