//C++ Includes
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cassert>

//CUDA Includes
#include <cuComplex.h>
#include <cufft.h>
#include "cuda.h"
#include "math.h"
#include "cuda_runtime_api.h"

//Our Include
#include "wtowers_common.h"

#define cudaError_check(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__host__ __device__ inline cuDoubleComplex cu_cexp_d (cuDoubleComplex z){

  cuDoubleComplex res;
  double t = exp (z.x);
  sincos (z.y, &res.y, &res.x);
  res.x *= t;
  res.y *= t;
  return res;

}

__host__ __device__ inline void fft_shift(cuDoubleComplex *uvgrid, int grid_size) {

    // Shift the FFT
    assert(grid_size % 2 == 0);
    int x, y;
    for (y = 0; y < grid_size; y++) {
        for (x = 0; x < grid_size/2; x++) {
            int ix0 = y * grid_size + x;
            int ix1 = (ix0 + (grid_size+1) * (grid_size/2)) % (grid_size*grid_size);
            cuDoubleComplex temp = uvgrid[ix0];
            uvgrid[ix0] = uvgrid[ix1];
            uvgrid[ix1] = temp;
        }
    }

}



//W-Towers Wrapper.
__host__ cudaError_t wtowers_CUDA(const char* visfile, const char* wkernfile, int grid_size,
			   double theta,  double lambda, double bl_min, double bl_max,
				  int subgrid_size, int subgrid_margin, double wincrement){

  cudaError_t error;
  cudaEvent_t start, stop;
  float elapsedTime;

  struct vis_data *vis_dat;
  struct w_kernel_data *wkern_dat;


  cudaError_check(cudaMallocManaged((void **)&vis_dat, sizeof(struct vis_data), cudaMemAttachGlobal));
  cudaError_check(cudaMallocManaged((void **)&wkern_dat, sizeof(struct w_kernel_data), cudaMemAttachGlobal));

  int error_hdf5;
  error_hdf5 = load_vis_CUDA(visfile,vis_dat,bl_min,bl_max);
  if (error_hdf5) {
    std::cout << "Failed to Load Visibilities \n";
    return error;
  }
  error_hdf5 = load_wkern_CUDA(wkernfile, theta, wkern_dat);
  if (error_hdf5) {
    std::cout << "Failed to Load W-Kernels \n";
    return error;
  }

  int total_gs = grid_size * grid_size;
  
  cuDoubleComplex *grid_dev, *grid_host;

  cudaError_check(cudaMalloc((void **)&grid_dev, total_gs * sizeof(cuDoubleComplex)));
  cudaError_check(cudaMallocHost((void **)&grid_host, total_gs * sizeof(cuDoubleComplex)));

  int subgrid_mem_size = sizeof(cuDoubleComplex) * subgrid_size * subgrid_size;
  
  cuDoubleComplex *wtransfer;

  cudaError_check(cudaMallocManaged((void **)&wtransfer, subgrid_mem_size, cudaMemAttachGlobal));

  int x,y;
  //The fresnel pattern. A kernel can be written for this.
  for (y=0; y < subgrid_size; ++y){

    for (x=0; x < subgrid_size; ++x){

      double l = theta * (double)(x - subgrid_size / 2) / subgrid_size;
      double m = theta * (double)(y - subgrid_size / 2) / subgrid_size;
      double ph = wincrement * (1 - sqrt(1 - l*l - m*m));

      cuDoubleComplex wtrans = make_cuDoubleComplex(0, 2 * M_PI * ph);
      wtransfer[y * subgrid_size + x] = cu_cexp_d(wtrans);
    }

  }

  cufftHandle fft_plan;
  cufftPlan2d(&fft_plan,subgrid_size,subgrid_size,CUFFT_D2Z);

  
  

  
  return error;

}
