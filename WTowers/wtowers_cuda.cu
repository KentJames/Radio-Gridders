//C++ Includes
#include <iostream>
#include <fstream>
#include <cstdlib>

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


//W-Towers Wrapper.
__host__ cudaError_t wtowers_host(const char* visfile, const char* wkernfile, int grid_size,
			   double theta,  double lambda, double bl_min, double bl_max,
				  int iter){

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
  

  
  return error;

}
