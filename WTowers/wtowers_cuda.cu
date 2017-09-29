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



/*****************************
      CUDA Error Checker
******************************/
 
#define cudaError_check(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


/*****************************
        Device Functions
 *****************************/


__host__ __device__ inline cuDoubleComplex cu_cexp_d (cuDoubleComplex z){

  cuDoubleComplex res;
  double t = exp (z.x);
  sincos (z.y, &res.y, &res.x);
  res.x *= t;
  res.y *= t;
  return res;

}


  
  

/******************************
            Kernels
*******************************/



//This is our Romein-style scatter gridder.
__global__ void scatter_grid_kernel(struct vis_data *vis, struct w_kernel_data *wkern,
				  cuDoubleComplex *uvgrid, int max_support, int grid_size){



}


__global__ void fresnel_pattern_kernel(cuDoubleComplex *subimg, cuDoubleComplex *subgrid,
				       cuDoubleComplex *fresnel, int subgrid_size, int w_plane){



}
				       






/******************************
	  Host Functions
*******************************/
 
__host__ inline void fft_shift(cuDoubleComplex *uvgrid, int grid_size) {

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
  //For Benchmarking.
  
  cudaError_t error;
  cudaEvent_t start, stop;
  float elapsedTime;

  // Load visibility and w-kernel data from HDF5 files.
  
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


  //Allocate our main grid.
  
  int total_gs = grid_size * grid_size;
  
  cuDoubleComplex *grid_dev, *grid_host;
  cudaError_check(cudaMalloc((void **)&grid_dev, total_gs * sizeof(cuDoubleComplex)));
  cudaError_check(cudaMallocHost((void **)&grid_host, total_gs * sizeof(cuDoubleComplex)));

  int subgrid_mem_size = sizeof(cuDoubleComplex) * subgrid_size * subgrid_size;


  //Create the fresnel interference pattern for the W-Dimension
  //See Tim Cornwells paper on W-Projection for more information.
  
  cuDoubleComplex *wtransfer;
  cudaError_check(cudaMallocManaged((void **)&wtransfer, subgrid_mem_size, cudaMemAttachGlobal));

  int x,y;
  for (y=0; y < subgrid_size; ++y){

    for (x=0; x < subgrid_size; ++x){

      double l = theta * (double)(x - subgrid_size / 2) / subgrid_size;
      double m = theta * (double)(y - subgrid_size / 2) / subgrid_size;
      double ph = wincrement * (1 - sqrt(1 - l*l - m*m));

      cuDoubleComplex wtrans = make_cuDoubleComplex(0, 2 * M_PI * ph);
      wtransfer[y * subgrid_size + x] = cu_cexp_d(wtrans);
    }

  }

  //Create FFT Plans for our frequent fft's.

  cufftHandle fft_plan;
  cufftPlan2d(&fft_plan,subgrid_size,subgrid_size,CUFFT_D2Z);


  //Allocate subgrids/subimgs on the GPU
  
  assert( grid_size % subgrid_size == 0);
  int chunk_count_1d = grid_size / subgrid_size;
  int total_chunks = chunk_count_1d * chunk_count_1d;

  cuDoubleComplex **subgrids, **subimgs;

  cudaError_check(cudaMallocManaged(&subgrids, total_chunks * sizeof(cuDoubleComplex)));
  cudaError_check(cudaMallocManaged(&subimgs, total_chunks * sizeof(cuDoubleComplex)));

  //Create streams for each tower and allocate our chunks on GPU memory.
  
  cudaStream_t streams[total_chunks];
  for(int i = 0; i < total_chunks; ++i){

    cudaStreamCreate(&streams[i]);

    cudaError_check(cudaMallocManaged(subgrids + i, subgrid_mem_size * sizeof(cuDoubleComplex)));
    cudaError_check(cudaMallocManaged(subimgs + i, subgrid_mem_size * sizeof(cuDoubleComplex)));

  }

  

  
  return error;

}
