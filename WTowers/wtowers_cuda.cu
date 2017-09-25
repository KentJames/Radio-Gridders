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



//W-Towers Wrapper.
__host__ cudaError_t wtowers_host(const char* visfile, const char* wkernfile, int grid_size,
			   double theta,  double lambda, double bl_min, double bl_max,
				  int iter){

  cudaError_t error;
  
  return error;

}
