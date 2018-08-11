//C++ Includes
#include <iostream>
#include <cassert>

//CUDA Includes
#include <cuComplex.h>
#include <cufft.h>
#include "cuda.h"
#include "math.h"
#include "cuda_runtime_api.h"
#include <thrust/random.h>

//Radio Gridders Includes
#include "radio.cuh"

// Saves me writing a CLI. Sorry Jayce.
#define GRID_SIZE 128 // UV Grid Size in 1-D
#define ANTENNAS 256 // Antennas to grid
#define ILLUM_X 3 // Aperture pattern extent. Assuming top hat,
#define ILLUM_Y 3 // '' for Y
#define NBATCH 1024
#define CHANNELS 4


// SoA format for mock fengine_data
struct fengine_data{
    double *x, *y, *z;
    cuComplex* fdata;
    int number_of_f_points;
};

/*****************************
        Device Functions
 *****************************/

//From Kyrills implementation in SKA/RC
__device__ void scatter_grid_add(cuComplex *uvgrid,
				 int grid_size,
				 int grid_pitch,
				 int grid_point_u,
				 int grid_point_v,
				 cuComplex sum){

    if (grid_point_u < 0 || grid_point_u >= grid_size ||
      grid_point_v < 0 || grid_point_v >= grid_size)
    return;

    // Add to grid. This is the bottleneck of the entire kernel
    atomicAdd(&uvgrid[grid_point_u + grid_pitch*grid_point_v].x, sum.x); // Re
    atomicAdd(&uvgrid[grid_point_u + grid_pitch*grid_point_v].y, sum.y); // Im
}

//Scatters grid points from a non-hierarchical dataset.
//Advantage: Locality is almost certainly better for fragmented datasets.
//Disadvantage: Not able to do baseline specific calibration, such as ionosphere correction.
// Most of this stuff doesn't matter for EPIC yet... someone elses problem mwahaha.
#ifdef __COUNT_VIS__
__device__ void scatter_grid_point(struct fengine_data* fourierdata, // Our bins of UV Data
				   cuComplex* uvgrid, // Our main UV Grid
				   cuComplex* illum, //Our W-Kernel
				   int max_supp, // Max size of W-Kernel
				   int myU, //Our assigned u/v points.
				   int myV, // ^^^
				   int grid_size, //The size of our w-towers subgrid.
				   unsigned long long int *visc_reg){ 
#else
__device__ void scatter_grid_point(struct fengine_data* fourierdata, // Our fourier values
				   cuComplex* uvgrid, // Our main UV Grid
				   cuComplex* illum, //Our Illumination-Kernel
				   int max_supp, // Max size of W-Kernel
				   int myU, //Our assigned u/v points.
				   int myV, // ^^^
				   int grid_size){ 
#endif
  
  int grid_point_u = myU, grid_point_v = myV;
  cuComplex sum  = make_cuComplex(0.0,0.0);
  short supp = ILLUM_X;
  int vi = 0;
  
  for (vi = 0; vi < fourierdata->number_of_f_points; ++vi){

    int u = fourierdata->x[vi]; 
    int v = fourierdata->y[vi];

    // Determine convolution point. This is basically just an
    // optimised way to calculate.
    //int myConvU = myU - u;
    //int myConvV = myV - v;
    int myConvU = (u - myU) % max_supp;
    int myConvV = (v - myV) % max_supp;    
    if (myConvU < 0) myConvU += max_supp;
    if (myConvV < 0) myConvV += max_supp;

    // Determine grid point. Because of the above we know here that
    //   myGridU % max_supp = myU
    //   myGridV % max_supp = myV
    int myGridU = u + myConvU
      , myGridV = v + myConvV;

    // Grid point changed?
    if (myGridU != grid_point_u || myGridV != grid_point_v) {
      // Atomically add to grid. This is the bottleneck of this kernel.
      scatter_grid_add(uvgrid, grid_size, grid_size, grid_point_u, grid_point_v, sum);
      // Switch to new point
      sum = make_cuComplex(0.0, 0.0);
      grid_point_u = myGridU;
      grid_point_v = myGridV;
    }
    //TODO: Re-do the w-kernel/gcf for our data.
    //	cuDoubleComplex px;
    cuComplex px = illum[myConvV * supp + myConvU];// ??
    //cuComplex px = *(cuComplex*)&wkern->kern_by_w[w_plane].data[sub_offset + myConvV * supp + myConvU];	
    // Sum up
    cuComplex vi_v = fourierdata->fdata[vi];
    sum = cuCfmaf(cuConjf(px), vi_v, sum);


  }
  // Add remaining sum to grid
  #ifdef __COUNT_VIS__
  atomicAdd(visc_reg,vi);
  #endif
  scatter_grid_add(uvgrid, grid_size, grid_size, grid_point_u, grid_point_v, sum);
}



/*******************
   Romein Kernel
 ******************/
 
#ifdef __COUNT_VIS__
__global__ void scatter_grid_kernel(struct fengine_data* vis, // Mock F-Engine Data
				    cuComplex* illum, // Illumination Pattern
				    cuComplex* uvgrid, //Our UV-Grid
				    int max_support, //  Convolution size
				    int grid_size, // Subgrid size
				    unsigned long long int* visc_reg){
#else
__global__ void scatter_grid_kernel(struct fengine_data* vis, // No. of visibilities
				      cuComplex* illum, // Illumination Pattern
				      cuComplex* uvgrid, //Our UV-Grid
				      int max_support, //  Convolution size
				      int grid_size){
				
#endif
  //Assign some visibilities to grid;

      for(int i = threadIdx.x; i < max_support * max_support; i += blockDim.x){
	  //  int i = threadIdx.x + blockIdx.x * blockDim.x;
	  int myU = i % max_support;
	  int myV = i / max_support;
    
#ifdef __COUNT_VIS__
	  scatter_grid_point(vis, illum, uvgrid, max_support, myU, myV, grid_size, visc_reg);
#else
	  scatter_grid_point(vis, illum, uvgrid, max_support, myU, myV, grid_size);
#endif
		       
  }
}
__host__ cudaError_t epic_romein(struct fengine_data* fourierData,
				 cuComplex* illum,
				 cuComplex* uvgrid,
				 int support_size,
				 int grid_size){

    // Launch a batch of Romein-Kernels.
    for (int i = 0; i < NBATCH; i++){



    }
	  
    return cudaSuccess;
}

int main(){


    //Get information on GPU's in system.
    std::cout << "CUDA System Information: \n\n";
    int numberofgpus;
    
    
    cudaGetDeviceCount(&numberofgpus);
    std::cout << " Number of GPUs Detected: " << numberofgpus << "\n\n";
    
    cudaDeviceProp *prop = (cudaDeviceProp*)malloc(sizeof(cudaDeviceProp) * numberofgpus);
    
    for(int i=0; i<numberofgpus;i++){
	
	
	cudaGetDeviceProperties(&prop[i],i);
	
	std::cout << "\tDevice Number: " << i <<" \n";
	std::cout << "\t\tDevice Name: " << prop->name <<"\n";
	std::cout << "\t\tTotal Memory: " << (double)prop->totalGlobalMem / (1024 * 1024) << " MB \n";
	std::cout << "\t\tShared Memory (per block): " << (double)prop->sharedMemPerBlock / 1024 << " kB \n";
	std::cout << "\t\tClock Rate: " << prop->clockRate << "\n";
	std::cout << "\t\tMultiprocessors: " << prop->multiProcessorCount << "\n";
	std::cout << "\t\tThreads Per MP: " << prop->maxThreadsPerMultiProcessor << "\n";
	std::cout << "\t\tThreads Per Block: " << prop->maxThreadsPerBlock << "\n";
	std::cout << "\t\tThreads Per Dim: " << prop->maxThreadsDim << "\n";
	std::cout << "\t\tThreads Per Warp: " << prop->warpSize << "\n";
	std::cout << "\t\tUnified Addressing: " << prop->unifiedAddressing << "\n";
	std::cout << "\n";	
    }
    
    cuComplex* uvgrid; // Our U-V Grid
    cuComplex* invec; // Vector of fourier values.
    cuComplex* illumination; //Our illumination pattern
    
    int gsize = GRID_SIZE * GRID_SIZE * NBATCH * sizeof(cuComplex);
    int vecsize = ANTENNAS * NBATCH * sizeof(cuComplex);
    int illumsize = ILLUM_X * ILLUM_Y * sizeof(cuComplex);
    
    // Mallocs
    cudaError_check(cudaMallocManaged((void **)&uvgrid, gsize,cudaMemAttachGlobal));
    cudaError_check(cudaMallocManaged((void **)&invec, vecsize, cudaMemAttachGlobal));
    cudaError_check(cudaMallocManaged((void **)&illumination, illumsize, cudaMemAttachGlobal));

    // Initialise input vector with random fourier values.
    thrust::default_random_engine generator;
    thrust::normal_distribution<float> distribution(0.0,5.0);
   
    for (int i = 0; i < ANTENNAS * NBATCH; ++i){
	invec[i] = make_cuComplex(distribution(generator),distribution(generator));
    }

    // Initialise illumination pattern. Square top-hat function.
    for (int i = 0; i < ILLUM_X * ILLUM_Y; ++i) illumination[i] = make_cuComplex(1.0,0.0);

    // Initialise and run CUDA.
    //cudaError_check(epic_romein(invec, illumination, uvgrid, ILLUM_X, GRID_SIZE));

}