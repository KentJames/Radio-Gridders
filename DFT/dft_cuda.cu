//C++ Includes
#include <iostream>
#include <cstdlib>

//CUDA Includes
#include <cuComplex.h>
#include <cufft.h>
#include "cuda.h"
#include "math.h"
#include "cuda_runtime_api.h"

//Our include
#include "dft_common.h"


//Using unified memory instead of a deep copy.

// This calculates the DFT at a specific point on the grid.
// It adds to a local register and then does an atomic add to device memory.
__device__ cuDoubleComplex calculate_dft_sum(struct vis_data *vis, double l, double m){

  //nvcc should put this in a register.
  cuDoubleComplex grid_point = make_cuDoubleComplex(0.0,0.0);
  
  for (int bl = 0; bl < vis->bl_count; ++bl){

    for (int time = 0; time < vis->bl[bl].time_count; ++time){

      for (int freq = 0; freq < vis->bl[bl].freq_count; ++freq){

	//This step is quite convoluted due to mixing C and CUDA compelx datatypes..
	cuDoubleComplex visibility;
	double __complex__ visibility_c = vis->bl[bl].vis[time*vis->bl[bl].freq_count + freq];
	memcpy(&visibility, &visibility_c, sizeof(double __complex__));
	

	//nvcc should optimise this section.
	double subang1 = m * vis->bl[bl].uvw[time*vis->bl[bl].freq_count + freq];
	double subang2 = l * vis->bl[bl].uvw[time*vis->bl[bl].freq_count + freq + 1];
	double subang3 = (sqrtf(1-l*l-m*m)-1) * vis->bl[bl].uvw[time*vis->bl[bl].freq_count +
								freq + 2];

	double angle = 2 * M_PI * subang1 + subang2 + subang3;

	double real_p = cuCreal(visibility) * cos(angle) + cuCimag(visibility) * sin(angle);
	double complex_p = -cuCreal(visibility) * sin(angle) + cuCimag(visibility) * cos(angle);

	//Add these to our grid_point so far.
	grid_point = cuCadd(grid_point, make_cuDoubleComplex(real_p, complex_p));
							       

      }
    }
  }
  

  return grid_point; //Placeholder
}


//Executes a direct DFT from a given visibility dataset.
__global__ void image_dft(struct vis_data *vis, cuDoubleComplex *uvgrid, int grid_size,
			  double lambda, int iter, int N){

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  
  int y = floor( (double)(idx / grid_size) ); //Typecast makes sure that we use the CUDA floor, not the C one.
  int x = idx % grid_size;

  double l = (y - grid_size / 2)/lambda;
  double m = (x - grid_size / 2)/lambda;
  
  uvgrid[idx] = calculate_dft_sum(vis, l, m);


}


//This wraps the CUDA Kernel. Otherwise g++ doesn't recognise the <<< operator.
__host__ void image_dft_host(const char* visfile, int grid_size,
		    double theta,  double lambda, double bl_min, double bl_max,
		    int iter){

  cudaError_t error;

  cudaEvent_t start, stop;
  float elapsedTime;
  
  //  error = cudaMallocManaged(reinterpret_cast<void **>(&vis_dat),sizeof(struct vis_data), cudaMemAttachGlobal);
  struct vis_data vis_dat;
  int viserr = load_vis(visfile,&vis_dat,bl_min,bl_max);

  if (viserr){
    std::cout << "Failed to Load Visibilities \n";
    return; //Kill Program.
  }

  // Now to get visibilities to the device.

  struct vis_data vis_dat_gpu;

  //Declare our grid.
  //int grid_size = floor(lambda * theta);

  std::cout << "Theta: " << theta << "\n";
  std::cout << "Lambda: " << lambda << "\n";
  std::cout << "Grid Size: " << grid_size << " x " << grid_size << "\n";
  std::cout << "Grid Memory: " << (grid_size * grid_size * sizeof(double _Complex))/1e9 << "\n";


  
  std::cout<<"\n\n Executing Kernel \n";
  int total_gs = grid_size * grid_size;

  std::cout<<"Total Size: " << total_gs << "\n\n";
  

  cuDoubleComplex *grid_dev;
  error = cudaMalloc((void **)&grid_dev, grid_size * grid_size * sizeof(cuDoubleComplex));
  if (error == cudaSuccess){

    cudaEventCreate(&start);
    cudaEventRecord(start, 0);
    image_dft <<< 4096,1024>>> (&vis_dat, grid_dev, grid_size, lambda, iter, total_gs);
    cudaEventCreate(&stop);
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    std::cout << "Elapsed Time: " << elapsedTime << "\n";
  }
  else {

    std::cout << "Memory Allocation Failed. \n";

  }


  std::cout << "DFT Value: " << cuCreal(grid_dev[500]);
  

  //Check it actually ran...
  cudaError_t err = cudaGetLastError();

  std::cout << "Error: " << cudaGetErrorString(err) << "\n";
  

}