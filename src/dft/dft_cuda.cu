//C++ Includes
#include <iostream>
#include <fstream>
#include <cstdlib>

//CUDA Includes
#include <cuComplex.h>
#include "cuda.h"
#include "math.h"
#include "cuda_runtime_api.h"

//Our include
#include "dft_common.cuh"

__device__ cuDoubleComplex calculate_dft_sum_flat(struct flat_vis_data *vis, double l, double m){


  cuDoubleComplex grid_point = make_cuDoubleComplex(0.0, 0.0);
  for (int vi = 0; vi < vis->number_of_vis; ++vi){
    
    cuDoubleComplex visibility = *(cuDoubleComplex*)&vis->vis[vi];
    double u = vis->u[vi];
    double v = vis->v[vi];
    double w = vis->w[vi];

    double subang1 = l * u;
    double subang2 = m * v;
    double subang3 = (1-sqrt(1-l*l-m*m)) * w;

    double angle = 2 * M_PI * (subang1 + subang2 + subang3);

    double real_p = cuCreal(visibility) * cos(angle) + cuCimag(visibility) * sin(angle);
    double complex_p = -cuCreal(visibility) * sin(angle) + cuCimag(visibility) * cos(angle);

    grid_point = cuCadd(grid_point, make_cuDoubleComplex(real_p, complex_p));
  }
  return grid_point;

}

//Using unified memory instead of a deep copy.
// This calculates the DFT at a specific point on the grid.
// It adds to a local register and then does an atomic add to device memory.
__device__ cuDoubleComplex calculate_dft_sum(struct vis_data *vis, double l, double m){

  //nvcc should put this in a register.
  cuDoubleComplex grid_point = make_cuDoubleComplex(0.0,0.0);
  
  for (int bl = 0; bl < vis->bl_count; ++bl){

    for (int time = 0; time < vis->bl[bl].time_count; ++time){

      for (int freq = 0; freq < vis->bl[bl].freq_count; ++freq){

	double u = vis->bl[bl].uvw[time*vis->bl[bl].freq_count + freq];
	double v = vis->bl[bl].uvw[time*vis->bl[bl].freq_count + freq + 1];
	double w = vis->bl[bl].uvw[time*vis->bl[bl].freq_count + freq + 2];
        //Pointer cast because of mixing cuDoubleComplex and double _Complex.
	cuDoubleComplex visibility = *(cuDoubleComplex*)&vis->bl[bl].vis[time*vis->bl[bl].freq_count + freq];
	//nvcc should optimise this section.
	double subang1 = l * u;
	double subang2 = m * v;
	double subang3 = (1-sqrt(1-l*l-m*m)) * w;
	double angle = 2 * M_PI * (subang1 + subang2 + subang3);

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
__global__ void image_dft(struct vis_data *vis, cuDoubleComplex *uvgrid,
			  int grid_size, double lambda, double theta){

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  
  int y = idx / grid_size; //Typecast makes sure that we use the CUDA floor, not the C one.
  int x = idx % grid_size;

  double l = theta * (y - grid_size / 2) / grid_size;
  double m = theta * (x - grid_size / 2) / grid_size;
  
  uvgrid[idx] = calculate_dft_sum(vis, l, m);


}

//Executes a direct DFT from a given visibility dataset.
__global__ void image_dft_flat(struct flat_vis_data *vis, cuDoubleComplex *uvgrid,
			  int grid_size, double lambda){

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  
  int y = floor( (double)(idx / grid_size) ); //Typecast makes sure that we use the CUDA floor, not the C one.
  int x = idx % grid_size;

  double l = ((y - grid_size / 2)/lambda) * resolution;
  double m = ((x - grid_size / 2)/lambda) * resolution;
  
  uvgrid[idx] = calculate_dft_sum_flat(vis, l, m);


}




//This wraps the CUDA Kernel. Otherwise g++ doesn't recognise the <<< operator.
__host__ cudaError_t image_dft_host(const char* visfile, cuDoubleComplex *grid_host, cuDoubleComplex *grid_dev,
				    int grid_size, double theta,  double lambda, double bl_min, double bl_max,
				    int blocks, int threads_block){

  cudaError_t error = cudaSuccess;

  cudaEvent_t start, stop;
  float elapsedTime;

  struct vis_data *vis_dat;
  cudaError_check(cudaMallocManaged((void **)&vis_dat,sizeof(struct vis_data), cudaMemAttachGlobal));

  int viserr = load_vis_CUDA(visfile,vis_dat,bl_min,bl_max);

  if (viserr){
    std::cout << "Failed to Load Visibilities \n";
    return error; //Kill Program.
  }  

  //Declare our grid.
  //int grid_size = floor(lambda * theta);

  std::cout << "Theta: " << theta << "\n";
  std::cout << "Lambda: " << lambda << "\n";
  std::cout << "Grid Size: " << grid_size << " x " << grid_size << "\n";
  std::cout << "Grid Memory: " << (grid_size * grid_size * sizeof(double _Complex))/1e9 << "\n";


  
  std::cout<<"\n\n Executing Kernel \n";
  int total_gs = grid_size * grid_size;

  std::cout<<"Total Size: " << total_gs << "\n\n";
  

  cudaEventCreate(&start);
  cudaEventRecord(start, 0);
  image_dft <<< blocks , threads_block >>> (vis_dat, grid_dev, grid_size, lambda, theta);
  cudaEventCreate(&stop);
  cudaEventRecord(stop, 0);

  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);

  std::cout << "Elapsed Time: " << elapsedTime << "\n";
  std::cout << "Copying grid from device to host... \n";
  cudaError_check(cudaMemcpy(grid_host,grid_dev, grid_size * grid_size * sizeof(cuDoubleComplex),
			     cudaMemcpyDeviceToHost));
  return error;
}

//This wraps the CUDA Kernel. Otherwise g++ doesn't recognise the <<< operator.
__host__ cudaError_t image_dft_host_flat(const char* visfile, cuDoubleComplex *grid_host, cuDoubleComplex *grid_dev,
					 int grid_size, double theta,  double lambda, double bl_min, double bl_max,
					 int blocks, int threads_block){

  cudaError_t error = cudaSuccess;

  cudaEvent_t start, stop;
  float elapsedTime;

  struct vis_data *vis_dat = (struct vis_data*)malloc(sizeof(struct vis_data));
  struct flat_vis_data *flat_vis_dat;
  cudaError_check(cudaMallocManaged((void**)&flat_vis_dat, sizeof(struct flat_vis_data)));
  int viserr = load_vis(visfile,vis_dat,bl_min,bl_max);

  if (viserr){
    std::cout << "Failed to Load Visibilities \n";
    return error; //Kill Program.
  }  

  //Flatten Visibilities
  std::cout << "Flattening visibilities: \n";
  flatten_visibilities_CUDA(vis_dat, flat_vis_dat);
  
  //Declare our grid.
  //int grid_size = floor(lambda * theta);

  std::cout << "Theta: " << theta << "\n";
  std::cout << "Lambda: " << lambda << "\n";
  std::cout << "Grid Size: " << grid_size << " x " << grid_size << "\n";
  std::cout << "Grid Memory: " << (grid_size * grid_size * sizeof(double _Complex))/1e9 << "\n";


  
  std::cout<<"\n\n Executing Kernel \n";
  int total_gs = grid_size * grid_size;

  std::cout<<"Total Size: " << total_gs << "\n\n";
  
  cudaEventCreate(&start);
  cudaEventRecord(start, 0);
  image_dft_flat <<< blocks , threads_block >>> (flat_vis_dat, grid_dev, grid_size, lambda);
  cudaEventCreate(&stop);
  cudaEventRecord(stop, 0);

  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);

  std::cout << "Elapsed Time: " << elapsedTime << "\n";
  std::cout << "Copying grid from device to host... \n";
  cudaError_check(cudaMemcpy(grid_host,grid_dev, grid_size * grid_size * sizeof(cuDoubleComplex),
			     cudaMemcpyDeviceToHost));
  
  return error;
}
