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
#include "grid_common.cuh"
#include "common_kernels.cuh"



/*****************************
        Device Functions
 *****************************/

//From Kyrills implementation in SKA/RC
__device__ void scatter_grid_add(cuDoubleComplex *uvgrid, int grid_size, int grid_pitch,
					int grid_point_u, int grid_point_v, cuDoubleComplex sum){

  // Atomically add to grid. This is the bottleneck of this kernel.
  if (grid_point_u < 0 || grid_point_u >= grid_size ||
      grid_point_v < 0 || grid_point_v >= grid_size)
    return;

  // Add to grid. This is the bottleneck of the entire kernel
  atomicAdd(&uvgrid[grid_point_u + grid_pitch*grid_point_v].x, sum.x);
  atomicAdd(&uvgrid[grid_point_u + grid_pitch*grid_point_v].y, sum.y);
}

//Scatters grid points from a non-hierarchical dataset.
//Advantage: Locality is almost certainly better for fragmented datasets.
//Disadvantage: Not able to do baseline specific calibration, such as ionosphere correction.
#ifdef __COUNT_VIS__
__device__ void scatter_grid_point_flat(struct flat_vis_data *vis, // Our bins of UV Data
					cuDoubleComplex *uvgrid, // Our main UV Grid
					struct w_kernel_data *wkern, //Our W-Kernel
					int max_supp, // Max size of W-Kernel
					int myU, //Our assigned u/v points.
					int myV, // ^^^
					double wstep, // W-Increment 
					int subgrid_size, //The size of our w-towers subgrid.
					double theta, // Field of View Size
					double offset_u, // Offset from top left of main grid to t.l of subgrid.
					double offset_v, // ^^^^
					double offset_w,
					unsigned long long int *visc_reg){ 
#else
__device__ void scatter_grid_point_flat(struct flat_vis_data *vis, // Our bins of UV Data
					cuDoubleComplex *uvgrid, // Our main UV Grid
					struct w_kernel_data *wkern, //Our W-Kernel
					int max_supp, // Max size of W-Kernel
					int myU, //Our assigned u/v points.
					int myV, // ^^^
					double wstep, // W-Increment 
					int subgrid_size, //The size of our w-towers subgrid.
					double theta, // Field of View Size
					double offset_u, // Offset from top left of main grid to t.l of subgrid.
					double offset_v, // ^^^^
					double offset_w){ 
#endif
  
  int grid_point_u = myU, grid_point_v = myV;
  cuDoubleComplex sum  = make_cuDoubleComplex(0.0,0.0);
  int supp = wkern->size_x;
  int vi = 0;
  
  for (vi = 0; vi < vis->number_of_vis; ++vi){

    double w = vis->w[vi] - offset_w;
    int w_plane = floor((w - wkern->w_min) / wkern->w_step + .5);
    int grid_offset, sub_offset;
    frac_coord_flat(subgrid_size, wkern->size_x, wkern->oversampling,
		    theta, vis, vi, offset_u, offset_v, &grid_offset, &sub_offset);
    int u = grid_offset % subgrid_size; 
    int v = grid_offset / subgrid_size;

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
      scatter_grid_add(uvgrid, subgrid_size, subgrid_size, grid_point_u, grid_point_v, sum);
      // Switch to new point
      sum = make_cuDoubleComplex(0.0, 0.0);
      grid_point_u = myGridU;
      grid_point_v = myGridV;
    }
    //TODO: Re-do the w-kernel/gcf for our data.
    //	cuDoubleComplex px;
    cuDoubleComplex px = *(cuDoubleComplex*)&wkern->kern_by_w[w_plane].data[sub_offset + myConvV * supp + myConvU];	
    // Sum up
    cuDoubleComplex vi_v = *(cuDoubleComplex*)&vis->vis[vi];
    sum = cuCfma(cuConj(px), vi_v, sum);


  }
  // Add remaining sum to grid
  #ifdef __COUNT_VIS__
  atomicAdd(visc_reg,vi);
  #endif
  scatter_grid_add(uvgrid, subgrid_size, subgrid_size, grid_point_u, grid_point_v, sum);
}



//From Kyrills Implementation in SKA/RC. Modified to suit our data format.
//Assumes pre-binned (in u/v) data
#ifdef __COUNT_VIS__
__device__ void scatter_grid_point(struct vis_data *bin, // Our bins of UV Data
				   cuDoubleComplex *uvgrid, // Our main UV Grid
				   struct w_kernel_data *wkern, //Our W-Kernel
				   int max_supp, // Max size of W-Kernel
				   int myU, //Our assigned u/v points.
				   int myV, // ^^^
				   double wstep, // W-Increment 
				   int subgrid_size, //The size of our w-towers subgrid.
				   double theta, // Field of View Size
				   double offset_u, // Offset from top left of main grid to t.l of subgrid.
				   double offset_v, // ^^^^
				   double offset_w,
				   unsigned long long int *visc_reg){
#else
__device__ void scatter_grid_point(struct vis_data *bin, // Our bins of UV Data
				   cuDoubleComplex *uvgrid, // Our main UV Grid
				   struct w_kernel_data *wkern, //Our W-Kernel
				   int max_supp, // Max size of W-Kernel
				   int myU, //Our assigned u/v points.
				   int myV, // ^^^
				   double wstep, // W-Increment 
				   int subgrid_size, //The size of our w-towers subgrid.
				   double theta, // Field of View Size
				   double offset_u, // Offset from top left of main grid to t.l of subgrid.
				   double offset_v, // ^^^^
				   double offset_w){
#endif

  int grid_point_u = myU, grid_point_v = myV;
  cuDoubleComplex sum  = make_cuDoubleComplex(0.0,0.0);

  short supp = short(wkern->size_x);
  int vi = 0;
  //  for (int i = 0; i < visibilities; i++) {
  int bl, time, freq;
  for (bl = 0; bl < bin->bl_count; ++bl){

    struct bl_data *bl_d = &bin->bl[bl];    
    for (time = 0; time < bl_d->time_count; ++time){
      for(freq = 0; freq < bl_d->freq_count; ++freq){
	++vi;
	// Load pre-calculated positions
	//int u = uvo[i].u, v = uvo[i].v;
	//	int u = (int)uvw_lambda(bl_d, time, freq, 0);
	//int v = (int)uvw_lambda(bl_d, time, freq, 1);
	double w = uvw_lambda(bl_d, time, freq, 2) - offset_w;
	int w_plane = floor((w - wkern->w_min) / wkern->w_step + .5);
	int grid_offset, sub_offset;
	frac_coord(subgrid_size, wkern->size_x, wkern->oversampling,
		   theta, bl_d, time, freq, offset_u, offset_v, &grid_offset, &sub_offset);
	int u = grid_offset % subgrid_size; 
	int v = grid_offset / subgrid_size;

	// Determine convolution point. This is basically just an
	// optimised way to calculate
	//   myConvU = (myU - u) % max_supp
	//   myConvV = (myV - v) % max_supp
	//	int2 xy = getcoords_xy(u,v,subgrid_size,theta,max_supp);
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
	  scatter_grid_add(uvgrid, subgrid_size, subgrid_size, grid_point_u, grid_point_v, sum);
	  // Switch to new point
	  sum = make_cuDoubleComplex(0.0, 0.0);
	  grid_point_u = myGridU;
	  grid_point_v = myGridV;
	}
	//TODO: Re-do the w-kernel/gcf for our data.
	//	cuDoubleComplex px;
	cuDoubleComplex px = *(cuDoubleComplex*)&wkern->kern_by_w[w_plane].data[sub_offset + myConvV * supp + myConvU];	
	// Sum up.
	cuDoubleComplex vi = *(cuDoubleComplex*)&bl_d->vis[time*bl_d->freq_count+freq];
	sum = cuCfma(cuConj(px), vi, sum);
      }
    }
  }
  #ifdef __COUNT_VIS__
  atomicAdd(visc_reg,vi);
  #endif
  // Add remaining sum to grid
  scatter_grid_add(uvgrid, subgrid_size, subgrid_size, grid_point_u, grid_point_v, sum);
}


/******************************
            Kernels
*******************************/




//This is our Romein-style scatter gridder. Works on flat visibility data.
#ifdef __COUNT_VIS__
__global__ void scatter_grid_kernel_flat(struct flat_vis_data *vis, // No. of visibilities
					 struct w_kernel_data *wkern, // No. of wkernels
					 cuDoubleComplex *uvgrid, //Our UV-Grid
					 int max_support, //  Convolution size
					 int subgrid_size, // Subgrid size
					 double wstep, // W-Increment
					 double theta, // Field of View
					 double offset_u, // Top left offset from top left main grid
					 double offset_v, // ^^^^
					 double offset_w,
					 unsigned long long int *visc_reg){
#else
  __global__ void scatter_grid_kernel_flat(struct flat_vis_data *vis, // No. of visibilities
					 struct w_kernel_data *wkern, // No. of wkernels
					 cuDoubleComplex *uvgrid, //Our UV-Grid
					 int max_support, //  Convolution size
					 int subgrid_size, // Subgrid size
					 double wstep, // W-Increment
					 double theta, // Field of View
					 double offset_u, // Top left offset from top left main grid
					 double offset_v, // ^^^^
					 double offset_w){
#endif
  //Assign some visibilities to grid;
  
  for(int i = threadIdx.x; i < max_support * max_support; i += blockDim.x){
    //  int i = threadIdx.x + blockIdx.x * blockDim.x;
    int myU = i % max_support;
    int myV = i / max_support;
    
    #ifdef __COUNT_VIS__
    scatter_grid_point_flat(vis+blockIdx.x, uvgrid, wkern, max_support, myU, myV, wstep,
			    subgrid_size, theta, offset_u, offset_v, offset_w, visc_reg);
    #else
        scatter_grid_point_flat(vis+blockIdx.x, uvgrid, wkern, max_support, myU, myV, wstep,
			    subgrid_size, theta, offset_u, offset_v, offset_w);
    #endif
		       
  }
}


//This is our Romein-style scatter gridder. Works on hierarchical visibility data (bl->time->freq).
#ifdef __COUNT_VIS__
__global__ void scatter_grid_kernel(struct vis_data *bin, // Baseline bin
				    struct w_kernel_data *wkern, // No. of wkernels
				    cuDoubleComplex *uvgrid, //Our UV-Grid
				    int max_support, //  Convolution size
				    int subgrid_size, // Subgrid size
				    double wstep, // W-Increment
				    double theta, // Field of View
				    double offset_u, // Top left offset from top left main grid
				    double offset_v, // ^^^^
				    double offset_w,
				    unsigned long long int *visc_reg){
#else
  __global__ void scatter_grid_kernel(struct vis_data *bin, // Baseline bin
				    struct w_kernel_data *wkern, // No. of wkernels
				    cuDoubleComplex *uvgrid, //Our UV-Grid
				    int max_support, //  Convolution size
				    int subgrid_size, // Subgrid size
				    double wstep, // W-Increment
				    double theta, // Field of View
				    double offset_u, // Top left offset from top left main grid
				    double offset_v, // ^^^^
				    double offset_w){
#endif
  
  for(int i = threadIdx.x; i < max_support * max_support; i += blockDim.x){
    //  int i = threadIdx.x + blockIdx.x * blockDim.x;
    int myU = i % max_support;
    int myV = i / max_support;
    #ifdef __COUNT_VIS__
    scatter_grid_point(bin, uvgrid, wkern, max_support, myU, myV, wstep,
		       subgrid_size, theta, offset_u, offset_v, offset_w, visc_reg);
    #else
    scatter_grid_point(bin, uvgrid, wkern, max_support, myU, myV, wstep,
		       subgrid_size, theta, offset_u, offset_v, offset_w);
    #endif
  }
}

/******************************
	  Host Functions
*******************************/

//W-Towers Wrapper.
__host__ cudaError_t wtowers_CUDA(const char* visfile, const char* wkernfile,
				  cuDoubleComplex *grid, cuDoubleComplex *gridh, int grid_size,
				  double theta,  double lambda, double bl_min, double bl_max,
				  int subgrid_size, int subgrid_margin, double wincrement){

  //API Variables
  cudaError_t error = (cudaError_t)0;
  //For Benchmarking.
  cudaEvent_t start, stop;
  float elapsedTime;
  size_t free_bytes;
  size_t total_bytes;
  
  #ifdef __COUNT_VIS__
  unsigned long long int *vis_count;
  cudaError_check(cudaMallocManaged((void **)&vis_count, sizeof(unsigned long long int)));
  #endif
  
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

  // Work out our minimum and maximum w-planes.
  double vis_w_min = 0, vis_w_max = 0;
  for (int bl = 0; bl < vis_dat->bl_count; ++bl){
    double w_min = lambda_min(&vis_dat->bl[bl], vis_dat->bl[bl].w_min);
    double w_max = lambda_max(&vis_dat->bl[bl], vis_dat->bl[bl].w_max);
    if (w_min < vis_w_min) { vis_w_min = w_min; }
    if (w_max > vis_w_max) { vis_w_max = w_max; }
  }
  int wp_min = (int) floor(vis_w_min / wincrement + 0.5);
  int wp_max = (int) floor(vis_w_max / wincrement + 0.5);
  std::cout << "Our W-Plane Min/Max: " << wp_min << " " << wp_max << "\n";

  //Create the fresnel interference pattern for the W-Dimension
  //Can make this a kernel.
  //See Tim Cornwells paper on W-Projection for more information.
  int subgrid_mem_size = sizeof(cuDoubleComplex) * subgrid_size * subgrid_size;  
  cuDoubleComplex *wtransfer;
  cudaError_check(cudaMallocManaged((void **)&wtransfer, subgrid_mem_size, cudaMemAttachGlobal));

  int x,y;
  for (y=0; y < subgrid_size; ++y){
    for (x=0; x < subgrid_size; ++x){
      double l = theta * (double)(x - subgrid_size / 2) / subgrid_size;
      double m = theta * (double)(y - subgrid_size / 2) / subgrid_size;
      double ph = wincrement * (1 - sqrt(1 - l*l - m*m));
      cuDoubleComplex wtrans = make_cuDoubleComplex(0.0, 2 * M_PI * ph);
      wtransfer[y * subgrid_size + x] = cu_cexp_d(wtrans);
    }
  }
  fft_shift(wtransfer, subgrid_size);

  //Allocate subgrids/subimgs on the GPU
  
  assert( grid_size % subgrid_size == 0);
  
  int chunk_size = subgrid_size - subgrid_margin;
  int chunk_count_1d = grid_size / chunk_size + 1;
  int total_chunks = chunk_count_1d * chunk_count_1d;

  //Allocate all our subgrids/subimgs contiguously.
  cuDoubleComplex *subgrids, *subimgs;
  
  cudaError_check(cudaMallocManaged((void **)&subgrids,
				    total_chunks * subgrid_mem_size  * sizeof(cuDoubleComplex),
				    cudaMemAttachGlobal));
  cudaError_check(cudaMallocManaged((void **)&subimgs,
				    total_chunks * subgrid_mem_size * sizeof(cuDoubleComplex),
				    cudaMemAttachGlobal));

  //Create streams for each tower and allocate our chunks in unified memory.
  //Also set our FFT plans while we are here.
  //Initialise cublas handle. We use cublas to multiply our fresnel phase screen.

  cudaStream_t *streams = (cudaStream_t *) malloc(total_chunks * sizeof(cudaStream_t));
  cufftHandle *subgrid_plans = (cufftHandle *) malloc(total_chunks * sizeof(cufftHandle));
  
  for(int i = 0; i < total_chunks; ++i){

    //Create stream.
    cudaError_check(cudaStreamCreate(&streams[i]));

    //Assign FFT Plan to each stream
    cuFFTError_check(cufftPlan2d(&subgrid_plans[i],subgrid_size,subgrid_size,CUFFT_Z2Z));
    cuFFTError_check(cufftSetStream(subgrid_plans[i], streams[i]));
  }

  //Our FFT plan for our final transform.
  cufftHandle grid_plan;
  cuFFTError_check(cufftPlan2d(&grid_plan, grid_size, grid_size, CUFFT_Z2Z));

  //Allocate our bins and Bin in U/V
  struct vis_data *bins;

  cudaError_check(cudaMallocManaged(&bins, total_chunks * sizeof(struct vis_data), cudaMemAttachGlobal));
  bin_visibilities(vis_dat, bins, chunk_count_1d, wincrement, theta, grid_size, chunk_size, &wp_min, &wp_max);
  
  
  //Record Start
  cudaEventCreate(&start);
  cudaEventRecord(start,0);
  
  int fft_gs = 32;
  int fft_bs = subgrid_size/fft_gs;
  dim3 dimGrid(fft_bs,fft_bs);
  dim3 dimBlock(fft_gs,fft_gs);
  dim3 dimBlock_main(16,16);
  dim3 dimGrid_main(128,128);

  int last_wp = wp_min;
  // Lets get gridding!

  int wkern_size = wkern_dat->size_x;
  int wkern_wstep = wkern_dat->w_step;

  for(int chunk =0; chunk < total_chunks; ++chunk){

    int subgrid_offset = chunk * subgrid_size * subgrid_size;
    //std::cout << "Launching kernels for chunk: " << chunk << wp_min << wp_max <<"\n";
    int cx = chunk % chunk_count_1d;
    int cy = floor(chunk / chunk_count_1d);

    int x_min = cx * chunk_size - grid_size/2;
    int y_min = cy * chunk_size - grid_size/2;

    double u_mid = (double)(x_min + chunk_size / 2) / theta;
    double v_mid = (double)(y_min + chunk_size / 2) / theta;

    cudaError_check(cudaMemsetAsync(subgrids+subgrid_offset, 0, subgrid_mem_size, streams[chunk]));
    cudaError_check(cudaMemsetAsync(subimgs+subgrid_offset, 0, subgrid_mem_size, streams[chunk]));
    
    for(int wp = wp_min; wp<=wp_max; ++wp){
      //std::cout << "WP: " << wp << "\n";
      double w_mid = (double)wp * wincrement;
      #ifdef __COUNT_VIS__
      scatter_grid_kernel <<< 1, 64, 0, streams[chunk] >>>
	(vis_dat, wkern_dat, subgrids+subgrid_offset, wkern_size,
	 subgrid_size, wkern_wstep, theta,
	 u_mid, v_mid, w_mid,vis_count);
      #else
      scatter_grid_kernel <<< 1, 64, 0, streams[chunk] >>>
	(vis_dat, wkern_dat, subgrids+subgrid_offset, wkern_size,
	 subgrid_size, wkern_wstep, theta,
	 u_mid, v_mid, w_mid);
      #endif
      cuFFTError_check(cufftExecZ2Z(subgrid_plans[chunk], subgrids+subgrid_offset, subgrids+subgrid_offset, CUFFT_INVERSE));
      fresnel_subimg_mul <<< dimGrid, dimBlock, 0, streams[chunk] >>> (subgrids+subgrid_offset, wtransfer, subimgs+subgrid_offset, subgrid_size, wp - last_wp);
      cudaError_check(cudaMemsetAsync(subgrids+subgrid_offset, 0.0, subgrid_mem_size, streams[chunk]));
      last_wp = wp;
    }
       
    w0_transfer_kernel <<< dimGrid , dimBlock, 0, streams[chunk] >>> (subimgs+subgrid_offset, wtransfer, last_wp, subgrid_size);
    cuFFTError_check(cufftExecZ2Z(subgrid_plans[chunk], subimgs+subgrid_offset, subimgs+subgrid_offset, CUFFT_FORWARD)); //This can be sped up with a batched strided FFT. Future optimisation...
    

  }
  cudaError_check(cudaDeviceSynchronize());
  add_subs2main_kernel <<< dimGrid_main, dimBlock_main >>> (grid, subimgs, grid_size, subgrid_size,
  					  subgrid_margin, chunk_count_1d, chunk_size);  
  
  fft_shift_kernel <<< dimGrid_main, dimBlock_main >>> (grid, grid_size);
  cuFFTError_check(cufftExecZ2Z(grid_plan, grid, grid, CUFFT_INVERSE));

  fft_shift_kernel <<< dimGrid_main, dimBlock_main >>> (grid, grid_size);
  

  cudaEventCreate(&stop);
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime,start,stop);
  cudaError_check(cudaMemGetInfo(&free_bytes, &total_bytes));
  std::cout << "Memory Usage:: Free: " << free_bytes << " Total: " << total_bytes << "\n";
  std::cout << "Scatter Gridder Elapsed Time: " << elapsedTime/1000.0 << " seconds\n";
  return error;
   
}

//W-Towers Wrapper.
__host__ cudaError_t wtowers_CUDA_flat(const char* visfile, const char* wkernfile,
				       cuDoubleComplex *grid, cuDoubleComplex *gridh, int grid_size,
				       double theta,  double lambda, double bl_min, double bl_max,
				       int subgrid_size, int subgrid_margin, double wincrement){

  #ifdef __COUNT_VIS__
  unsigned long long int *vis_count;
  cudaError_check(cudaMallocManaged((void **)&vis_count, sizeof(unsigned long long int)));
  cudaError_check(cudaMemset(vis_count, 0, sizeof(unsigned long long int)));
  #endif
  
  //API Variables.
  cudaError_t error = (cudaError_t)0;
  //Subgrid arithmetic.
  assert( grid_size % subgrid_size == 0);  
  int chunk_size = subgrid_size - subgrid_margin;
  int chunk_count_1d = grid_size / chunk_size + 1;
  int total_chunks = chunk_count_1d * chunk_count_1d;
  int wp_min, wp_max;   
  //Allocate our main grid.
  
  int total_gs = grid_size * grid_size;
  
  //For Benchmarking.
  cudaEvent_t start, stop;
  float elapsedTime;
  size_t free_bytes;
  size_t total_bytes;

  // Load visibility and w-kernel data from HDF5 files.
  struct vis_data *vis_dat;
  struct flat_vis_data *flat_vis_dat, *vis_bins, *vis_bins_w;
  struct w_kernel_data *wkern_dat;

  cudaError_check(cudaMallocManaged((void **)&vis_dat, sizeof(struct vis_data), cudaMemAttachGlobal));
  cudaError_check(cudaMallocManaged((void **)&wkern_dat, sizeof(struct w_kernel_data), cudaMemAttachGlobal));

  int error_hdf5;
  error_hdf5 = load_vis(visfile,vis_dat,bl_min,bl_max);
  if (error_hdf5) {
    std::cout << "Failed to Load Visibilities \n";
    return error;
  }
  error_hdf5 = load_wkern_CUDA(wkernfile, theta, wkern_dat);
  if (error_hdf5) {
    std::cout << "Failed to Load W-Kernels \n";
    return error;
  }

  double vis_w_min = 0, vis_w_max = 0;
  for (int bl = 0; bl < vis_dat->bl_count; ++bl){
    double w_min = lambda_min(&vis_dat->bl[bl], vis_dat->bl[bl].w_min);
    double w_max = lambda_max(&vis_dat->bl[bl], vis_dat->bl[bl].w_max);
    if (w_min < vis_w_min) { vis_w_min = w_min; }
    if (w_max > vis_w_max) { vis_w_max = w_max; }
  }
  wp_min = (int) floor(vis_w_min / wincrement + 0.5);
  wp_max = (int) floor(vis_w_max / wincrement + 0.5);
  std::cout << "Our W-Plane Min/Max: " << wp_min << " " << wp_max << "\n";
  int wp_tot = abs(wp_min - wp_max) + 1;
  std::cout << "WP_TOT: " << wp_tot << "\n";
  
  int vis_blocks=128;
  cudaError_check(cudaMallocHost((void**)&flat_vis_dat, sizeof(struct flat_vis_data)));
  cudaError_check(cudaMallocHost((void**)&vis_bins, sizeof(struct flat_vis_data) * total_chunks));
  cudaError_check(cudaMallocHost((void**)&vis_bins_w, sizeof(struct flat_vis_data) * total_chunks * wp_tot));
  flatten_visibilities_CUDA(vis_dat, flat_vis_dat);
  //  weight_flat((unsigned int *)gridh, grid_size, theta, flat_vis_dat);
  cudaError_check(cudaMemset(grid,0.0,total_gs * sizeof(cuDoubleComplex)));
  bin_flat_uv_bins(vis_bins, flat_vis_dat, chunk_count_1d, wincrement, theta, grid_size, chunk_size, &wp_min, &wp_max);
  free_flat_visibilities_CUDAh(flat_vis_dat, 1);
  bin_flat_w_vis(vis_bins, vis_bins_w, vis_w_max, vis_w_min, wincrement, chunk_count_1d);
  //free_flat_visibilities_CUDAh(vis_bins, chunk_count_1d * chunk_count_1d);
  struct flat_vis_data *flat_vis_dat_chunked;
  cudaError_check(cudaMallocManaged((void **)&flat_vis_dat_chunked, sizeof(struct flat_vis_data) * total_chunks * wp_tot * vis_blocks));
  
  for(int i = 0; i < total_chunks * wp_tot; ++i){
    bin_flat_visibilities(flat_vis_dat_chunked+vis_blocks*i, vis_bins_w+i, vis_blocks);
  }  
  //free_flat_visibilities_CUDAd(vis_bins_w, chunk_count_1d * chunk_count_1d * wp_tot);

  //Create the fresnel interference pattern for the W-Dimension
  //Can make this a kernel.
  //See Tim Cornwells paper on W-Projection for more information.
  int subgrid_mem_size = sizeof(cuDoubleComplex) * subgrid_size * subgrid_size;  
  cuDoubleComplex *wtransfer;
  cudaError_check(cudaMallocManaged((void **)&wtransfer, subgrid_mem_size, cudaMemAttachGlobal));
  std::cout << "Generating Fresnel Pattern... \n";
  int x,y;
  for (y=0; y < subgrid_size; ++y){
    for (x=0; x < subgrid_size; ++x){
      double l = theta * (double)(x - subgrid_size / 2) / subgrid_size;
      double m = theta * (double)(y - subgrid_size / 2) / subgrid_size;
      double ph = wincrement * (1 - sqrt(1 - l*l - m*m));
      //wtransfer[y * subgrid_size + x] = make_cuDoubleComplex(1.0,0.0);
      cuDoubleComplex wtrans = make_cuDoubleComplex(0.0, 2 * M_PI * ph);
      //wtransfer[y * subgrid_size + x] = wtrans;
      wtransfer[y * subgrid_size + x] = cu_cexp_d(wtrans);
    }
  }
  fft_shift(wtransfer, subgrid_size);

  //Allocate all our subgrids/subimgs contiguously.
  cuDoubleComplex *subgrids, *subimgs;
  
  cudaError_check(cudaMallocManaged((void **)&subgrids,
				    total_chunks * subgrid_mem_size,
				    cudaMemAttachGlobal));
  cudaError_check(cudaMallocManaged((void **)&subimgs,
				    total_chunks * subgrid_mem_size,
				    cudaMemAttachGlobal));

  //Create streams for each tower and allocate our chunks in unified memory.
  //Also set our FFT plans while we are here.
  cudaStream_t *streams = (cudaStream_t *) malloc(total_chunks * sizeof(cudaStream_t));
  cufftHandle *subgrid_plans = (cufftHandle *) malloc(total_chunks * sizeof(cufftHandle));
  
  for(int i = 0; i < total_chunks; ++i){

    //Create stream.
    cudaError_check(cudaStreamCreate(&streams[i]));

    //Assign FFT Plan to each stream
    cuFFTError_check(cufftPlan2d(&subgrid_plans[i],subgrid_size,subgrid_size,CUFFT_Z2Z));
    cuFFTError_check(cufftSetStream(subgrid_plans[i], streams[i]));  
  }

  //Our FFT plan for our final transform.
  cufftHandle grid_plan;
  cuFFTError_check(cufftPlan2d(&grid_plan, grid_size, grid_size, CUFFT_Z2Z));

  int fft_gs = 32;
  int fft_bs = subgrid_size/fft_gs;
  dim3 dimGrid(fft_bs,fft_bs);
  dim3 dimBlock(fft_gs,fft_gs);
  dim3 dimBlock_main(32,32);
  dim3 dimGrid_main(64,64);


  // Lets get gridding!

  std::cout << "Begin Gridding. W-Plane Min/Max: " << wp_min << " " << wp_max << "\n\n";

  int wkern_size = wkern_dat->size_x;
  int wkern_wstep = wkern_dat->w_step;

  //Record Start
  cudaEventCreate(&start);
  cudaEventRecord(start,0);
  
  for(int chunk =0; chunk < total_chunks; ++chunk){

    int subgrid_offset = chunk * subgrid_size * subgrid_size;
    int last_wp = wp_min;
    int cx = chunk % chunk_count_1d;
    int cy = floor(chunk / chunk_count_1d);
    int x_min = cx * chunk_size - grid_size/2;
    int y_min = cy * chunk_size - grid_size/2;

    double u_mid = (double)(x_min + chunk_size / 2) / theta;
    double v_mid = (double)(y_min + chunk_size / 2) / theta;

    cudaError_check(cudaMemsetAsync(subgrids+subgrid_offset, 0, subgrid_mem_size, streams[chunk]));
    cudaError_check(cudaMemsetAsync(subimgs+subgrid_offset, 0, subgrid_mem_size, streams[chunk]));
    
    for(int wp = wp_min; wp<=wp_max; ++wp){
      //std::cout << "WP: " << wp << "\n";
      double w_mid = (double)wp * wincrement;

      int wp_zero = wp + abs(wp_min);
      struct flat_vis_data *binp = flat_vis_dat_chunked + chunk * wp_tot * vis_blocks + wp_zero * vis_blocks;
      struct flat_vis_data *bincp = vis_bins_w + chunk * wp_tot + wp_zero;
      //if(bincp->number_of_vis){ //If no visibilities on that particular floor just skip it.
#ifdef __COUNT_VIS__
	scatter_grid_kernel_flat <<< vis_blocks, 128, 0, streams[chunk] >>>
	  (binp, wkern_dat, subgrids+subgrid_offset, wkern_size,
	   subgrid_size, wkern_wstep, theta,
	   u_mid, v_mid, w_mid, vis_count);
#else
	scatter_grid_kernel_flat <<< vis_blocks, 128, 0, streams[chunk] >>>
	  (binp, wkern_dat, subgrids+subgrid_offset, wkern_size,
	   subgrid_size, wkern_wstep, theta,
	   u_mid, v_mid, w_mid);
#endif
      cuFFTError_check(cufftExecZ2Z(subgrid_plans[chunk], subgrids+subgrid_offset, subgrids+subgrid_offset, CUFFT_INVERSE));
      //}
      fresnel_subimg_mul <<< dimGrid, dimBlock, 0, streams[chunk] >>> (subgrids+subgrid_offset, wtransfer, subimgs+subgrid_offset, subgrid_size, wp - last_wp);
      
      last_wp = wp;
    }
    if(last_wp != 0){
    w0_transfer_kernel <<< dimGrid , dimBlock, 0, streams[chunk] >>> (subimgs+subgrid_offset, wtransfer, last_wp, subgrid_size);
    }
    cuFFTError_check(cufftExecZ2Z(subgrid_plans[chunk], subimgs+subgrid_offset, subimgs+subgrid_offset, CUFFT_FORWARD)); //This can be sped up with a batched strided FFT. Future optimisation...
    

  }
  cudaError_check(cudaDeviceSynchronize());

  cudaEventCreate(&stop);
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime,start,stop);
  cudaError_check(cudaMemGetInfo(&free_bytes, &total_bytes));
  std::cout << "Memory Usage:: Free: " << free_bytes << " Total: " << total_bytes << "\n";
  std::cout << "Scatter Gridder Elapsed Time: " << elapsedTime/1000.0 << " seconds\n";
  #ifdef __COUNT_VIS__
  std::cout << "Visibilities processed: " << *vis_count << "\n";
  #endif
  add_subs2main_kernel <<< dimGrid_main, dimBlock_main >>> (grid, subimgs, grid_size, subgrid_size,
  					  subgrid_margin, chunk_count_1d, chunk_size);  
  fft_shift_kernel <<< dimGrid_main, dimBlock_main >>> (grid, grid_size);
  cuFFTError_check(cufftExecZ2Z(grid_plan, grid, grid, CUFFT_INVERSE));
  fft_shift_kernel <<< dimGrid_main, dimBlock_main >>> (grid, grid_size);
  
  return error;

}

// Pure W-Projection on a Hierarchical Dataset. (AoS)
__host__ cudaError_t wproj_CUDA(const char* visfile, const char* wkernfile,
				cuDoubleComplex *grid, cuDoubleComplex *gridh, int grid_size,
				double theta,  double lambda, double bl_min, double bl_max,
				int threads_per_block){
  #ifdef __COUNT_VIS__
  unsigned long long int *vis_count;
  cudaError_check(cudaMallocManaged((void **)&vis_count, sizeof(unsigned long long int)));
  #endif

  //For Benchmarking.
  size_t free_bytes;
  size_t total_bytes;

  cudaError_t error = (cudaError_t)0; //Initialise as CUDA_Success
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
    
  //Weight visibilities

  //weight((unsigned int *)gridh, grid_size, theta, vis_dat);
  std::cout << "Inititalise scatter gridder... \n";

  cudaEventCreate(&start);
  cudaEventRecord(start, 0);
  #ifdef __COUNT_VIS__
  scatter_grid_kernel <<< 1 , 32 >>> (vis_dat,wkern_dat, grid, wkern_dat->size_x,
				       grid_size,wkern_dat->w_step, theta, 0, 0, 0, vis_count);
  #else
  scatter_grid_kernel <<< 1 , 32 >>> (vis_dat,wkern_dat, grid, wkern_dat->size_x,
				       grid_size,wkern_dat->w_step, theta, 0, 0, 0);
  #endif
  cudaEventCreate(&stop);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime,start,stop);
  cudaError_check(cudaMemGetInfo(&free_bytes, &total_bytes));  
  std::cout << "Memory Usage:: Free: " << free_bytes << " Total: " << total_bytes << "\n";
  std::cout << "Scatter Gridder Elapsed Time: " << elapsedTime/1000.0 << " seconds\n";

  //Shift our grid to the right position for the FFT.
  int fft_gs = 32;
  int fft_bs = grid_size / fft_gs;
  
  dim3 dimBlock(fft_bs,fft_bs);
  dim3 dimGrid(fft_gs,fft_gs);
  fft_shift_kernel <<< dimBlock, dimGrid >>> (grid, grid_size);
  std::cout << "Executing iFFT back to Image Space... \n";
  
  cufftHandle fft_plan;
  cuFFTError_check(cufftPlan2d(&fft_plan,grid_size,grid_size,CUFFT_Z2Z));
  cuFFTError_check(cufftExecZ2Z(fft_plan, grid, grid, CUFFT_INVERSE));
  fft_shift_kernel <<< dimBlock, dimGrid >>> (grid, grid_size);

  return error;
}

// W-Project on flat SoA Dataset.
__host__ cudaError_t wproj_CUDA_flat(const char* visfile, const char* wkernfile,
				     cuDoubleComplex *grid, cuDoubleComplex *gridh, int grid_size,
				     double theta,  double lambda, double bl_min, double bl_max,
				     int threads_per_block){
  
  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
  //For Benchmarking.
  size_t free_bytes;
  size_t total_bytes;


  
  cudaError_t error = (cudaError_t)0;
  cudaEvent_t start, stop;
  float elapsedTime;

  #ifdef __COUNT_VIS__
  unsigned long long int *vis_count;
  cudaError_check(cudaMallocManaged((void **)&vis_count, sizeof(unsigned long long int)));
  #endif

  // Load visibility and w-kernel data from HDF5 files.
  struct vis_data *vis_dat = (struct vis_data*)malloc(sizeof(struct vis_data));
  struct w_kernel_data *wkern_dat;
   vis_dat->antenna_count = 0;
  cudaError_check(cudaMallocManaged((void **)&wkern_dat, sizeof(struct w_kernel_data), cudaMemAttachGlobal));

  int error_hdf5;
  error_hdf5 = load_vis(visfile,vis_dat,bl_min,bl_max);
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
  
  //Make sure our grids are all zero.
  cudaError_check(cudaMemset(grid, 0, total_gs * sizeof(cuDoubleComplex)));
  std::cout << "Bin Data: \n";
  struct flat_vis_data *flat_vis_dat;
  cudaError_check(cudaMallocHost((void**)&flat_vis_dat, sizeof(struct flat_vis_data)));

  //Flatten the visibilities and weight them.
  flatten_visibilities_CUDA(vis_dat,flat_vis_dat);
  //weight_flat((unsigned int *)gridh, grid_size, theta, flat_vis_dat);
  std::cout << "No of Vis: " << flat_vis_dat->number_of_vis << "\n";
  //Now bin them per block.
  struct flat_vis_data *vis_bins;
  cudaError_check(cudaMallocManaged((void**)&vis_bins, sizeof(struct flat_vis_data) * 512, cudaMemAttachGlobal));
  cudaError_check(cudaMemset(vis_bins, 0, sizeof(struct flat_vis_data) * 512));
  bin_flat_visibilities(vis_bins, flat_vis_dat, 512);

  //Get scattering..
  std::cout << "Inititalise scatter gridder... \n";
  cudaEventCreate(&start);
  cudaEventRecord(start, 0);
  #ifdef __COUNT_VIS__
  scatter_grid_kernel_flat <<< 512, 256 >>> (vis_bins, wkern_dat, grid, wkern_dat->size_x,
					     grid_size, wkern_dat->w_step, theta, 0, 0, 0, vis_count);
  #else
  scatter_grid_kernel_flat <<< 512, 256 >>> (vis_bins, wkern_dat, grid, wkern_dat->size_x,
					      grid_size, wkern_dat->w_step, theta, 0, 0, 0);
  #endif
  cudaEventCreate(&stop);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime,start,stop);
  cudaError_check(cudaMemGetInfo(&free_bytes, &total_bytes));
  std::cout << "Scatter Gridder Elapsed Time: " << elapsedTime/1000.0 << " seconds\n";
  std::cout << "Memory Usage:: Free: " << free_bytes << " Total: " << total_bytes << "\n";
  #ifdef __COUNT_VIS__
  std::cout << "Visibilities Processed: " << *vis_count << "\n";
  #endif
  //Shift our grid to the right position for the FFT.
  int fft_gs = 32;
  int fft_bs = grid_size / fft_gs;
  
  dim3 dimBlock(fft_bs,fft_bs);
  dim3 dimGrid(fft_gs,fft_gs);
  fft_shift_kernel <<< dimBlock, dimGrid >>> (grid, grid_size);
  std::cout << "Executing iFFT back to Image Space... \n";
  
  cufftHandle fft_plan;
  cuFFTError_check(cufftPlan2d(&fft_plan,grid_size,grid_size,CUFFT_Z2Z));
  cuFFTError_check(cufftExecZ2Z(fft_plan, grid, grid, CUFFT_INVERSE));
  fft_shift_kernel <<< dimBlock, dimGrid >>> (grid, grid_size);
  
  return error;
} 
