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
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){

  if (code != cudaSuccess){
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

#define cuFFTError_check(ans) { cufftAssert((ans), __FILE__, __LINE__); }
inline void cufftAssert(cufftResult code, const char *file, int line, bool abort=true){

  if (code != CUFFT_SUCCESS){
    fprintf(stderr,"cufftAssert: %d %s %d\n", code, file, line);
    if (abort) exit(code);
  }
}

/*****************************
        Device Functions
 *****************************/

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600


#else //Pre-pascal devices.

__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

#endif

__host__ __device__ inline cuDoubleComplex cu_cexp_d (cuDoubleComplex z){

  cuDoubleComplex res;
  double t = exp (z.x);
  sincos (z.y, &res.y, &res.x);
  res.x *= t;
  res.y *= t;
  return res;

}

__host__ __device__ inline static double uvw_lambda(struct bl_data *bl_data,
				  int time, int freq, int uvw) {
    return bl_data->uvw[3*time+uvw] * bl_data->freq[freq] / c;
    
  }



__host__ __device__ inline static int2 getcoords_xy(double u, double v, int grid_size,
				    double theta, int max_support){

  int2 xy;

  xy.x = ((int)floor(theta * u + 0.5) + (grid_size/2)) % max_support;
  xy.y = ((int)floor(theta * v + 0.5) + (grid_size/2)) % max_support;

  return xy;
    

}

__host__ __device__ inline static void frac_coord_flat(int grid_size, int kernel_size, int oversample,
                              double theta,
                              struct flat_vis_data *vis,
                              int i,
                              double d_u, double d_v,
                              int *grid_offset, int *sub_offset) {
#ifdef ASSUME_UVW_0
    double x = 0, y = 0;
#else
    double x = theta * (vis->u[i] - d_u);
    double y = theta * (vis->v[i] - d_v);
#endif
    int flx = (int)floor(x + .5 / oversample);
    int fly = (int)floor(y + .5 / oversample);
    int xf = (int)floor((x - (double)flx) * oversample + .5);
    int yf = (int)floor((y - (double)fly) * oversample + .5);
    *grid_offset =
        (fly+grid_size/2-kernel_size/2)*grid_size +
        (flx+grid_size/2-kernel_size/2);
    *sub_offset = kernel_size * kernel_size * (yf*oversample + xf);
}



__host__ __device__ inline static void frac_coord(int grid_size, int kernel_size, int oversample,
                              double theta,
                              struct bl_data *bl_data,
                              int time, int freq,
                              double d_u, double d_v,
                              int *grid_offset, int *sub_offset) {
#ifdef ASSUME_UVW_0
    double x = 0, y = 0;
#else
    double x = theta * (uvw_lambda(bl_data, time, freq, 0) - d_u);
    double y = theta * (uvw_lambda(bl_data, time, freq, 1) - d_v);
#endif
    int flx = (int)floor(x + .5 / oversample);
    int fly = (int)floor(y + .5 / oversample);
    int xf = (int)floor((x - (double)flx) * oversample + .5);
    int yf = (int)floor((y - (double)fly) * oversample + .5);
    *grid_offset =
        (fly+grid_size/2-kernel_size/2)*grid_size +
        (flx+grid_size/2-kernel_size/2);
    *sub_offset = kernel_size * kernel_size * (yf*oversample + xf);
}


//From Kyrills implementation in SKA/RC
__device__ inline void scatter_grid_add(cuDoubleComplex *uvgrid, int grid_size, int grid_pitch,
					int grid_point_u, int grid_point_v, cuDoubleComplex sum){

  
  // Atomically add to grid. This is the bottleneck of this kernel.
  if (grid_point_u < 0 || grid_point_u >= grid_size ||
      grid_point_v < 0 || grid_point_v >= grid_size)
    return;

  // Bottom half? Mirror
  //if (grid_point_u >= grid_size / 2) {
  //  grid_point_v = grid_size - grid_point_v - 1;
  //  grid_point_u = grid_size - grid_point_u - 1;
  //}

  // Add to grid. This is the bottleneck of the entire kernel
  atomicAdd(&uvgrid[grid_point_u + grid_pitch*grid_point_v].x, sum.x);
  atomicAdd(&uvgrid[grid_point_u + grid_pitch*grid_point_v].y, sum.y);

}


__device__ inline void scatter_grid_point_flat(
					  struct flat_vis_data *vis, // Our bins of UV Data
					  cuDoubleComplex *uvgrid, // Our main UV Grid
					  struct w_kernel_data *wkern, //Our W-Kernel
					  int max_supp, // Max size of W-Kernel
					  int myU, //Our assigned u/v points.
					  int myV, // ^^^
					  double wstep, // W-Increment 
					  int subgrid_size, //The size of our w-towers subgrid.
					  int subgrid_pitch, // Not too sure about ths one
					  double theta, // Field of View Size
					  int offset_u, // Offset from top left of main grid to t.l of subgrid.
					  int offset_v, // ^^^^
					  int offset_w
					  ){ 

  int grid_point_u = myU, grid_point_v = myV;
  cuDoubleComplex sum  = make_cuDoubleComplex(0.0,0.0);

  short supp = short(wkern->size_x);
  
  //  for (int i = 0; i < visibilities; i++) {
  int vi;
  for (vi = 0; vi < vis->number_of_vis; ++vi){
    

    
    //double u = vis->u[vi];
    //double v = vis->v[vi];
    double w = vis->w[vi];
    int w_plane = fabs((w - wkern->w_min) / (wkern->w_step + .5));
    int grid_offset, sub_offset;
    frac_coord_flat(subgrid_size, wkern->size_x, wkern->oversampling,
		    theta, vis, vi, offset_u, offset_v, &grid_offset, &sub_offset);
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
      scatter_grid_add(uvgrid, subgrid_size, subgrid_pitch, grid_point_u, grid_point_v, sum);
      // Switch to new point
      sum = make_cuDoubleComplex(0.0, 0.0);
      grid_point_u = myGridU;
      grid_point_v = myGridV;
    }
    //TODO: Re-do the w-kernel/gcf for our data.
    //	cuDoubleComplex px;
    cuDoubleComplex px = *(cuDoubleComplex*)&wkern->kern_by_w[w_plane].data[sub_offset + myConvU * supp + myConvV];	
    // Sum up
    cuDoubleComplex vi_v = *(cuDoubleComplex*)&vis->vis[vi];
    sum = cuCfma(px, vi_v, sum);
      
    
  }

  // Add remaining sum to grid
  scatter_grid_add(uvgrid, subgrid_size, subgrid_pitch, grid_point_u, grid_point_v, sum);

}



//From Kyrills Implementation in SKA/RC. Modified to suit our data format.
//Assumes pre-binned (in u/v) data
__device__ inline void scatter_grid_point(
					  struct bl_data **bin, // Our bins of UV Data
					  int bl_count, // Number of baselines.
					  cuDoubleComplex *uvgrid, // Our main UV Grid
					  struct w_kernel_data *wkern, //Our W-Kernel
					  int max_supp, // Max size of W-Kernel
					  int myU, //Our assigned u/v points.
					  int myV, // ^^^
					  double wstep, // W-Increment 
					  int subgrid_size, //The size of our w-towers subgrid.
					  int subgrid_pitch, // Not too sure about ths one
					  double theta, // Field of View Size
					  int offset_u, // Offset from top left of main grid to t.l of subgrid.
					  int offset_v, // ^^^^
					  int offset_w
					  ){ 

  int grid_point_u = myU, grid_point_v = myV;
  cuDoubleComplex sum  = make_cuDoubleComplex(0.0,0.0);

  short supp = short(wkern->size_x);
  
  //  for (int i = 0; i < visibilities; i++) {
  int bl, time, freq;
  for (bl = 0; bl < bl_count; ++bl){
    struct bl_data *bl_d = *bin+bl;
    for (time = 0; time < bl_d->time_count; ++time){
      for(freq = 0; freq < bl_d->freq_count; ++freq){
	// Load pre-calculated positions
	//int u = uvo[i].u, v = uvo[i].v;
	//	int u = (int)uvw_lambda(bl_d, time, freq, 0);
	//int v = (int)uvw_lambda(bl_d, time, freq, 1);
	double w = uvw_lambda(bl_d, time, freq, 2) - offset_w;
	int w_plane = fabs((w - wkern->w_min) / (wkern->w_step + .5));
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
	  scatter_grid_add(uvgrid, subgrid_size, subgrid_pitch, grid_point_u, grid_point_v, sum);
	  // Switch to new point
	  sum = make_cuDoubleComplex(0.0, 0.0);
	  grid_point_u = myGridU;
	  grid_point_v = myGridV;
	  }
	//TODO: Re-do the w-kernel/gcf for our data.
	//	cuDoubleComplex px;
	cuDoubleComplex px = *(cuDoubleComplex*)&wkern->kern_by_w[w_plane].data[sub_offset + myConvU * supp + myConvV];	
	// Sum up
	cuDoubleComplex vi = *(cuDoubleComplex*)&bl_d->vis[time*bl_d->freq_count+freq];
	sum = cuCfma(px, vi, sum);
      }
    }
  }

  // Add remaining sum to grid
  scatter_grid_add(uvgrid, subgrid_size, subgrid_pitch, grid_point_u, grid_point_v, sum);

}


/******************************
            Kernels
*******************************/


//This is our Romein-style scatter gridder. Works on flat visibility data.
__global__ void scatter_grid_kernel_flat(
				    struct flat_vis_data *vis, // No. of visibilities
				    struct w_kernel_data *wkern, // No. of wkernels
				    cuDoubleComplex *uvgrid, //Our UV-Grid
				    int max_support, //  Convolution size
				    int subgrid_size, // Subgrid size
				    int subgrid_pitch, // Subgrid pitch (what is this?)
				    double wstep, // W-Increment
				    double theta, // Field of View
				    int offset_u, // Top left offset from top left main grid
				    int offset_v, // ^^^^
				    int offset_w // W Offset
				    ){

  //Assign some visibilities to grid;

  

  
  for(int i = threadIdx.x; i < max_support * max_support; i += blockDim.x){
    //  int i = threadIdx.x + blockIdx.x * blockDim.x;
    int myU = i % max_support;
    int myV = i / max_support;

    scatter_grid_point_flat(vis+blockIdx.x, uvgrid, wkern, max_support, myU, myV, wstep,
		       subgrid_size, subgrid_pitch, theta, offset_u, offset_v, offset_w);
		       
  }
}


//This is our Romein-style scatter gridder. Works on hierarchical visibility data (bl->time->freq)
__global__ void scatter_grid_kernel(struct bl_data **bin, // Baseline bin
				    int bl_count, // No. of baselines
				    struct vis_data *vis, // No. of visibilities
				    struct w_kernel_data *wkern, // No. of wkernels
				    cuDoubleComplex *uvgrid, //Our UV-Grid
				    int max_support, //  Convolution size
				    int subgrid_size, // Subgrid size
				    int subgrid_pitch, // Subgrid pitch (what is this?)
				    double wstep, // W-Increment
				    double theta, // Field of View
				    int offset_u, // Top left offset from top left main grid
				    int offset_v, // ^^^^
				    int offset_w // W Offset
				    ){
  
  for(int i = threadIdx.x; i < max_support * max_support; i += blockDim.x){
    //  int i = threadIdx.x + blockIdx.x * blockDim.x;
    int myU = i % max_support;
    int myV = i / max_support;

    scatter_grid_point(bin, bl_count, uvgrid, wkern, max_support, myU, myV, wstep,
		       subgrid_size, subgrid_pitch, theta, offset_u, offset_v, offset_w);
		       
  }
}

//For multiplying the fresnel pattern.
__global__ void fresnel_pattern_kernel(cuDoubleComplex *subimg, cuDoubleComplex *subgrid,
				       cuDoubleComplex *fresnel, int subgrid_size, int w_plane){

  

}
				       

/******************************
	  Host Functions
*******************************/

//Gets minimum/maximum co-ordinate in a particular baseline.
__host__ inline double lambda_min(struct bl_data *bl_data, double u) {
    return u * (u < 0 ? bl_data->f_max : bl_data->f_min) / c;
}

__host__ inline double lambda_max(struct bl_data *bl_data, double u) {
    return u * (u < 0 ? bl_data->f_min : bl_data->f_max) / c;
}


__host__ inline void init_grid_zero(cuDoubleComplex *uvgrid, int grid_size){

  for(int x = 0; x< grid_size; ++x){
    for(int y = 0; y< grid_size; ++y){
      *(uvgrid+x*grid_size+y) = make_cuDoubleComplex(0.0, 0.0);
    }
  }

}

// Get coarse-grained co-ordinate.
__host__ inline static int coord(int grid_size, double theta,
                 struct bl_data *bl_data,
                 int time, int freq) {
#ifdef ASSUME_UVW_0
    int x = 0, y = 0;
#else
    int x = (int)floor(theta * uvw_lambda(bl_data, time, freq, 0) + .5);
    int y = (int)floor(theta * uvw_lambda(bl_data, time, freq, 1) + .5);
#endif
    return (y+grid_size/2) * grid_size + (x+grid_size/2);
}

// Get coarse-grained co-ordinate.
__host__ inline static int coord_flat(int grid_size, double theta,
                 struct flat_vis_data *vis_data,
                 int vi) {
#ifdef ASSUME_UVW_0
    int x = 0, y = 0;
#else
    int x = (int)floor(theta * vis_data->u[vi] + .5);
    int y = (int)floor(theta * vis_data->v[vi] + .5);
#endif
    return (y+grid_size/2) * grid_size + (x+grid_size/2);
}

// Uniformly weights all visibilities.
__host__ inline void weight(unsigned int *wgrid, int grid_size, double theta,
            struct vis_data *vis) {

  int total_vis=0;
    // Simple uniform weighting
  int bl, time, freq;
    memset(wgrid, 0, grid_size * grid_size * sizeof(unsigned int));
    for (bl = 0; bl < vis->bl_count; bl++) {
        for (time = 0; time < vis->bl[bl].time_count; time++) {
            for (freq = 0; freq < vis->bl[bl].freq_count; freq++) {
                wgrid[coord(grid_size, theta, &vis->bl[bl], time, freq)]++;
		++total_vis;
            }
        }
    }

    
    for (bl = 0; bl < vis->bl_count; bl++) {
        for (time = 0; time < vis->bl[bl].time_count; time++) {
            for (freq = 0; freq < vis->bl[bl].freq_count; freq++) {
                vis->bl[bl].vis[time*vis->bl[bl].freq_count + freq]
                    /= wgrid[coord(grid_size, theta, &vis->bl[bl], time, freq)];
            }
        }
    }

}

// Uniformly weights all visibilities on a flat structre.
__host__ inline void weight_flat(unsigned int *wgrid, int grid_size, double theta,
            struct flat_vis_data *vis) {

    // Simple uniform weighting

    memset(wgrid, 0, grid_size * grid_size * sizeof(unsigned int));
    int vii;

    for (vii = 0; vii<vis->number_of_vis; ++vii){
      wgrid[coord_flat(grid_size, theta, vis, vii)]++;
    }

    for (vii = 0; vii<vis->number_of_vis; ++vii){
      vis->vis[vii] /= wgrid[coord_flat(grid_size, theta, vis, vii)];
    }
      
}


//Shifts middle of image to top left corner, to make sure FFT is correct.
// (Remember to use this again after the FFT too...)
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

//Ensures 2-D array is hermitian symmetric.
__host__ inline void make_hermitian(cuDoubleComplex *uvgrid, int grid_size){

  cuDoubleComplex *p0;

  if (grid_size % 2 == 0) {
    p0 = uvgrid + grid_size + 1;
  }
  else {
    p0 = uvgrid;
  }

  cuDoubleComplex *p1 = uvgrid + grid_size * grid_size - 1;

  while (p0 < p1) {
    cuDoubleComplex g0 = *p0;

    cuCadd(*p0++,cuConj(*p1));
    cuCadd(*p1--,cuConj(g0));
    //    *p0++ += cuConj(*p1);
    //*p1-- += cuConj(g0);
  }

  assert ( p0 == p1 && p0 == uvgrid + (grid_size + 1) * (grid_size/2));
  cuCadd(*p0,cuConj(*p0));
  //  *p0 += cuConj(*p0);


}

// Flattens all visibilities stored in hdf5 into a structure of arrays(SoA) format.
// Might help locality vOv
void flatten_visibilities_CUDA(struct vis_data *vis, struct flat_vis_data *flat_vis){


  int flat_vis_iter=0;


  //Pre-loop to get size. 
  for(int bl = 0; bl<vis->bl_count; ++bl){
    struct bl_data bl_d = vis->bl[bl];
    
    for(int time = 0; time< bl_d.time_count;++time){
      for(int freq = 0; freq< bl_d.freq_count;++freq){
	++flat_vis_iter;
      }
    }
  }

  cudaMallocManaged((void**)&flat_vis->u, sizeof(double) * flat_vis_iter, cudaMemAttachGlobal);
  cudaMallocManaged((void**)&flat_vis->v, sizeof(double) * flat_vis_iter, cudaMemAttachGlobal);
  cudaMallocManaged((void**)&flat_vis->w, sizeof(double) * flat_vis_iter, cudaMemAttachGlobal);
  cudaMallocManaged((void**)&flat_vis->vis, sizeof(double _Complex) * flat_vis_iter, cudaMemAttachGlobal);


  int total_vis = flat_vis_iter;
  flat_vis_iter = 0;
  for(int bl = 0; bl<vis->bl_count; ++bl){
    struct bl_data bl_d = vis->bl[bl];
    
    for(int time = 0; time< bl_d.time_count;++time){
      for(int freq = 0; freq< bl_d.freq_count;++freq){
	++flat_vis_iter;

	//Flatten

	
	flat_vis->u[flat_vis_iter] = uvw_lambda(&bl_d, time, freq, 0);
	flat_vis->v[flat_vis_iter] = uvw_lambda(&bl_d, time, freq, 1);
	flat_vis->w[flat_vis_iter] = uvw_lambda(&bl_d, time, freq, 2);

	flat_vis->vis[flat_vis_iter] = bl_d.vis[time*bl_d.freq_count+freq];
	
	

      }
    }
  }

  flat_vis->number_of_vis = total_vis;
}


//Splits our visibilities up into contiguous bins, for each block to apply.
__host__ inline void bin_flat_visibilities(struct flat_vis_data *vis_bins, struct flat_vis_data *vis,
					   int blocks){

  int vis_per_block = vis->number_of_vis / blocks;
  int leftovers = vis->number_of_vis % blocks;


  int i;
  for(i = 0; i < blocks-1; ++i){

    cudaError_check(cudaMallocManaged((void**)&(vis_bins+i)->u, sizeof(double) * vis_per_block, cudaMemAttachGlobal));
    cudaError_check(cudaMallocManaged((void**)&(vis_bins+i)->v, sizeof(double) * vis_per_block, cudaMemAttachGlobal));
    cudaError_check(cudaMallocManaged((void**)&(vis_bins+i)->w, sizeof(double) * vis_per_block, cudaMemAttachGlobal));
    cudaError_check(cudaMallocManaged((void**)&(vis_bins+i)->vis, sizeof(double _Complex) * vis_per_block, cudaMemAttachGlobal));


    cudaError_check(cudaMemcpy((vis_bins+i)->u, vis->u + vis_per_block * i, sizeof(double) * vis_per_block, cudaMemcpyDefault));
    cudaError_check(cudaMemcpy((vis_bins+i)->v, vis->v + vis_per_block * i, sizeof(double) * vis_per_block, cudaMemcpyDefault));
    cudaError_check(cudaMemcpy((vis_bins+i)->w, vis->w + vis_per_block * i, sizeof(double) * vis_per_block, cudaMemcpyDefault));
    cudaError_check(cudaMemcpy((vis_bins+i)->vis, vis->vis + vis_per_block * i, sizeof(double _Complex) * vis_per_block, cudaMemcpyDefault));
    (vis_bins+i)->number_of_vis = vis_per_block;
  }
  
  //Last one gets remainders.


  cudaError_check(cudaMallocManaged((void**)&(vis_bins+i)->u, sizeof(double) * (vis_per_block + leftovers), cudaMemAttachGlobal));
  cudaError_check(cudaMallocManaged((void**)&(vis_bins+i)->v, sizeof(double) * (vis_per_block + leftovers), cudaMemAttachGlobal));
  cudaError_check(cudaMallocManaged((void**)&(vis_bins+i)->w, sizeof(double) * (vis_per_block + leftovers), cudaMemAttachGlobal));
  cudaError_check(cudaMallocManaged((void**)&(vis_bins+i)->vis, sizeof(double _Complex) * (vis_per_block + leftovers), cudaMemAttachGlobal));
    

  
  cudaError_check(cudaMemcpy((vis_bins+i)->u, vis->u + vis_per_block * i, sizeof(double) * (vis_per_block+leftovers), cudaMemcpyDefault));
  cudaError_check(cudaMemcpy((vis_bins+i)->v, vis->v + vis_per_block * i, sizeof(double) * (vis_per_block+leftovers), cudaMemcpyDefault));
  cudaError_check(cudaMemcpy((vis_bins+i)->w, vis->w + vis_per_block * i, sizeof(double) * (vis_per_block+leftovers), cudaMemcpyDefault));
  cudaError_check(cudaMemcpy((vis_bins+i)->vis, vis->vis + vis_per_block * i, sizeof(double _Complex) * (vis_per_block+leftovers), cudaMemcpyDefault));
   (vis_bins+i)->number_of_vis = vis_per_block + leftovers;

  
}


//Bins visibilities in u/v for w-towers style subgrids.
__host__ inline void bin_visibilities(struct vis_data *vis, struct bl_data ***bins,
				      int chunk_count, int wincrement, double theta,
				      int grid_size, int chunk_size){

  std::cout << "Binning our visibilities in U/V for our chunks..\n";
  // Determine bounds in w
  double vis_w_min = 0, vis_w_max = 0;
  int bl;
  for (bl = 0; bl < vis->bl_count; bl++) {
    double w_min = lambda_min(&vis->bl[bl], vis->bl[bl].w_min);
    double w_max = lambda_max(&vis->bl[bl], vis->bl[bl].w_max);
    if (w_min < vis_w_min) { vis_w_min = w_min; }
    if (w_max > vis_w_max) { vis_w_max = w_max; }
  }

  int wp_min = (int) floor(vis_w_min / wincrement + 0.5);
  int wp_max = (int) floor(vis_w_max / wincrement + 0.5);

  // Bin in uv
  int bins_size = sizeof(void *) * chunk_count * chunk_count;
  cudaError_check(cudaMallocManaged(&bins, bins_size, cudaMemAttachGlobal));
  cudaError_check(cudaMemset(bins, 0, bins_size));
  
  int bins_count_size = sizeof(int) * chunk_count * chunk_count;
  int *bins_count;
  cudaError_check(cudaMallocManaged(&bins_count, bins_count_size, cudaMemAttachGlobal));
  cudaError_check(cudaMemset(bins_count, 0, bins_count_size));
  for (bl = 0; bl < vis->bl_count; bl++) {
    
    // Determine bounds (could be more precise, future work...)
    struct bl_data *bl_data = &vis->bl[bl];
    double u_min = lambda_min(bl_data, bl_data->u_min);
    double u_max = lambda_max(bl_data, bl_data->u_max);
    double v_min = lambda_min(bl_data, bl_data->v_min);
    double v_max = lambda_max(bl_data, bl_data->v_max);
    
    // Determine first/last overlapping grid chunks
    int cx0 = (floor(u_min * theta + 0.5) + grid_size/2) / chunk_size;
    int cx1 = (floor(u_max * theta + 0.5) + grid_size/2) / chunk_size;
    int cy0 = (floor(v_min * theta + 0.5) + grid_size/2) / chunk_size;
    int cy1 = (floor(v_max * theta + 0.5) + grid_size/2) / chunk_size;

    int cy, cx;
    for (cy = cy0; cy <= cy1; cy++) {
      for (cx = cx0; cx <= cx1; cx++) {
	// Lazy dynamically sized vector

	int bcount = ++bins_count[cy*chunk_count + cx];
	int bcount_p = bcount - bins_count[cy*chunk_count + cx];

	// This is a horrible way of doing this.
	// Why can't NVIDIA re-implement realloc? Also C/C++ is the
	// work of satan (all hail boost::any).
	struct bl_data **bl_data_old = bins[cy*chunk_count + cx];
	cudaError_check(cudaMallocManaged(&bins[cy*chunk_count + cx],sizeof(void *) * bcount, cudaMemAttachGlobal));
	cudaError_check(cudaMemcpy((void **)bins[cy*chunk_count + cx], (void **)bl_data_old,sizeof(void *) * --bcount, cudaMemcpyDefault));
	cudaError_check(cudaFree((void **)bl_data_old));
	bins[cy*chunk_count + cx][bcount-1] = bl_data;

      }
    }
  }
  std::cout << "Bins processed: " << bins_size << "\n";
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
  error_hdf5 = load_vis_CUDA(visfile,vis_dat,bl_min,1000);
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

  struct bl_data ***bins;

  bin_visibilities(vis_dat, bins, chunk_count_1d, wincrement, theta, grid_size, subgrid_size);

  // Lets get gridding!
  for(int chunk = 0; chunk < (chunk_count_1d + 1) * (chunk_count_1d + 1); ++chunk){

    

  }
  
  return error;

}


__host__ cudaError_t wprojection_CUDA(const char* visfile, const char* wkernfile, int grid_size,
				      double theta,  double lambda, double bl_min, double bl_max,
				      int threads_per_block){

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

  int blocks = total_gs / 256;

  //Weight visibilities

  weight((unsigned int *)grid_host, grid_size, theta, vis_dat);
    
  struct bl_data **bl_d = &vis_dat->bl;
  
  std::cout << "Inititalise scatter gridder... \n";

  cudaEventCreate(&start);
  cudaEventRecord(start, 0);

  //scatter_grid_kernel_flat <<< 16, 32 >>> (flat_vis_dat, wkern_dat, grid_dev, wkern_dat->size_x,
  //					  grid_size, grid_size, wkern_dat->w_step, theta, 0, 0, 0);
  //
  scatter_grid_kernel <<< 16 , 32 >>> (bl_d,vis_dat->bl_count,
				       vis_dat, wkern_dat, grid_dev, wkern_dat->size_x,
				       grid_size, grid_size, wkern_dat->w_step, theta, 0, 0, 0);
  cudaEventCreate(&stop);
  cudaEventRecord(stop, 0);

  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime,start,stop);

  std::cout << "Scatter Gridder Elapsed Time: " << elapsedTime/1000.0 << " seconds\n";


  free_vis_CUDA(vis_dat);
  //Transfer back to host.
  cudaError_check(cudaMemcpy(grid_host, grid_dev, total_gs * sizeof(cuDoubleComplex),
			     cudaMemcpyDeviceToHost));



//Write Image to disk on host.

  std::ofstream image_pref ("pre_fft.out", std::ofstream::out | std::ofstream::binary);
  std::cout << "Writing Image to File... \n";

  double *row;
  cudaError_check(cudaMallocHost(&row, grid_size * sizeof(double)));

      
  for(int i = 0; i < grid_size; ++i){

    for(int j = 0; j< grid_size; ++j){
      
      row[j] = cuCreal(grid_host[i*grid_size + j]);
    }
    image_pref.write((char*)row, sizeof(double) * grid_size);
  }
  image_pref.close();


  
  fft_shift(grid_host, grid_size);
  make_hermitian(grid_host, grid_size);


  
  cudaError_check(cudaMemcpy(grid_dev, grid_host, total_gs * sizeof(cuDoubleComplex),
			     cudaMemcpyHostToDevice));
  
  std::cout << "Executing iFFT back to Image Space... \n";
  
  cufftHandle fft_plan;
  cuFFTError_check(cufftPlan2d(&fft_plan,grid_size,grid_size,CUFFT_Z2Z));
  cuFFTError_check(cufftExecZ2Z(fft_plan, grid_dev, grid_dev, CUFFT_INVERSE));

  //Transfer back to host.
  cudaError_check(cudaMemcpy(grid_host, grid_dev, total_gs * sizeof(cuDoubleComplex),
			     cudaMemcpyDeviceToHost));


  //Write Image to disk on host.

  std::ofstream image_f ("image.out", std::ofstream::out | std::ofstream::binary);
  std::cout << "Writing Image to File... \n";

  //double *row;
  //cudaError_check(cudaMallocHost(&row, grid_size * sizeof(double)));
  fft_shift(grid_host,grid_size);
  for(int i = 0; i < grid_size; ++i){

    for(int j = 0; j< grid_size; ++j){

      row[j] = cuCreal(grid_host[i*grid_size + j]);
    }
    image_f.write((char*)row, sizeof(double) * grid_size);
  }

  image_f.close();

  //Check it actually ran...
  cudaError_t err = cudaGetLastError();
  

  std::cout << "Error: " << cudaGetErrorString(err) << "\n";
  return err;
}

// W-Project on flat SoA Dataset.
__host__ cudaError_t wprojection_CUDA_flat(const char* visfile, const char* wkernfile, int grid_size,
				      double theta,  double lambda, double bl_min, double bl_max,
				      int threads_per_block){

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

  struct flat_vis_data *flat_vis_dat;
  cudaError_check(cudaMallocManaged((void**)&flat_vis_dat, sizeof(struct flat_vis_data), cudaMemAttachGlobal));

  //Flatten the visibilities and weight them.
  flatten_visibilities_CUDA(vis_dat,flat_vis_dat);
  weight_flat((unsigned int *)grid_host, grid_size, theta, flat_vis_dat);

  //Now bin them per block.
  struct flat_vis_data *vis_bins;
  cudaError_check(cudaMallocManaged((void**)&vis_bins, sizeof(struct flat_vis_data) * 256, cudaMemAttachGlobal));

  bin_flat_visibilities(vis_bins, flat_vis_dat, 256);
  

  
  std::cout << "Inititalise scatter gridder... \n";

  cudaEventCreate(&start);
  cudaEventRecord(start, 0);

  scatter_grid_kernel_flat <<< 256, 32 >>> (vis_bins, wkern_dat, grid_dev, wkern_dat->size_x,
  					  grid_size, grid_size, wkern_dat->w_step, theta, 0, 0, 0);
  
  cudaEventCreate(&stop);
  cudaEventRecord(stop, 0);

  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime,start,stop);

  std::cout << "Scatter Gridder Elapsed Time: " << elapsedTime/1000.0 << " seconds\n";


  //free_vis_CUDA(vis_dat);
  //This needs a kernel at some point.
  
  //Transfer back to host.
  cudaError_check(cudaMemcpy(grid_host, grid_dev, total_gs * sizeof(cuDoubleComplex),
			     cudaMemcpyDeviceToHost));



//Write Image to disk on host.

  std::ofstream image_pref ("pre_fft.out", std::ofstream::out | std::ofstream::binary);
  std::cout << "Writing Image to File... \n";

  double *row;
  cudaError_check(cudaMallocHost(&row, grid_size * sizeof(double)));

      
  for(int i = 0; i < grid_size; ++i){

    for(int j = 0; j< grid_size; ++j){
      
      row[j] = cuCreal(grid_host[i*grid_size + j]);
    }
    image_pref.write((char*)row, sizeof(double) * grid_size);
  }
  image_pref.close();


  
  fft_shift(grid_host, grid_size);
  make_hermitian(grid_host, grid_size);


  
  cudaError_check(cudaMemcpy(grid_dev, grid_host, total_gs * sizeof(cuDoubleComplex),
			     cudaMemcpyHostToDevice));
  
  std::cout << "Executing iFFT back to Image Space... \n";
  
  cufftHandle fft_plan;
  cuFFTError_check(cufftPlan2d(&fft_plan,grid_size,grid_size,CUFFT_Z2Z));
  cuFFTError_check(cufftExecZ2Z(fft_plan, grid_dev, grid_dev, CUFFT_INVERSE));

  //Transfer back to host.
  cudaError_check(cudaMemcpy(grid_host, grid_dev, total_gs * sizeof(cuDoubleComplex),
			     cudaMemcpyDeviceToHost));


  //Write Image to disk on host.

  std::ofstream image_f ("image.out", std::ofstream::out | std::ofstream::binary);
  std::cout << "Writing Image to File... \n";

  fft_shift(grid_host,grid_size);
  for(int i = 0; i < grid_size; ++i){

    for(int j = 0; j< grid_size; ++j){

      row[j] = cuCreal(grid_host[i*grid_size + j]);
    }
    image_f.write((char*)row, sizeof(double) * grid_size);
  }

  image_f.close();

  //Check it actually ran...
  cudaError_t err = cudaGetLastError();
  

  std::cout << "Error: " << cudaGetErrorString(err) << "\n";
  return err;
} 