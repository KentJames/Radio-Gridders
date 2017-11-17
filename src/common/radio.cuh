//Defines lots of inline functions for our CUDA files.

#ifndef RADIO_H
#define RADIO_H

//C++ Includes
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cassert>

//CUDA Includes
#include <cufft.h>
#include <cuComplex.h>
#include "cuda.h"
#include "cuda_runtime_api.h"

#include "hdf5_h.h"


/****************************
	CUDA Error Checkers
*****************************/		

 
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
******************************/

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600


#else //Pre-pascal devices.

__device__ inline double atomicAdd(double* address, double val)
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

// Complex functions that I wished were implemented...
// Sometimes I feel NVIDIA's Complex library is a bit half finished.

__host__ __device__ inline cuDoubleComplex cu_cexp_d (cuDoubleComplex z){

  cuDoubleComplex res;
  double t = exp (z.x);
  sincos (z.y, &res.y, &res.x);
  res.x *= t;
  res.y *= t;
  return res;

}

//Stolen from Peter Wortmann (who stole it from Stack Overflow)
__host__ __device__ inline cuDoubleComplex cu_cpow(cuDoubleComplex base, int exp){

  cuDoubleComplex result = make_cuDoubleComplex(1.0,1.0);
  //Can't recurse on a device function!!!
  //  if (exp < 0) return cuCdiv(make_cuDoubleComplex(1.0,1.0), cu_cpow(base, -exp));
  if (exp < 0) return base; 
  if (exp == 1) return base;
  while (exp){
    if (exp & 1) result = cuCmul(base,result);
    exp >>= 1;
    base = cuCmul(base,base);
  }
  return result;
}


//Gets minimum/maximum co-ordinate in a particular baseline.
__host__ __device__ inline double lambda_min(struct bl_data *bl_data, double u) {
    return u * (u < 0 ? bl_data->f_max : bl_data->f_min) / c;
}

__host__ __device__ inline double lambda_max(struct bl_data *bl_data, double u) {
    return u * (u < 0 ? bl_data->f_min : bl_data->f_max) / c;
}


__host__ __device__ inline static double uvw_lambda(struct bl_data *bl_data,
				  int time, int freq, int uvw) {
    return bl_data->uvw[3*time+uvw] * bl_data->freq[freq] / c;
    
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

/**************************************
              Host Functions
****************************************/


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

//W-Towers binning. Splits visibilities according to their respective chunk.
__host__ inline void bin_flat_uv_bins(struct flat_vis_data *vis_bins,
				  struct flat_vis_data *vis,
				  int chunk_count,
				  int wincrement,
				  double theta,
				  int grid_size,
				  int chunk_size,
				  int *w_min,
				  int *w_max){

  //1) Bin in U-V in first instance
  int total_bins = chunk_count * chunk_count;

  //1a) Pre-compute amount of memory for each u-v bin.
  int *bin_chunk_count = (int *)malloc(total_bins * sizeof(int));
  memset(bin_chunk_count, 0, total_bins * sizeof(int));
  for(int vi = 0; vi < vis->number_of_vis; ++vi){
    
    double u = vis->u[vi];
    double v = vis->v[vi];

    int cx = (floor(u * theta + 0.5) + grid_size/2) / chunk_size;
    int cy = (floor(v * theta + 0.5) + grid_size/2) / chunk_size;

    ++bin_chunk_count[cy * chunk_count + cx];
  }
  
  //1b) Allocate memory

  for(int cyi = 0; cyi< chunk_count; ++cyi){
    for(int cxi = 0; cxi< chunk_count; ++cxi){
      int nv = bin_chunk_count[cyi * chunk_count + cxi];
      std::cout<< "Vis count in chunk " << (cyi * chunk_count + cxi) << " : " << nv << "\n";
      
      cudaError_check(cudaMallocManaged((void **)&vis_bins[cyi * chunk_count + cxi].u, nv * sizeof(double)));
      cudaError_check(cudaMallocManaged((void **)&vis_bins[cyi * chunk_count + cxi].v, nv * sizeof(double)));
      cudaError_check(cudaMallocManaged((void **)&vis_bins[cyi * chunk_count + cxi].w, nv * sizeof(double)));
      cudaError_check(cudaMallocManaged((void **)&vis_bins[cyi * chunk_count + cxi].vis, nv * sizeof(double _Complex)));
      }
  }
  free(bin_chunk_count); //Cleanliness is godliness. 
  
  //1c) Actually bin in memory.
  // We can re-use our bin chunks counts.

  
  for(int vi = 0; vi< vis->number_of_vis; ++vi){

    double u = vis->u[vi];
    double v = vis->v[vi];
    double w = vis->w[vi];
    double _Complex visl = vis->vis[vi];

    int cx = (floor(u * theta + 0.5) + grid_size/2) / chunk_size;
    int cy = (floor(v * theta + 0.5) + grid_size/2) / chunk_size;

    int ci = vis_bins[cy * chunk_count + cx].number_of_vis;
    
    vis_bins[cy * chunk_count + cx].u[ci] = u;
    vis_bins[cy * chunk_count + cx].v[ci] = v;
    vis_bins[cy * chunk_count + cx].w[ci] = w;
    vis_bins[cy * chunk_count + cx].vis[ci] = visl;

    ++vis_bins[cy * chunk_count + cx].number_of_vis;
  }
}


__host__ inline void bin_flat_w_vis(struct flat_vis_data *vis_bins, //Our (filled) U-V gridded bins.
			       struct flat_vis_data *new_bins, //Our (pre-allocated) U-V-W gridder bins. 
			       double w_max, // Maximum w-value.
			       double w_min, // Minimum w-value
			       double wincrement, //Increment between w-planes
			       int chunk_count){


  int wp_min = (int) floor(w_min / wincrement + 0.5);
  int wp_max = (int) floor(w_max / wincrement + 0.5);
  int wp_tot = abs(wp_min - wp_max);
  std::cout << "WP_TOT: " << wp_tot << "\n";
  int total_bins = chunk_count * chunk_count;
  int *bin_chunk_count = (int *)malloc(total_bins * wp_tot * sizeof(int));
  memset(bin_chunk_count, 0, total_bins * wp_tot * sizeof(int));
  //Pre-compute size of bins.
  for(int cyi = 0; cyi < chunk_count; ++cyi){
    for(int cxi = 0; cxi < chunk_count; ++cxi){

      struct flat_vis_data *uv_bin = &vis_bins[cyi * chunk_count + cxi];
      for(int i = 0; i<uv_bin->number_of_vis; ++i){
	double w = uv_bin->w[i];
	int wp = abs((w - w_min) / (wincrement + 0.5));
	++bin_chunk_count[cyi * (chunk_count + wp_tot) + cxi * wp_tot + wp];
      }
    }
  }

  //Allocate memory to all bins. Even empty bins. Kernels run through empty bins in ~50uS.
  for(int cyi = 0; cyi < chunk_count; ++cyi){
    for(int cxi = 0; cxi < chunk_count; ++cxi){
      for(int wp = 0; wp < wp_tot; ++wp){
	int bi = (cyi * (chunk_count * wp_tot)) + (cxi * wp_tot) + wp;
	int nv = bin_chunk_count[bi];
	cudaError_check(cudaMallocManaged((void **)&new_bins[bi].u, nv * sizeof(double)));
	cudaError_check(cudaMallocManaged((void **)&new_bins[bi].v, nv * sizeof(double)));
	cudaError_check(cudaMallocManaged((void **)&new_bins[bi].w, nv * sizeof(double)));
	cudaError_check(cudaMallocManaged((void **)&new_bins[bi].vis, nv * sizeof(double _Complex)));
      }
    }
  }
  
  //Re-bin data as UV -> UVW
  for(int cyi = 0; cyi < chunk_count; ++cyi){
    for(int cxi = 0; cxi < chunk_count; ++cxi){

      struct flat_vis_data *uv_bin = &vis_bins[cyi * chunk_count + cxi];
      for(int vi = 0; vi < uv_bin->number_of_vis; ++vi){

	double u = uv_bin->u[vi];
	double v = uv_bin->v[vi];
	double w = uv_bin->w[vi];
	double _Complex visl = uv_bin->vis[vi];
	int wp = abs((w - w_min) / (wincrement + 0.5));

	int bi = cyi * (chunk_count + wp_tot) + cxi * wp_tot + wp;
	int ci = new_bins[bi].number_of_vis;
	new_bins[bi].u[ci] = u;
	new_bins[bi].v[ci] = v;
	new_bins[bi].w[ci] = w;
	new_bins[bi].vis[ci] = visl;

	++new_bins[bi].number_of_vis;
      }
    }
  }			       
}			       



//Splits our visibilities up into contiguous bins, for each block to apply.
__host__ inline void bin_flat_visibilities(struct flat_vis_data *vis_bins,
					   struct flat_vis_data *vis,
					   int blocks){

  std::cout << "Binning Visibilities. No. of vis: " << vis->number_of_vis << " No. of Blocks: " << blocks << "\n";
  
  int vis_per_block = vis->number_of_vis / blocks;
  int leftovers = vis->number_of_vis % blocks;


  int i;
  for(i = 0; i < blocks-1; ++i){

    cudaError_check(cudaMallocManaged((void**)&(vis_bins+i)->u,
				      sizeof(double) * vis_per_block, cudaMemAttachGlobal));
    cudaError_check(cudaMallocManaged((void**)&(vis_bins+i)->v,
				      sizeof(double) * vis_per_block, cudaMemAttachGlobal));
    cudaError_check(cudaMallocManaged((void**)&(vis_bins+i)->w,
				      sizeof(double) * vis_per_block, cudaMemAttachGlobal));
    cudaError_check(cudaMallocManaged((void**)&(vis_bins+i)->vis,
				      sizeof(double _Complex) * vis_per_block, cudaMemAttachGlobal));


    cudaError_check(cudaMemcpy((vis_bins+i)->u, vis->u + vis_per_block * i,
			       sizeof(double) * vis_per_block, cudaMemcpyDefault));
    cudaError_check(cudaMemcpy((vis_bins+i)->v, vis->v + vis_per_block * i,
			       sizeof(double) * vis_per_block, cudaMemcpyDefault));
    cudaError_check(cudaMemcpy((vis_bins+i)->w, vis->w + vis_per_block * i,
			       sizeof(double) * vis_per_block, cudaMemcpyDefault));
    cudaError_check(cudaMemcpy((vis_bins+i)->vis, vis->vis + vis_per_block * i,
			       sizeof(double _Complex) * vis_per_block, cudaMemcpyDefault));
    (vis_bins+i)->number_of_vis = vis_per_block;
  }
  
  //Last one gets remainders.


  cudaError_check(cudaMallocManaged((void**)&(vis_bins+i)->u,
				    sizeof(double) * (vis_per_block + leftovers), cudaMemAttachGlobal));
  cudaError_check(cudaMallocManaged((void**)&(vis_bins+i)->v,
				    sizeof(double) * (vis_per_block + leftovers), cudaMemAttachGlobal));
  cudaError_check(cudaMallocManaged((void**)&(vis_bins+i)->w,
				    sizeof(double) * (vis_per_block + leftovers), cudaMemAttachGlobal));
  cudaError_check(cudaMallocManaged((void**)&(vis_bins+i)->vis,
				    sizeof(double _Complex) * (vis_per_block + leftovers), cudaMemAttachGlobal));
    

  
  cudaError_check(cudaMemcpy((vis_bins+i)->u, vis->u + vis_per_block * i,
			     sizeof(double) * (vis_per_block+leftovers), cudaMemcpyDefault));
  cudaError_check(cudaMemcpy((vis_bins+i)->v, vis->v + vis_per_block * i,
			     sizeof(double) * (vis_per_block+leftovers), cudaMemcpyDefault));
  cudaError_check(cudaMemcpy((vis_bins+i)->w, vis->w + vis_per_block * i,
			     sizeof(double) * (vis_per_block+leftovers), cudaMemcpyDefault));
  cudaError_check(cudaMemcpy((vis_bins+i)->vis, vis->vis + vis_per_block * i,
			     sizeof(double _Complex) * (vis_per_block+leftovers), cudaMemcpyDefault));
   (vis_bins+i)->number_of_vis = vis_per_block + leftovers;

  
}


//Bins visibilities in u/v for w-towers style subgrids.
__host__ inline void bin_visibilities(struct vis_data *vis, struct vis_data *bins,
				      int chunk_count, int wincrement, double theta,
				      int grid_size, int chunk_size, int *w_min, int *w_max){

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

  *w_min = wp_min; // Report w-values back to calling function
  *w_max = wp_max;

  // Bin in uv
  int bins_size = sizeof(struct vis_data) * chunk_count * chunk_count;
  //cudaError_check(cudaMallocManaged(&bins, bins_size, cudaMemAttachGlobal));
  //cudaError_check(cudaMemset(bins, 0, bins_size));
  
  int bins_count_size = sizeof(int) * chunk_count * chunk_count;
  int *bins_count;
  cudaError_check(cudaMallocManaged((void**)&bins_count, bins_count_size, cudaMemAttachGlobal));
  cudaError_check(cudaMemset(bins_count, 0, bins_count_size));
  for (bl = 0; bl < vis->bl_count; bl++) {
    
    // Determine bounds (could be more precise, future work...)
    struct bl_data *bl_data = &vis->bl[bl];
    //cudaError_check(cudaMallocManaged(&bl_data, sizeof(struct bl_data), cudaMemAttachGlobal));
    //bl_data = &vis->bl[bl];
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
	
	int bcount = ++bins[cy*chunk_count + cx].bl_count;
	int bcount_p = bcount - 1;
	
	struct bl_data *bl_data_old = bins[cy*chunk_count+cx].bl;
	struct bl_data *temp;
	
	cudaError_check(cudaMallocManaged((void**)&temp,
					  sizeof(struct bl_data) * bcount));
	cudaError_check(cudaMemcpy(temp, bl_data_old,
				   sizeof(struct bl_data)*(bcount-1),
				   cudaMemcpyDefault));
	cudaError_check(cudaMemcpy(temp+bcount_p, bl_data,
				   sizeof(struct bl_data),cudaMemcpyDefault));

	bins[cy*chunk_count + cx].bl = temp;
	cudaError_check(cudaFree(bl_data_old));
	//	cudaError_check(cudaFree(temp));
						
      }
    }
  }
  std::cout << "Bins processed: " << bins_size << "\n";
}


// Free all flattened visibilities.

__host__ inline void free_flat_visibilities(struct flat_vis_data *fvd, int length){

  for(int i = 0; i < length; ++i){

    free(fvd[i].u);
    free(fvd[i].v);
    free(fvd[i].w);
    free(fvd[i].vis);

  }
  free(fvd);
}

__host__ inline void free_flat_visibilities_CUDAh(struct flat_vis_data *fvd, int length){

  for(int i = 0; i < length; ++i){
    cudaError_check(cudaFreeHost((void *)fvd[i].u));
    cudaError_check(cudaFreeHost((void *)fvd[i].v));
    cudaError_check(cudaFreeHost((void *)fvd[i].w));
    cudaError_check(cudaFreeHost((void *)fvd[i].vis));
  }
  cudaError_check(cudaFreeHost(fvd));
}

__host__ inline void free_flat_visibilities_CUDAd(struct flat_vis_data *fvd, int length){

  for(int i = 0; i < length; ++i){
    cudaError_check(cudaFree((void *)fvd[i].u));
    cudaError_check(cudaFree((void *)fvd[i].v));
    cudaError_check(cudaFree((void *)fvd[i].w));
    cudaError_check(cudaFree((void *)fvd[i].vis));
  }
  cudaError_check(cudaFree(fvd));
}
#endif
