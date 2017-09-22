#ifndef GRID_H
#define GRID_H

#include <cuComplex.h>
#include "cuda.h"
#include "cuda_runtime_api.h"
#define THREADS_BLOCK 16

//HDF5 is C, so lets avoid the name mangling by the c++ compiler.
#ifdef __cplusplus
extern "C" {
  #endif


  static const double c = 299792458.0;

  //Fudge constant for resolution.
  //Resolution varies depending on block/thread sizing.
  static const double resolution = 1.0;
  
  // Visibility data
  struct bl_data
  {
    int antenna1, antenna2;
    int time_count;
    int freq_count;
    double *time;
    double *freq;
    double *uvw;
    double _Complex *vis;
    double _Complex *awkern;

    double u_min, u_max; // in m
    double v_min, v_max; // in m
    double w_min, w_max; // in m
    double t_min, t_max; // in h
    double f_min, f_max; // in Hz

    uint64_t flops;
  };
  struct vis_data
  {
    int antenna_count;
    int bl_count;
    struct bl_data *bl;
  };

  // Static W-kernel data
  struct w_kernel
  {
    double _Complex *data;
    double w;
  };

  struct w_kernel_data
  {
    int plane_count;
    struct w_kernel *kern;
    struct w_kernel *kern_by_w;
    double w_min, w_max, w_step;
    int size_x, size_y;
    int oversampling;
  };

  // Variable W-Kernel data.

  struct var_w_kernel
  {
    double _Complex *data;
    double w;
    int size_x, size_y;
    int oversampling;
  };

  struct var_w_kernel_data
  {
    int plane_count;
    struct var_w_kernel *kern;
    struct var_w_kernel *kern_by_w;
    double w_min, w_max, w_step;
  };

  // A-kernel data
  struct a_kernel
  {
    double _Complex *data;
    int antenna;
    double time;
    double freq;
  };
  struct a_kernel_data
  {
    int antenna_count, time_count, freq_count;
    struct a_kernel *kern;
    struct a_kernel *kern_by_atf;
    double t_min, t_max, t_step;
    double f_min, f_max, f_step;
    int size_x, size_y;
  };

  // Performance counter data
  struct perf_counters
  {
    int x87;
    int sse_ss;
    int sse_sd;
    int sse_ps;
    int sse_pd;
    int llc_miss;
  };

  void init_dtype_cpx();

  inline static double uvw_lambda(struct bl_data *bl_data,
				  int time, int freq, int uvw) {
    return bl_data->uvw[3*time+uvw] * bl_data->freq[freq] / c;
  }
  

  int load_vis_CUDA(const char *filename, struct vis_data *vis,
		    double min_len, double max_len);

  int load_vis(const char *filename, struct vis_data *vis,
	       double min_len, double max_len);

#ifdef __cplusplus
}
#endif

#ifdef __CUDACC__

//Our CUDA Prototypes.

__device__ cuDoubleComplex calculate_dft_sum(struct vis_data *vis, double l, double m);

__global__ void image_dft(struct vis_data *vis, cuDoubleComplex *uvgrid, int grid_size,
			  double lambda, int iter, int N);


#endif

cudaError_t image_dft_host(const char* visfile,  int grid_size,
			   double theta,  double lambda, double bl_min, double bl_max,
			   int iter);


#endif
