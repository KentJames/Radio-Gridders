#ifndef HDF5_CUH
#define HDF5_CUH

#ifdef CUDA_ACCELERATION
#include "cuda.h"
#include "cuda_runtime_api.h"
#endif

//HDF5 is C, so lets avoid the name mangling by the c++ compiler.
#ifdef __cplusplus
extern "C" {
  #endif


  static const double c = 299792458.0;

  //Fudge constant for resolution.
  //Resolution varies depending on block/thread sizing.
  static const double resolution = 1.0;

  struct flat_vis_uvw_bin
  {
    double *u, *v, *w;
    double _Complex *vis;
    double u_mid, v_mid, w_mid;
    int number_of_vis;
  };
  

  // Potentially faster for Romein gridder, not too sure yet. Pure SoA vs AoS below.
  struct flat_vis_data
  {

    double *u, *v, *w;
    double _Complex *vis;
    int number_of_vis;
  };

  
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


   // Seperable kernel data
  struct sep_kernel_data
  {
    double *data; // Assumed to be real
    int size;
    int oversampling;
    double du;
    double dw;
    double x0;
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

  int free_vis(struct vis_data *vis);


  int load_vis(const char *filename, struct vis_data *vis,
	       double min_len, double max_len);


  int load_sep_kern(const char *filename, struct sep_kernel_data *sepkern);
  int load_sep_kern_T(const char *filename, struct sep_kernel_data *sepkern);
    

#ifdef VAR_W_KERN
  int load_wkern(const char *filename, double theta, struct var_w_kernel_data *wkern);
#else
  int load_wkern(const char *filename, double theta, struct w_kernel_data *wkern);
#endif


#ifdef CUDA_ACCELERATION



    
  int free_vis_CUDA(struct vis_data *vis);
    
  int load_vis_CUDA(const char *filename, struct vis_data *vis,
		    double min_len, double max_len);

  int load_sep_kern_CUDA(const char *filename, struct sep_kernel_data *sepkern);

  int load_sep_kern_CUDA_T(const char *filename, struct sep_kernel_data *sepkern);
    
  void flatten_visibilities_CUDA(struct vis_data *vis, struct flat_vis_data *flat_vis);

      //You must compile both hdf5.c AND your kernel with -DVAR_W_KERN to use variable w-kernels.
#ifdef VAR_W_KERN
  int load_wkern_CUDA(const char *filename, double theta, struct var_w_kernel_data *wkern);
#else
  int load_wkern_CUDA(const char *filename, double theta, struct w_kernel_data *wkern);
#endif
#endif

    
#ifdef __cplusplus
}
#endif

#endif
