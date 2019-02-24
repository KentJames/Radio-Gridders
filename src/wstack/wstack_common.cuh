#include "hdf5_h.h"
#include "radio.cuh"

#ifndef WSTACK_H
#define WSTACK_H

#include <cuComplex.h>
#include "cuda.h"
#include "cuda_runtime_api.h"
#define THREADS_BLOCK 16


std::complex<double> wstack_predict(double theta,
				    double lam,
				    int npts,
				    double u,
				    double v,
				    double w,
				    double du, // Sze-Tan Optimum Spacing in U/V
				    double dw, // Sze-Tan Optimum Spacing in W
				    double aa_support_uv,
				    double aa_support_w,
				    struct sep_kernel_data *grid_conv_uv,
				    struct sep_kernel_data *grid_conv_w,
				    struct sep_kernel_data *grid_corr_lm,
				    struct sep_kernel_data *grid_corr_n);




#endif
