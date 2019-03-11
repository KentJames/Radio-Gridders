#ifdef CUDA_ACCELERATION
#include "radio.cuh"
#endif
#include "hdf5_h.h"

#ifndef WSTACK_H
#define WSTACK_H

#define THREADS_BLOCK 16



std::complex<double> predict_visibility(const std::vector<double>& points,
					double u,
					double v,
					double w);

std::complex<double> predict_visibility_quantized(const std::vector<double>& points,
						  double theta,
						  double lam,
						  double u,
						  double v,
						  double w);

std::vector<double> generate_random_points(int npts, double theta, double lam);


std::complex<double> wstack_predict(double theta,
				    double lam,
				    const std::vector<double>& points,
				    double u,
				    double v,
				    double w,
				    double du, // Sze-Tan Optimum Spacing in U/V
				    double dw, // Sze-Tan Optimum Spacing in W
				    int aa_support_uv,
				    int aa_support_w,
				    double x0,
				    struct sep_kernel_data *grid_conv_uv,
				    struct sep_kernel_data *grid_conv_w,
				    struct sep_kernel_data *grid_corr_lm,
				    struct sep_kernel_data *grid_corr_n);

// std::complex<double> wstack_predict_test(double theta,
// 					 double lam,
// 					 const std::vector<double>& points,
// 					 double u,
// 					 double v,
// 					 double w,
// 					 double du, // Sze-Tan Optimum Spacing in U/V
// 					 double dw, // Sze-Tan Optimum Spacing in W
// 					 double aa_support_uv,
// 					 double aa_support_w,
// 					 struct sep_kernel_data *grid_conv_uv,
// 					 struct sep_kernel_data *grid_conv_w,
// 					 struct sep_kernel_data *grid_corr_lm,
// 					 struct sep_kernel_data *grid_corr_n);




#endif
