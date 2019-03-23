//C++ Includes
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cassert>
#include <complex>

//CUDA Includes
#include <cuComplex.h>
#include <cufft.h>
#include "cuda.h"
#include "math.h"
#include "cuda_runtime_api.h"

//Thrust (CUDA STL) Includes..
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>


//Our includes
#include "wstack_common.h"
#include "common_kernels.cuh"
#include "radio.cuh"



//Elementwise multiplication of subimg with fresnel.
template <typename RealType>
__global__ void fresnel_sky_mul(thrust::complex<RealType> *sky,
				   thrust::complex<RealType> *fresnel,
				   int n,
				   int wp){

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x < n && y < n){
      thrust::complex<RealType> pown = {wp,0.0};
      thrust::complex<RealType> wtrans = thrust::pow(fresnel[y * n + x], pown);
      thrust::complex<RealType> skyv = sky[y*n + x];
      thrust::complex<RealType> fresnelv = fresnel[y *n + x];
      thrust::complex<RealType> test = {0.0, 0.0};
      
      if(wp == 1){
	  sky[y*n + x] = skyv;
      } else {
	  if (fresnelv == test){} else {
	      sky[y*n + x] = skyv * thrust::pow(fresnelv,wp);
	  }
      }
  }
}


// We return std::complex to keep interface consistency with the C++ code.
__host__ std::complex<double> wstack_predict_cu(double theta,
						double lam,
						const std::vector<double>& points,
						std::vector<double> u,
						std::vector<double> v,
						std::vector<double> w,
						double du,
						double dw,
						int aa_support_uv,
						int aa_support_w,
						double x0,
						struct sep_kernel_data *grid_conv_uv,
						struct sep_kernel_data *grid_conv_w,
						struct sep_kernel_data *grid_corr_lm,
						struct sep_kernel_data *grid_corr_n){
    
    int grid_size = static_cast<int>(std::floor(theta * lam));
    double x0ih = std::round(0.5/x0);
    int oversampg = static_cast<int>(x0ih * grid_size);
    assert(oversampg > grid_size);


    // CUDA Grid Parameters
    int wstack_gs = 32;
    int wstack_bs = grid_size/wstack_gs;
    dim3 dimGrid(wstack_bs,wstack_bs);
    dim3 dimBlock(wstack_gs,wstack_gs);
    

    // Fresnel Pattern
    
    vector2D<std::complex<double>> wtransfer = generate_fresnel(theta,lam,dw,x0);
    thrust::host_vector<thrust::complex<double>> wtransfer_h(oversampg*oversampg,{0.0,0.0});
    std::memcpy(wtransfer_h.data(),wtransfer.dp(),sizeof(std::complex<double>) * oversampg * oversampg);
    thrust::device_vector<thrust::complex<double>> wtransfer_d = wtransfer_h;
    
    // Work out range of W-Planes    
    double max_w = std::sin(theta/2) * std::sqrt(2*(grid_size*grid_size));
    int w_planes = std::ceil(max_w/dw) + aa_support_w + 1;

    std::cout << "Max W: " << max_w << "\n";
    std::cout << "W Planes: " << w_planes << "\n";

    vector2D<std::complex<double>> skyp(oversampg,oversampg,{0.0,0.0});
    generate_sky(points,skyp, theta, lam, du, dw, x0, grid_corr_lm, grid_corr_n);


    // We just copy into a thrust vector to make life easier from here on out.
    thrust::host_vector<thrust::complex<double>> skyp_h(oversampg*oversampg,{0.0,0.0});
    
    std::memcpy(skyp_h.data(),skyp.dp(),sizeof(std::complex<double>) * oversampg * oversampg);
    thrust::device_vector<thrust::complex<double>> skyp_d = skyp_h;    
    thrust::device_vector<thrust::complex<double>> wstacks(oversampg*oversampg,{0.0,0.0});
    
    // Setup FFT plan for our image/grid using cufft.

    cufftHandle plan;
    cuFFTError_check(cufftPlan2d(&plan,oversampg,oversampg,CUFFT_Z2Z));

    // FFT Shift our Sky and Fresnel Pattern
    fft_shift_kernel <thrust::complex<double>>
	<<< dimGrid, dimBlock >>>
	((thrust::complex<double>*)thrust::raw_pointer_cast(wtransfer_d.data()),
	oversampg);
    fft_shift_kernel <thrust::complex<double>>
	<<< dimGrid, dimBlock >>>
	((thrust::complex<double>*)thrust::raw_pointer_cast(skyp_d.data()),
	 oversampg);
    
    fresnel_sky_mul <double>
	<<< dimGrid, dimBlock >>>
	((thrust::complex<double>*)thrust::raw_pointer_cast(skyp_d.data()),
	 (thrust::complex<double>*)thrust::raw_pointer_cast(wtransfer_d.data()),
	 skyp_d.size(),
	 floor(-w_planes/2));

    
    for (int wplane = 0; wplane < w_planes; ++wplane){
	cuFFTError_check(cufftExecZ2Z(plan,
				      (cuDoubleComplex*)thrust::raw_pointer_cast(skyp_d.data()),
				      (cuDoubleComplex*)thrust::raw_pointer_cast(wstacks.data()),
				      CUFFT_FORWARD));

	fresnel_sky_mul <double>
	    <<< dimGrid, dimBlock >>>
	    ((thrust::complex<double>*)thrust::raw_pointer_cast(skyp_d.data()),
	     (thrust::complex<double>*)thrust::raw_pointer_cast(wtransfer_d.data()),
	     skyp_d.size(),
	     1);

	
    }
    
    std::complex<double> vis = {0.0,0.0};
    return vis;

}