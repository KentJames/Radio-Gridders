//C++ Includes
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cassert>
#include <complex>
#include <algorithm>

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
#include "predict.cuh"




//Elementwise multiplication of subimg with fresnel.
template <typename FloatType>
__global__ void fresnel_sky_mul(thrust::complex<FloatType> *sky,
				thrust::complex<FloatType> *fresnel,
				int n,
				int wp){

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(x < n && y < n){
	thrust::complex<FloatType> pown = {wp,0.0};
	thrust::complex<FloatType> wtrans = thrust::pow(fresnel[y * n + x], pown);
	thrust::complex<FloatType> skyv = sky[y*n + x];
	thrust::complex<FloatType> fresnelv = fresnel[y *n + x];
	thrust::complex<FloatType> test = {0.0, 0.0};
	
	if(wp == 1){
	    sky[y*n + x] = skyv;
	} else {
	    if (fresnelv == test){} else {
		sky[y*n + x] = skyv * thrust::pow(fresnelv,wp);
	    }
	}
    }
}


__host__ void bin_predictions_cu(double theta,
				 double lam,
				 std::vector <double> uvec,
				 std::vector <double> vvec,
				 std::vector <double> wvec,
				 double du,
				 double dw,
				 double w_min,
				 int aa_support_uv,
				 int aa_support_w,
				 struct w_plane_locs <double> *bins,
				 struct sep_kernel_data *grid_conv_uv,
				 struct sep_kernel_data *grid_conv_w){

    
    std::size_t grid_size = static_cast<std::size_t>(std::floor(theta * lam));
    // Work out range of W-Planes    
    double max_w = std::sin(theta/2) * std::sqrt(2*(grid_size*grid_size));
    int w_planes = std::ceil(max_w/dw) + aa_support_w + 1;

	
    for(std::size_t wp = 0; wp < w_planes; ++wp){
	
	struct w_plane_locs <double> *current_bin = &bins[wp];
	current_bin->wpi = wp;
	std::size_t contributions_in_plane = 0;

	// First do a pass over location vectors to work out how many contributions in the plane.

	for(std::size_t dp = 0; dp < wvec.size(); ++dp){

	    double w = wvec[dp];
	    std::size_t wpi = std::floor((w-w_min)/dw + 0.5) + aa_support_w/2;
	    std::size_t wp_min = wpi - aa_support_w/2;
	    std::size_t wp_max = wpi + aa_support_w/2;

	    if ((wp >= wp_min) && (wp < wp_max)) ++contributions_in_plane;

	}
	std::cout << "Contribs in Plane: " << contributions_in_plane <<"\n";

	
	if (contributions_in_plane == 0) continue; // Can't malloc a size of zero -> Things explode 
	cudaError_check(cudaMallocManaged((void**)&current_bin->visc,sizeof(struct vis_contribution <double>) * contributions_in_plane,cudaMemAttachGlobal));
	cudaError_check(cudaMallocManaged((void**)&current_bin->contrib_index,sizeof(struct vis_contribution <double>) * contributions_in_plane));
	current_bin->num_contribs = contributions_in_plane;
	std::size_t current_visn = 0;
	for(std::size_t dp = 0; dp < uvec.size(); ++dp){
    
	    double u = uvec[dp];
	    double v = vvec[dp];
	    double w = wvec[dp];
	    
	    std::size_t wpi = std::floor((w-w_min)/dw + 0.5) + aa_support_w/2;
	    std::size_t wp_min = wpi - aa_support_w/2;
	    std::size_t wp_max = wpi + aa_support_w/2;
	    

	    if ((wp >= wp_min) && (wp < wp_max)){
	    
		struct vis_contribution <double> *current_visc = &current_bin->visc[current_visn];
		current_bin->contrib_index[current_visn] = dp;
		//Malloc in our vis contributions
		cudaError_check(cudaMallocManaged((void **)&current_visc->locs_u,sizeof(int) * aa_support_uv));
		cudaError_check(cudaMallocManaged((void **)&current_visc->locs_v,sizeof(int) * aa_support_uv));
		cudaError_check(cudaMallocManaged((void **)&current_visc->gcf_u,sizeof(double) * aa_support_uv));
		cudaError_check(cudaMallocManaged((void **)&current_visc->gcf_v,sizeof(double) * aa_support_uv));

		// We can use the seperability (and squareness) of the kernel to save ourselves some space.
		std::size_t u_gp = std::floor(u/du) + grid_size/2;
		std::size_t v_gp = std::floor(v/du) + grid_size/2;
 
		// U/V/W oversample values
		int oversampling = grid_conv_uv->oversampling;
	    
		double flu = u - std::ceil(u/du)*du;
		double flv = v - std::ceil(v/du)*du;
		int ovu = static_cast<int>(std::floor(std::abs(flu)/du * oversampling));
		int ovv = static_cast<int>(std::floor(std::abs(flv)/du * oversampling));
	    
		for (std::size_t ul = 0; ul < aa_support_uv; ++ul) {

		    std::size_t aas_u = (ul * oversampling + ovu);
		    std::size_t aas_v = (ul * oversampling + ovv);
		
		    current_visc->locs_u[ul] = u_gp - aa_support_uv/2 + ul;
		    current_visc->locs_v[ul] = v_gp - aa_support_uv/2 + ul;
		    current_visc->gcf_u[ul] = grid_conv_uv->data[aas_u];
		    current_visc->gcf_v[ul] = grid_conv_uv->data[aas_v];
		
		}
		++current_visn;
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


    std::cout << "Generating sky... " << std::flush;
    vector2D<std::complex<double>> skyp(oversampg,oversampg,{0.0,0.0});
    generate_sky(points,skyp, theta, lam, du, dw, x0, grid_corr_lm, grid_corr_n);
    

    // We just copy into a thrust vector to make life easier from here on out.
    thrust::host_vector<thrust::complex<double>> skyp_h(oversampg*oversampg,{0.0,0.0});
    
    std::memcpy(skyp_h.data(),skyp.dp(),sizeof(std::complex<double>) * oversampg * oversampg);
    thrust::device_vector<thrust::complex<double>> skyp_d = skyp_h;    
    thrust::device_vector<thrust::complex<double>> wstacks(oversampg*oversampg,{0.0,0.0});
    std::cout << "done\n";

    
    // Setup FFT plan for our image/grid using cufft.
    std::cout << "Planning CUDA FFT's... " << std::flush;
    cufftHandle plan;
    cuFFTError_check(cufftPlan2d(&plan,oversampg,oversampg,CUFFT_Z2Z));
    std::cout << "done\n";
    
    // Bin our u/v/w prediction values in terms of w-plane contribution.
    double w_min = *std::min_element(std::begin(w), std::end(w));
    struct w_plane_locs <double> *wbins;
    cudaError_check(cudaMallocManaged((void **)&wbins, sizeof(struct w_plane_locs <double>) * w_planes));


    
    thrust::device_vector<thrust::complex<double> > visibilities_d(u.size(),{0.0,0.0});
    thrust::host_vector<thrust::complex<double> > visibilities_h;
    bin_predictions_cu(theta, lam, u, v, w, du, dw, w_min,
    		       aa_support_uv, aa_support_w, wbins,
    		       grid_conv_uv, grid_conv_w);
    

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
    	 oversampg,
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
    	     oversampg,
    	     1);

	
    }
    
    std::complex<double> vis = {0.0,0.0};
    return vis;

}