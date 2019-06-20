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



/*******************************************
           3D Convolution Kernels
********************************************/

//Do everyone a favor Nvidia and just template everything properly. It's 2019 ffs.
__device__ double cuda_floor(double x){ return floor(x);}
__device__ float cuda_floor(float x){ return floorf(x);}

__device__ double cuda_ceil(double x){ return ceil(x); }
__device__ float cuda_ceil(float x){ return ceilf(x); }



#define VIS_ACCUM_PER 4096 // Accumulate X visibilities at a time. 
template <typename FloatType>
__global__ void deconvolve_3D(thrust::complex<FloatType> *wstacks,
			      thrust::complex<FloatType> *vis,
			      FloatType *uvec,
			      FloatType *vvec,
			      FloatType *wvec,
			      FloatType *gcf_uv,
			      FloatType *gcf_w,
			      FloatType du,
			      FloatType dw,
			      std::size_t vis_num,
			      std::size_t aa_support_uv,
			      std::size_t aa_support_w,
			      std::size_t oversampling,
			      std::size_t oversampling_w,
			      std::size_t w_planes,
			      std::size_t grid_size){


    std::size_t x = blockIdx.x * blockDim.x + threadIdx.x;

      
    std::size_t aa_h = aa_support_uv/2;
    std::size_t aaw_h = aa_support_w/2;

    for(std::size_t i = 0; i < vis_num; i+= VIS_ACCUM_PER){

	FloatType u = uvec[i+x];
	FloatType v = vvec[i+x];
	FloatType w = wvec[i+x];
    
	FloatType flu = u - cuda_ceil(u/du)*du;
	FloatType flv = v - cuda_ceil(v/du)*du;
	FloatType flw = w - cuda_ceil(w/dw)*dw;

	std::size_t ovu = static_cast<std::size_t>(cuda_floor(abs(flu)/du * oversampling));
	std::size_t ovv = static_cast<std::size_t>(cuda_floor(abs(flv)/du * oversampling));
	std::size_t ovw = static_cast<std::size_t>(cuda_floor(abs(flw)/dw * oversampling));
    
	for(std::size_t dwi = -aaw_h; dwi < aaw_h; ++dwi){

	    std::size_t dws = static_cast<std::size_t>(cuda_ceil(w/dw)) +
		w_planes/2 + dwi;
	    std::size_t aas_w = aa_support_w * ovw + (dwi+aaw_h);
	    FloatType gridconv_w = gcf_w[aas_w];
	
	    for(std::size_t dvi = -aa_h; dvi < aa_h; ++dvi){
		
		std::size_t dvs = static_cast<std::size_t>(cuda_ceil(v/du)) + grid_size + dvi;
		std::size_t aas_v = aa_support_uv * ovv + (dvi + aa_h);
		FloatType gridconv_vw = gridconv_w * gcf_uv[aas_v];
	    
		for(std::size_t dui = -aa_h; dui < aa_h; ++dui){

		    std::size_t dus = static_cast<std::size_t>(cuda_ceil(u/du)) + grid_size + dui;
		    std::size_t aas_u = aa_support_uv * ovu + (dui + aa_h);
		    FloatType gridconv_uvw = gridconv_vw * gcf_uv[aas_u];
		    thrust::complex<FloatType> conv_point = wstacks[dws * grid_size * grid_size + dvs * grid_size + dus];
		    vis[i+x] += conv_point * gridconv_uvw;
		}
	    }
	}
	
    }


}
			      





			     
			      



/*******************************************
           2D Convolution Kernels
********************************************/



// First set of kernels are for doing individual visibility contribution processing.

template <typename FloatType>
__global__ void accumulate_cont(thrust::complex<FloatType> *grid,
				thrust::complex<FloatType> *accum_array,
				thrust::complex<FloatType> *pre_accum,
				struct vis_contribution <FloatType> *viscont,
				std::size_t supp_size,
				std::size_t grid_size){

    std::size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(x < supp_size && y < supp_size){
	pre_accum[ y * supp_size + x] =
	    grid[ viscont->v[y] * grid_size
		  + viscont->u[x]] *
	    viscont->gcf_u[x] * viscont->gcf_v[y];
    }
    
}

#define blockSize_reduc 1024
#define gridSize_reduc 24

template <typename FloatType>
__global__ void _reduce_to_vis(thrust::complex<FloatType> *pre_accum, //In
			      thrust::complex<FloatType> *accum_array, //Out
			      std::size_t accum_array_index,
			      std::size_t pre_accum_size){

    std::size_t idx = threadIdx.x;
    thrust::complex<FloatType> sum = 0;
    for (std::size_t i = idx; i < pre_accum_size; i += blockSize_reduc)
        sum += pre_accum[i];
    __shared__ thrust::complex<FloatType> r[blockSize_reduc];
    r[idx] = sum;
    __syncthreads();
    for (std::size_t size = blockSize_reduc/2; size>0; size/=2) { //uniform
        if (idx<size)
            r[idx] += r[idx+size];
        __syncthreads();
    }
    if (idx == 0)
        accum_array[accum_array_index] += r[0];
    
}

/* ---------------------------------------------------------------------------- */

// Second set of kernels are for doing the entire plane at once.

template <typename FloatType>
__global__ void accumulate_all_conts(thrust::complex<FloatType> *grid,
				      thrust::complex<FloatType> *pre_accum,
				      struct vis_contribution <FloatType> *viscont,
				      std::size_t supp_size,
				      std::size_t grid_size,
				      std::size_t num_contributions){

    std::size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    std::size_t z = blockIdx.z; // Used to index viscont
    
    
    if(x < supp_size && y < supp_size && z < num_contributions){
	pre_accum[ (z * supp_size * supp_size) +
		   y * supp_size + x] = grid[ viscont[z].locs_v[y] * grid_size
					      + viscont[z].locs_u[x]] *
	    viscont[z].gcf_u[x] *
	    viscont[z].gcf_v[y];
    }
}

template <typename FloatType>
__global__ void _reduce_all_to_vis(thrust::complex<FloatType> *pre_accum, //In
				    thrust::complex<FloatType> *accum_array, //Out
				    thrust::complex<FloatType> *r,
				    std::size_t *accum_array_index,
				    std::size_t pre_accum_size){

    std::size_t idx = threadIdx.x;
    thrust::complex<FloatType> sum = 0;
    for (std::size_t i = idx; i < pre_accum_size; i += blockSize_reduc)
        sum += pre_accum[blockIdx.x * pre_accum_size + i];
    r[blockIdx.x * pre_accum_size + idx] = sum;
    __syncthreads();
    for (std::size_t size = blockSize_reduc/2; size>0; size/=2) { //uniform
        if (idx<size)
            r[blockIdx.x * pre_accum_size + idx] += r[blockIdx.x * pre_accum_size + idx + size];
        __syncthreads();
    }
    if (idx == 0)
        accum_array[accum_array_index[blockIdx.x]] += r[blockIdx.x * pre_accum_size];

    
}


/* ---------------------------------------------------------------------------- */


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
__host__ std::vector<std::complex<double>> wstack_predict_cu_2D(double theta,
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

    // Deconvolution Grid Parameters
    int wstack_gs_reduc = 32;
    int wstack_vs_reduc = std::ceil(aa_support_uv/wstack_gs_reduc);
    
    for (int wplane = 0; wplane < w_planes; ++wplane){


	struct w_plane_locs <double> *wbin = &wbins[wplane];
	std::size_t contrn = wbin->num_contribs;
	thrust::device_vector<thrust::complex<double> > pre_accum(contrn * aa_support_uv * aa_support_uv,{0.0,0.0});

	dim3 dimGrid_reduc(wstack_vs_reduc,wstack_vs_reduc,contrn);
	dim3 dimBlock_reduc(wstack_gs_reduc,wstack_gs_reduc,contrn);
	thrust::device_vector<thrust::complex<double> > intermediate_reduction_arr(contrn * aa_support_uv * aa_support_uv,{0.0,0.0}); 	
    	cuFFTError_check(cufftExecZ2Z(plan,
    				      (cuDoubleComplex*)thrust::raw_pointer_cast(skyp_d.data()),
    				      (cuDoubleComplex*)thrust::raw_pointer_cast(wstacks.data()),
    				      CUFFT_FORWARD));
	fft_shift_kernel <thrust::complex<double>>
	    <<< dimGrid, dimBlock >>>
	    ((thrust::complex<double>*)thrust::raw_pointer_cast(wstacks.data()),
	     oversampg);

	accumulate_all_conts <double>
	    <<< dimGrid_reduc, dimBlock_reduc >>>
	    ((thrust::complex<double>*)thrust::raw_pointer_cast(wstacks.data()),
	     (thrust::complex<double>*)thrust::raw_pointer_cast(pre_accum.data()),
	     wbin->visc,
	     aa_support_uv,
	     oversampg,
	     wbin->num_contribs);
	    

	
    	fresnel_sky_mul <double>
    	    <<< dimGrid, dimBlock >>>
    	    ((thrust::complex<double>*)thrust::raw_pointer_cast(skyp_d.data()),
    	     (thrust::complex<double>*)thrust::raw_pointer_cast(wtransfer_d.data()),
    	     oversampg,
    	     1);

	
    }
    
    std::vector<std::complex<double> > vis(1,{0.0,0.0});
    return vis;

}


/* 
To optimise notes:
- Way more copies than I think is necessary but I am lazy and inexperienced with thrust.
- Deconvolution kernel is disgustingly inefficient, but that's v1.0 for you.
 */
 // We return std::complex to keep interface consistency with the C++ code.
__host__ std::vector<std::complex<double>> wstack_predict_cu_3D(double theta,
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
    
    std::size_t grid_size = static_cast<std::size_t>(std::floor(theta * lam));
    double x0ih = std::round(0.5/x0);
    std::size_t oversampg = static_cast<std::size_t>(x0ih * grid_size);
    assert(oversampg > grid_size);

    // CUDA Grid Parameters
    std::size_t wstack_gs = 32;
    std::size_t wstack_bs = grid_size/wstack_gs;
    dim3 dimGrid(wstack_bs,wstack_bs);
    dim3 dimBlock(wstack_gs,wstack_gs);
    

    // Fresnel Pattern
    
    vector2D<std::complex<double>> wtransfer = generate_fresnel(theta,lam,dw,x0);
    thrust::host_vector<thrust::complex<double>> wtransfer_h(oversampg*oversampg,{0.0,0.0});
    std::memcpy(wtransfer_h.data(),wtransfer.dp(),sizeof(std::complex<double>) * oversampg * oversampg);
    thrust::device_vector<thrust::complex<double>> wtransfer_d = wtransfer_h;
    
    // Work out range of W-Planes    
    double max_w = std::sin(theta/2) * std::sqrt(2*(grid_size*grid_size));
    std::size_t w_planes = std::ceil(max_w/dw) + aa_support_w + 1;

    std::cout << "Max W: " << max_w << "\n";
    std::cout << "W Planes: " << w_planes << "\n";


    std::cout << "Copying vis locs to GPU..." << std::flush;
    thrust::host_vector<double> uvec_h(u.size(),0.0);
    thrust::host_vector<double> vvec_h(v.size(),0.0);
    thrust::host_vector<double> wvec_h(w.size(),0.0);
    std::memcpy(uvec_h.data(),u.data(),sizeof(double) * u.size());
    std::memcpy(vvec_h.data(),v.data(),sizeof(double) * v.size());
    std::memcpy(wvec_h.data(),w.data(),sizeof(double) * w.size());    
    thrust::device_vector<double> uvec_d = uvec_h;
    thrust::device_vector<double> vvec_d = vvec_h;
    thrust::device_vector<double> wvec_d = wvec_h;
    
    //uvec_d = uvec_h; vvec_d = vvec_h; wvec_d = wvec_h;
    std::cout << "done\n";

    std::cout << "Copying convolution kernels to GPU..." << std::flush;
    std::size_t uv_conv_size = grid_conv_uv->oversampling * grid_conv_uv->size;
    std::size_t w_conv_size = grid_conv_w->oversampling * grid_conv_w->size;
    
    thrust::host_vector<double> gcf_uv_h(uv_conv_size, 0.0);
    thrust::host_vector<double> gcf_w_h(w_conv_size, 0.0);

    std::memcpy(gcf_uv_h.data(), grid_conv_uv->data, sizeof(double) * uv_conv_size);
    std::memcpy(gcf_w_h.data(), grid_conv_w->data, sizeof(double) * w_conv_size);
    thrust::device_vector<double> gcf_uv_d = gcf_uv_h;
    thrust::device_vector<double> gcf_w_d = gcf_w_h;
    std::cout << "done\n";

    

    std::cout << "Generating sky... " << std::flush;
    vector2D<std::complex<double>> skyp(oversampg,oversampg,{0.0,0.0});
    generate_sky(points,skyp, theta, lam, du, dw, x0, grid_corr_lm, grid_corr_n);
    

    // We just copy into a thrust vector to make life easier from here on out.
    thrust::host_vector<thrust::complex<double>> skyp_h(oversampg*oversampg,{0.0,0.0});
    
    std::memcpy(skyp_h.data(),skyp.dp(),sizeof(std::complex<double>) * oversampg * oversampg);
    thrust::device_vector<thrust::complex<double>> skyp_d = skyp_h;    
    thrust::device_vector<thrust::complex<double>> wstacks(w_planes*oversampg*oversampg,{0.0,0.0});
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
    				      (cuDoubleComplex*)thrust::raw_pointer_cast(&wstacks.data()[wplane * oversampg * oversampg]),
    				      CUFFT_FORWARD));
	fft_shift_kernel <thrust::complex<double>>
	    <<< dimGrid, dimBlock >>>
	    ((thrust::complex<double>*)thrust::raw_pointer_cast(&wstacks.data()[wplane * oversampg * oversampg]),
	     oversampg);
	
    	fresnel_sky_mul <double>
    	    <<< dimGrid, dimBlock >>>
    	    ((thrust::complex<double>*)thrust::raw_pointer_cast(skyp_d.data()),
    	     (thrust::complex<double>*)thrust::raw_pointer_cast(wtransfer_d.data()),
    	     oversampg,
    	     1);

	
    }

    
    deconvolve_3D<double> <<< 32, VIS_ACCUM_PER/32 >>>
	((thrust::complex<double> *)thrust::raw_pointer_cast(wstacks.data()),
	 (thrust::complex<double> *)thrust::raw_pointer_cast(visibilities_d.data()),
	 (double *)thrust::raw_pointer_cast(uvec_d.data()),
	 (double *)thrust::raw_pointer_cast(vvec_d.data()),
	 (double *)thrust::raw_pointer_cast(wvec_d.data()),
	 (double *)thrust::raw_pointer_cast(gcf_uv_d.data()),
	 (double *)thrust::raw_pointer_cast(gcf_w_d.data()),
	 du, dw,
	 u.size(),
	 static_cast<std::size_t>(grid_conv_uv->size),
	 static_cast<std::size_t>(grid_conv_w->size),
	 static_cast<std::size_t>(grid_conv_uv->oversampling),
	 static_cast<std::size_t>(grid_conv_w->oversampling),
	 w_planes,
	 grid_size);
		  
		  
		  
    
    visibilities_h = visibilities_d;
    std::vector<std::complex<double>> visr (visibilities_h.size(),{0.0,0.0});
    std::memcpy(visr.data(),visibilities_h.data(),sizeof(std::complex<double>) * visibilities_h.size());
    return visr;

}