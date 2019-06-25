#include <iostream>
#include <iomanip>
#include <complex>
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>
#include <algorithm>
#include <fftw3.h>
#include <omp.h>

#include "wstack_common.h"

// We use non-unit strides to alleviate cache thrashing effects.
    const int element_stride = 1;
    const int row_stride = 8;
    const int matrix_stride = 10;

/*
  Predicts a visibility at a particular point using the direct fourier transform.
 */

std::complex<double> predict_visibility(const std::vector<double>& points,
					double u,
					double v,
					double w){

    std::complex<double> vis = {0.0,0.0};
    int npts = points.size()/2;
    for(int i = 0; i < npts; ++i){
	double l = points[2*i];
	double m = points[2*i + 1];
	double n = std::sqrt(1 - l*l - m*m) - 1.0;	
	
	std::complex<double> phase = {0,-2 * PI<double> * (u*l + v*m + w*n)};
	std::complex<double> amp = {1.0,0.0};
	vis += amp * std::exp(phase);
	
    }
    return vis;
}

std::complex<double> predict_visibility_quantized(const std::vector<double>& points,
								double theta,
								double lam,
								double u,
								double v,
								double w){

    double grid_size = std::floor(theta * lam);
    
    std::complex<double> vis {0.0,0.0};
    std::size_t npts = points.size()/2;

    for(std::size_t i = 0; i < npts; ++i){
	double l = points[2*i];
	double m = points[2*i + 1];
	
	//Snap the l/m to a grid point
	double lc = std::floor((l / theta + 0.5) * (double)grid_size);
	double mc = std::floor((m / theta + 0.5) * (double)grid_size);
	std::cout << std::setprecision(15);
	
	//int lc = (int)std::floor(l / theta * (double)grid_size) + grid_size/2;
	//int mc = (int)std::floor(m / theta * (double)grid_size) + grid_size/2;
	//double lq = theta * ((lc - (double)grid_size/2)/(double)grid_size);
	//double mq = theta * ((mc - (double)grid_size/2)/(double)grid_size);
	double lq = (double)lc/lam - theta/2;
	double mq = (double)mc/lam - theta/2;
	
	
	// double lq = theta * (((double)lc/(double)grid_size) - 0.5);
	// double mq = theta * (((double)mc/(double)grid_size) - 0.5);
	double n = std::sqrt(1.0 - lq*lq - mq*mq) - 1.0;
	
	std::complex<double> phase = {0,-2 * PI<double> * (u*lq + v*mq + w*n)};
	
	vis += 1.0 * std::exp(phase);
    }

    return vis;
}

std::vector<std::complex<double> > predict_visibility_quantized_vec(const std::vector<double>& points,
								double theta,
								double lam,
								std::vector<double> uvw){

    double grid_size = std::floor(theta * lam);
    
    std::vector<std::complex<double> > vis (uvw.size(),{0.0,0.0});
    std::size_t npts = points.size()/2;


    //#pragma omp parallel
    for (std::size_t visi = 0; visi < uvw.size(); ++visi){
	
	double u = uvw[3 * visi + 0];
	double v = uvw[3 * visi + 1];
	double w = uvw[3 * visi + 2];
	for(std::size_t i = 0; i < npts; ++i){
	    double l = points[2*i];
	    double m = points[2*i + 1];

	    //Snap the l/m to a grid point
	    double lc = std::floor((l / theta + 0.5) * (double)grid_size);
	    double mc = std::floor((m / theta + 0.5) * (double)grid_size);
	    std::cout << std::setprecision(15);

	    //int lc = (int)std::floor(l / theta * (double)grid_size) + grid_size/2;
	    //int mc = (int)std::floor(m / theta * (double)grid_size) + grid_size/2;
	    //double lq = theta * ((lc - (double)grid_size/2)/(double)grid_size);
	    //double mq = theta * ((mc - (double)grid_size/2)/(double)grid_size);
	    double lq = (double)lc/lam - theta/2;
	    double mq = (double)mc/lam - theta/2;

	
	    // double lq = theta * (((double)lc/(double)grid_size) - 0.5);
	    // double mq = theta * (((double)mc/(double)grid_size) - 0.5);
	    double n = std::sqrt(1.0 - lq*lq - mq*mq) - 1.0;

	    std::complex<double> phase = {0,-2 * PI<double> * (u*lq + v*mq + w*n)};

	    vis[visi] += 1.0 * std::exp(phase);
	}
    }
    return vis;
}

std::vector<double> generate_random_points(int npts,
					   double theta){

    std::vector<double> points(2 * npts,0.0);

    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator;
    generator.seed(seed);
    std::uniform_real_distribution<double> distribution(-theta/2,theta/2);

    for(int i = 0; i < npts; ++i){
	points[2*i] = distribution(generator); // l
	points[2*i + 1] = distribution(generator); // m
    }

    return points;
}

std::vector<double> generate_testcard_dataset(double theta){

    std::vector<double> points = {0.95,0.95,-0.95,-0.95,0.95,-0.95,-0.95,0.95,0.0,0.5,0.0,-0.5,0.5,0.0,-0.5,0.0};
    std::transform(points.begin(), points.end(), points.begin(),
		   [theta](double c) -> double { return c * (theta/2);});

    for(std::size_t i = 0; i < points.size(); ++i){
	std::cout << points[i] << " ";
       
    }
    std::cout << "\n";
    return points;
}

std::vector<double> generate_testcard_dataset_simple(double theta){

    std::vector<double> points = {0.0,0.0};
    std::transform(points.begin(), points.end(), points.begin(),
		   [theta](double c) -> double { return c * (theta/2);});

    for(std::size_t i = 0; i < points.size(); ++i){
	std::cout << points[i] << " ";
       
    }
    std::cout << "\n";
    return points;
}



vector2D<std::complex<double> > generate_fresnel(double theta,
						double lam,
						double dw,
						double x0){

    int grid_size = static_cast<int>(std::floor(theta * lam));
    double x0ih = std::round(0.5/x0);
    int oversampg = static_cast<int>(x0ih * grid_size);
    assert(oversampg > grid_size);
    int gd = (oversampg - grid_size)/2;

    std::cout << "xoih: " << x0ih << "\n";
    std::cout << "oversampg: " << oversampg << "\n";
    std::cout << "gd: " << gd << "\n";

    vector2D<std::complex<double> > wtransfer(oversampg,oversampg,{0.0,0.0});
    
    for (int y=0; y < grid_size; ++y){
	for (int x=0; x < grid_size; ++x){
	    double l = theta * ((double)x - grid_size / 2) / grid_size;
	    double m = theta * ((double)y - grid_size / 2) / grid_size;
	    double ph = dw * (1.0 - std::sqrt(1.0 - l*l - m*m));
	    
	    std::complex<double> wtrans = {0.0, 2 * PI<double> * ph};
	    int xo = x+gd;
	    int yo = y+gd;
	    wtransfer(xo,yo) = std::exp(wtrans);
	}

	
    }

    return wtransfer;
}


//Quantises our sources onto the sky.
void generate_sky(const std::vector<double>& points,
		  vector2D<std::complex<double> >& sky, // We do this by reference because of FFTW planner.
		  double theta,
		  double lam,
		  double du,
		  double dw,
		  double x0,
		  struct sep_kernel_data *grid_corr_lm,
		  struct sep_kernel_data *grid_corr_n){
    
    
    int grid_size = static_cast<int>(std::floor(theta * lam));
    double x0ih = 0.5/x0;
    int oversampg = static_cast<int>(std::round(x0ih * grid_size));
    assert(oversampg > grid_size);
    int gd = (oversampg - grid_size)/2;

    int npts = points.size()/2;

    for (int i = 0; i < npts; ++i){

	// Calculate co-ordinates
	double l = points[2*i];
	double m = points[2*i + 1];

	int lc = static_cast<int>(std::floor((l / theta + 0.5) *
					     static_cast<double>(grid_size)));
        int mc = static_cast<int>(std::floor((m / theta + 0.5) *
					     static_cast<double>(grid_size)));
	// double lq = theta * ((static_cast<double>(lc) - (double)grid_size/2)/(double)grid_size);
	// double mq = theta * ((static_cast<double>(mc) - (double)grid_size/2)/(double)grid_size);

	double lq = (double)lc/lam - theta/2;
	double mq = (double)mc/lam - theta/2;
	double n = std::sqrt(1.0 - lq*lq - mq*mq) - 1.0;
	// // Calculate grid correction function

	int lm_size_t = grid_corr_lm->size * grid_corr_lm->oversampling;
	int n_size_t = grid_corr_n->size * grid_corr_n->oversampling;
	double lm_step = 1.0/(double)lm_size_t;
	double n_step = 1.0/(double)n_size_t; //Not too sure about this

	int aau = std::floor((du*lq)/lm_step) + lm_size_t/2;
	int aav = std::floor((du*mq)/lm_step) + lm_size_t/2;
	int aaw = std::floor((dw*n)/n_step) + n_size_t/2;
	
	double a = 1.0;
	a *= grid_corr_lm->data[aau];
	a *= grid_corr_lm->data[aav];
	a *= grid_corr_n->data[aaw];

	std::complex<double> source = {1.0,0.0};
	source = source / a;
	int lco = gd+lc;
	int mco = gd+mc;
	sky(lco,mco) += source; // Sky needs to be transposed, not quite sure why.
    }
    
}


void multiply_fresnel_pattern(vector2D<std::complex<double>>& fresnel,
			      vector2D<std::complex<double>>& sky,
			      int t){
    assert(fresnel.size() == sky.size());
    std::complex<double> ft = {0.0,0.0};
    std::complex<double> st = {0.0,0.0};
    std::complex<double> test = {0.0,0.0};

    size_t grid_sizex = fresnel.d1s();
    size_t grid_sizey = fresnel.d2s();
    
    for (std::size_t j = 0; j < grid_sizey; ++j){
	for (std::size_t i = 0; i < grid_sizex; ++i){
	    ft = fresnel(i,j);
	    st = sky(i,j);	
	
	    if (t == 1){
		sky(i,j) = st * ft;
	    } else {
	    
		if (ft == test) continue; // Otherwise std::pow goes a bit fruity
		sky(i,j) = st  * std::pow(ft,t);
	    }
	}
    }
}


void zero_pad_2Darray(const vector2D<std::complex<double>>& array,
		      vector2D<std::complex<double>>& padded_array,
		      double x0){

    int x0i = static_cast<int>(std::round(1.0/x0));
    std::cout << "xoi: " << x0i << "\n";
    int i0 = padded_array.d1s()/x0i;
    int i1 = 3*(padded_array.d1s()/x0i);
    int j0 = padded_array.d2s()/x0i;
    int j1 = 3*(padded_array.d2s()/x0i);
    for(int j = j0; j < j1; ++j){
	for(int i = i0; i < i1; ++i){
	    padded_array(i,j) = array(i-i0,j-j0);
	}
    }
}

// Stealing Peters code has become the hallmark of my PhD.
void fft_shift_2Darray(vector2D<std::complex<double>>& array){

    std::size_t grid_sizex = array.d1s();
    std::size_t grid_sizey = array.d2s();
    
    assert(grid_sizex % 2 == 0);
    assert(grid_sizey % 2 == 0);
    int i1,j1;
    for (std::size_t j = 0; j < grid_sizex; ++j){
	for (std::size_t i = 0; i < grid_sizey/2; ++i){
	    // int ix0 = j * grid_sizex + i;
	    // int ix1 = (ix0 + (grid_sizex + 1) * (grid_sizex/2)) % (grid_sizex * grid_sizey);

	    i1 = i + grid_sizex/2;
	    if (j < grid_sizey/2){
		j1 = j + grid_sizey/2;
	    } else {
		j1 = j - grid_sizey/2;
	    }
	   
	    std::complex<double> temp = array(i,j);
	    array(i,j) = array(i1,j1);
	    array(i1,j1) = temp;
	}
    }
}


inline void memcpy_plane_to_stack(vector2D<std::complex<double>>&plane,
			   vector3D<std::complex<double>>&stacks,
			   std::size_t grid_size,
			   std::size_t planei){

    //Calculate memory copy amount based on striding information.
    //Assume strides for n=1 and n=2 dimensions are the same between
    //stacks and plane. If they aren't then can't use memcpy directly.

    std::size_t p1s,p2s,s1s,s2s,s3s,p1d,p2d,s1d,s2d;


    p1d = plane.d1s();
    p2d = plane.d2s();
    s1d = stacks.d1s();
    s2d = stacks.d2s();

    // Make sure dimensions are the same
    assert(p1d == s1d);
    assert(p2d == s2d);
   
    
    p1s = plane.s1s();
    p2s = plane.s2s();
    s1s = stacks.s1s();
    s2s = stacks.s2s();
    s3s = stacks.s3s();
    
    // Let us really make sure the strides are the same
    assert(p1s == s1s);
    assert(p2s = s2s);

    

    std::size_t copy_size = (p1d*p1s + p2s) * p2d * sizeof(std::complex<double>); 

    std::complex<double> *wp = stacks.pp(planei);  
    std::complex<double> *pp = plane.dp();
    std::memcpy(wp,pp,copy_size);

}
			   


std::vector<std::complex<double>> wstack_predict(double theta,
						 double lam,
						 const std::vector<double>& points, // Sky points
						 std::vector<double> uvwvec, // U/V/W points to predict.
						 double du, // Sze-Tan Optimum Spacing in U/V
						 double dw, // Sze-Tan Optimum Spacing in W
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
    //int gd = (oversampg - grid_size)/2;


    
    
    // Fresnel Pattern
    
    vector2D<std::complex<double>> wtransfer = generate_fresnel(theta,lam,dw,x0);
    // Work out range of W-Planes
    
    double max_w = std::sin(theta/2) * std::sqrt(2*(grid_size*grid_size));
    int w_planes = std::ceil(max_w/dw) + aa_support_w + 1;

    std::cout << "Max W: " << max_w << "\n";
    std::cout << "W Planes: " << w_planes << "\n";
    
    
    // We double our grid size to get the optimal spacing.
    vector3D<std::complex<double> > wstacks(oversampg,oversampg,w_planes,{0.0,0.0},element_stride,row_stride,matrix_stride);
    vector2D<std::complex<double> > skyp(oversampg,oversampg,{0.0,0.0},element_stride,row_stride);
    vector2D<std::complex<double> > plane(oversampg,oversampg,{0.0,0.0},element_stride,row_stride);
    fftw_plan plan;
    std::cout << "Planning fft's... " << std::flush;

    fftw_init_threads();
    fftw_plan_with_nthreads(omp_get_max_threads());    
    fftw_import_wisdom_from_filename("fftw.wisdom");

    fftw_iodim *iodims_plane = (fftw_iodim *)malloc(2*sizeof(fftw_iodim));
    fftw_iodim *iodims_howmany = (fftw_iodim *)malloc(sizeof(fftw_iodim));
    // Setup row dims
    iodims_plane[0].n = 2*grid_size;
    iodims_plane[0].is = 1; // Keep row elements contiguous (for now)
    iodims_plane[0].os = iodims_plane[0].is;

    // Setup matrix dims
    iodims_plane[1].n = 2*grid_size;
    iodims_plane[1].is = row_stride  + 2*grid_size*element_stride;
    iodims_plane[1].os = iodims_plane[1].is;

    // Give a unit howmany rank dimensions
    iodims_howmany[0].n = 1;
    iodims_howmany[0].is = 1;
    iodims_howmany[0].os = 1;
    
    // I'm a big boy now. Guru mode fft's~~
    plan = fftw_plan_guru_dft(2, iodims_plane, 1, iodims_howmany,
			      reinterpret_cast<fftw_complex*>(skyp.dp()),
			      reinterpret_cast<fftw_complex*>(plane.dp()),
			      FFTW_FORWARD,
			      FFTW_MEASURE);
 
    fftw_export_wisdom_to_filename("fftw.wisdom");
    std::cout << "done\n" << std::flush;
    skyp.clear();
    plane.clear();
    std::cout << "Generating sky... " << std::flush;
    generate_sky(points,skyp,theta,lam,du,dw,x0,grid_corr_lm,grid_corr_n);
    std::cout << "Sky: " << skyp(grid_size,grid_size) << "\n";
    std::cout << "done\n" << std::flush;
    fft_shift_2Darray(skyp);
    fft_shift_2Darray(wtransfer);
    multiply_fresnel_pattern(wtransfer,skyp,(std::floor(-w_planes/2)));

    std::cout << "W-Stacker: \n";
    std::cout << std::setprecision(15);

    long flops_per_vis = 6 * aa_support_uv * aa_support_uv * aa_support_w; // 3D Deconvolve
    flops_per_vis += 3 * aa_support_uv * aa_support_uv * aa_support_w; // Compute seperable kernel
    long total_flops = flops_per_vis * uvwvec.size()/3;

    std::chrono::high_resolution_clock::time_point t1_ws = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < w_planes; ++i){

	std::cout << "Processing Plane: " << i << "\n";	
	fftw_execute(plan);
	fft_shift_2Darray(plane);
	memcpy_plane_to_stack(plane,
			      wstacks,
			      grid_size,
			      i);
	multiply_fresnel_pattern(wtransfer,skyp,1);
	plane.clear();
	
    }
    std::chrono::high_resolution_clock::time_point t2_ws = std::chrono::high_resolution_clock::now();
    auto duration_ws = std::chrono::duration_cast<std::chrono::milliseconds>( t2_ws - t1_ws ).count();
    
    std::cout << "W-Stack Time: " << duration_ws << "ms \n";
   

    std::vector<std::complex<double> > visibilities(uvwvec.size()/3,{0.0,0.0});
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
#pragma omp parallel
#pragma omp for schedule(static,1000)
    for (std::size_t i = 0; i < uvwvec.size()/3; ++i){

    	visibilities[i] = deconvolve_visibility_(uvwvec[3*i + 0],
						uvwvec[3*i + 1],
						uvwvec[3*i + 2],
						du,
						dw,
						aa_support_uv,
						aa_support_w,
						grid_conv_uv->oversampling,
					        grid_conv_w->oversampling,
						w_planes,
						grid_size,
						wstacks,
						grid_conv_uv,
						grid_conv_w);
    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
    float duration_s = static_cast<float>(duration)/1000;
    float gflops = static_cast<float>(total_flops) / duration_s;
    std::cout << "Deconvolve Time: " << duration << "ms \n";
    std::cout << "GFLOP/s: " << gflops << "\n";
    
    //Unfortunately LLVM and GCC are woefully behind Microsoft when it comes to parallel algorithm support in the STL!!

    // std::transform(std::execution::par, uvwvec.begin(), uvwvec.end(), visibilities.begin(),
    // 		   [du,
    // 		    dw,
    // 		    aa_support_uv,
    // 		    aa_support_w,
    // 		    w_planes,
    // 		    grid_size,
    // 		    wstacks,
    // 		    grid_conv_uv,
    // 		    grid_conv_w](std::vector<double> uvw) -> std::complex<double>
    // 		   {return deconvolve_visibility(uvw,
    // 						 du,
    // 						 dw,
    // 						 aa_support_uv,
    // 						 aa_support_w,
    // 						 w_planes,
    // 						 grid_size,
    // 						 wstacks,
    // 						 grid_conv_uv,
    // 						 grid_conv_w);
    // 		   });
     
		   

    
    return visibilities;
}


//Takes lines of visibilities.
std::vector<std::vector<std::complex<double>>> wstack_predict_lines(double theta,
						       double lam,
						       const std::vector<double>& points, // Sky points
						       std::vector<std::vector<double>> uvwvec, // U/V/W points to predict.
						       double du, // Sze-Tan Optimum Spacing in U/V
						       double dw, // Sze-Tan Optimum Spacing in W
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
    //int gd = (oversampg - grid_size)/2;

    // Fresnel Pattern
    
    vector2D<std::complex<double>> wtransfer = generate_fresnel(theta,lam,dw,x0);
    // Work out range of W-Planes
    
    double max_w = std::sin(theta/2) * std::sqrt(2*(grid_size*grid_size));
    int w_planes = std::ceil(max_w/dw) + aa_support_w + 1;

    std::cout << "Max W: " << max_w << "\n";
    std::cout << "W Planes: " << w_planes << "\n";
       
    // We double our grid size to get the optimal spacing.
    vector3D<std::complex<double> > wstacks(oversampg,oversampg,w_planes,{0.0,0.0}, element_stride, row_stride, matrix_stride);
    vector2D<std::complex<double> > skyp(oversampg,oversampg,{0.0,0.0}, element_stride, row_stride);
    vector2D<std::complex<double> > plane(oversampg,oversampg,{0.0,0.0}, element_stride, row_stride);
    fftw_plan plan;
    std::cout << "Planning fft's... " << std::flush;
    fftw_init_threads();
    fftw_plan_with_nthreads(omp_get_max_threads());
    fftw_import_wisdom_from_filename("fftw_l.wisdom");

    fftw_iodim *iodims_plane = (fftw_iodim *)malloc(2*sizeof(fftw_iodim));
    fftw_iodim *iodims_howmany = (fftw_iodim *)malloc(sizeof(fftw_iodim));
    // Setup row dims
    iodims_plane[0].n = 2*grid_size;
    iodims_plane[0].is = 1; // Keep row elements contiguous (for now)
    iodims_plane[0].os = iodims_plane[0].is;

    // Setup matrix dims
    iodims_plane[1].n = 2*grid_size;
    iodims_plane[1].is = row_stride  + 2*grid_size*element_stride;
    iodims_plane[1].os = iodims_plane[1].is;

    // Give a unit howmany rank dimensions
    iodims_howmany[0].n = 1;
    iodims_howmany[0].is = 1;
    iodims_howmany[0].os = 1;
    
    // I'm a big boy now. Guru mode fft's~~
    plan = fftw_plan_guru_dft(2, iodims_plane, 1, iodims_howmany,
			      reinterpret_cast<fftw_complex*>(skyp.dp()),
			      reinterpret_cast<fftw_complex*>(plane.dp()),
			      FFTW_FORWARD,
			      FFTW_MEASURE);
;
    fftw_export_wisdom_to_filename("fftw_l.wisdom");
    std::cout << "done\n" << std::flush;
    skyp.clear();
    plane.clear();
    std::cout << "Generating sky... " << std::flush;
    generate_sky(points,skyp,theta,lam,du,dw,x0,grid_corr_lm,grid_corr_n);    
    std::cout << "done\n" << std::flush;
    fft_shift_2Darray(skyp);
    fft_shift_2Darray(wtransfer);
    multiply_fresnel_pattern(wtransfer,skyp,(std::floor(-w_planes/2)));

    std::cout << "W-Stacker: \n";
    std::cout << std::setprecision(15);

    std::chrono::high_resolution_clock::time_point t1_ws = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < w_planes; ++i){

	std::cout << "Processing Plane: " << i << "\n";
	fftw_execute(plan);
	fft_shift_2Darray(plane);
	memcpy_plane_to_stack(plane,
			      wstacks,
			      grid_size,
			      i);
	multiply_fresnel_pattern(wtransfer,skyp,1);
	plane.clear();	
    }
    std::chrono::high_resolution_clock::time_point t2_ws = std::chrono::high_resolution_clock::now();
    auto duration_ws = std::chrono::duration_cast<std::chrono::milliseconds>( t2_ws - t1_ws ).count();
    std::cout << "W-Stack Time: " << duration_ws << "ms \n";;
    std::cout << " UVW Vec Size: " << uvwvec.size() << "\n";
    std::vector<std::vector<std::complex<double> > > visibilities(uvwvec.size());

    // To make threads play nice, pre-initialise
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel
    #pragma omp for schedule(dynamic)
    for (std::size_t line = 0; line < uvwvec.size(); ++ line){
	visibilities[line].resize(uvwvec[line].size()/3,0.0);
	for (std::size_t i = 0; i < uvwvec[line].size()/3; ++i){

	    visibilities[line][i] = deconvolve_visibility_(uvwvec[line][3*i + 0],
							  uvwvec[line][3*i + 1],
							  uvwvec[line][3*i + 2],
							  du,
							  dw,
							  aa_support_uv,
							  aa_support_w,
							  grid_conv_uv->oversampling,
							  grid_conv_w->oversampling,
							  w_planes,
							  grid_size,
							  wstacks,
							  grid_conv_uv,
							  grid_conv_w);
	}
    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
    
    std::cout << "Deconvolve Time: " << duration << "ms \n";;
   
    /*
    //Unfortunately LLVM and GCC are woefully behind Microsoft when it comes to parallel algorithm support in the STL!!

    std::transform(std::execution:par, uvwvec.begin(), uvwvec.end(), visibilities.begin(),
		   [du,
		    dw,
		    aa_support_uv,
		    aa_support_w,
		    w_planes,
		    grid_size,
		    wstacks,
		    grid_conv_uv,
		    grid_conv_w](std::vector<double> uvw) -> std::complex<double>
		   {return deconvolve_visibility(uvw,
						 du,
						 dw,
						 aa_support_uv,
						 aa_support_w,
						 w_planes,
						 grid_size,
						 wstacks,
						 grid_conv_uv,
						 grid_conv_w);
		   });
    */ 
		   

    
    return visibilities;
}

