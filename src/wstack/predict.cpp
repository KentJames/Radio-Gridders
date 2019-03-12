#include <iostream>
#include <iomanip>
#include <complex>
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <cmath>
#include <random>
#include <fftw3.h>

#include "wstack_common.h"
/*
  Predicts a visibility at a particular point using the direct fourier transform.
 */



/// My own 2D FFT Implementation, because I have no idea what is wrong.

// DIT, depth first, In-Place 1D FFT. Radix-2
void fft_1d_DIT_radix2(std::complex<double> *input, int fft_size){

  assert(fft_size && !(fft_size & (fft_size-1))); // Power of two check. Keep it simple..
  if(fft_size < 2) {
    //Bottom of the recursion pile.
  } else {
    
    //Seperate
      std::complex<double> *temp_odd = (std::complex<double>*)calloc(fft_size/2, sizeof(std::complex<double>));
    for(int i = 0; i < fft_size/2; ++i){
      temp_odd[i] = input[i*2 + 1];
    }
    for(int i = 0; i < fft_size/2; ++i){
      input[i] = input[i*2];
    }
    for(int i = 0; i < fft_size/2; ++i){
      input[i + fft_size/2] = temp_odd[i];
      
    }
    free(temp_odd);

    //Recurse
    fft_1d_DIT_radix2(input, fft_size/2);
    fft_1d_DIT_radix2(input + fft_size/2, fft_size/2);
    
    //Butterfly
    for(int i = 0; i < fft_size/2; ++i){
	std::complex<double> even = input[i];
	std::complex<double> odd =  input[i + fft_size/2];
	std::complex<double> ph = {0.0,-2 * PI<double> * i/fft_size};
	std::complex<double> twiddle = std::exp(ph);
	input[i] = even + twiddle * odd;
	input[i + fft_size/2] = even - twiddle * odd;
    }
  }	 
}

void fft_2d_DIT_radix2(vector2D<std::complex<double>>& array,
		       int xsize,
		       int ysize){

    std::complex<double> *dp = array.dp();

    // FFT Along X Directions
    for (int y = 0; y < ysize; ++y){
	std::complex<double> *fftp = dp + y * xsize;
	fft_1d_DIT_radix2(fftp, xsize);
    }

    array.transpose();

    // FFT Along X Directions
    for (int y = 0; y < ysize; ++y){
	std::complex<double> *fftp = dp + y * xsize;
	fft_1d_DIT_radix2(fftp, xsize);
    }

    
}


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
    
    std::complex<double> vis = {0.0,0.0};
    int npts = points.size()/2;
    for(int i = 0; i < npts; ++i){
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

std::vector<double> generate_random_points(int npts,
					   double theta){

    std::vector<double> points(2 * npts,0.0);
    std::srand ( time(NULL) );
    std::default_random_engine generator;
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

    for(int i = 0; i < points.size(); ++i){
	std::cout << points[i] << " ";
       
    }
    std::cout << "\n";
    return points;
}


vector2D<std::complex<double>> generate_fresnel(double theta,
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

    vector2D<std::complex<double>> wtransfer(oversampg,oversampg,{0.0,0.0});
    
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
		  vector2D<std::complex<double>>& sky, // We do this by reference because of FFTW planner.
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
    for (int i = 0; i < sky.size(); ++i){
	ft = fresnel(i);
	st = sky(i);

	
	
	if (t == 1){
	    sky(i) = st * ft;
	} else {
	    
	    if (ft == test) continue; // Otherwise std::pow goes a bit fruity
	    sky(i) = st  * std::pow(ft,t);
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

    size_t grid_sizex = array.d1s();
    size_t grid_sizey = array.d2s();
    
    assert(grid_sizex % 2 == 0);
    assert(grid_sizey % 2 == 0);
    
    for (int j = 0; j < grid_sizex; ++j){
	for (int i = 0; i < grid_sizey/2; ++i){
	    int ix0 = j * grid_sizex + i;
	    int ix1 = (ix0 + (grid_sizex + 1) * (grid_sizex/2)) % (grid_sizex * grid_sizey);

	    std::complex<double> temp = array(ix0);
	    array(ix0) = array(ix1);
	    array(ix1) = temp;
	}
    }

}

//TODO: There is a subtle bug raising error. Where is it...
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
				    struct sep_kernel_data *grid_corr_n){

    
    int grid_size = static_cast<int>(std::floor(theta * lam));
    double x0ih = std::round(0.5/x0);
    int oversampg = static_cast<int>(x0ih * grid_size);
    assert(oversampg > grid_size);
    int gd = (oversampg - grid_size)/2;
    
    // Fresnel Pattern
    
    vector2D<std::complex<double>> wtransfer = generate_fresnel(theta,lam,dw,x0);
    // Work out range of W-Planes
    
    double max_w = std::sin(theta/2) * std::sqrt(2*(grid_size*grid_size));
    int w_planes = std::ceil(max_w/dw) + aa_support_w + 1;

    std::cout << "Max W: " << max_w << "\n";
    std::cout << "W Planes: " << w_planes << "\n";
    
    
    // We double our grid size to get the optimal spacing.
    vector3D<std::complex<double>> wstacks(oversampg,oversampg,w_planes,{0.0,0.0});
    vector2D<std::complex<double>> skyp(oversampg,oversampg,{0.0,0.0});
    vector2D<std::complex<double>> plane(oversampg,oversampg,{0.0,0.0});
    fftw_plan plan;
    std::cout << "Planning fft's... " << std::flush;
    plan = fftw_plan_dft_2d(2*grid_size,2*grid_size,
    			    reinterpret_cast<fftw_complex*>(skyp.dp()),
     			    reinterpret_cast<fftw_complex*>(plane.dp()),
     			    FFTW_FORWARD,
     			    FFTW_MEASURE);  
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
    for(int i = 0; i < w_planes; ++i){

	std::cout << "Processing Plane: " << i << "\n";
	
	
	fftw_execute(plan);
	fft_shift_2Darray(plane);
	
	//Copy Plane into our stacks
	std::complex<double> *wp = wstacks.dp() + i*(4*grid_size*grid_size);
	std::complex<double> *pp = plane.dp();
	std::memcpy(wp,pp,sizeof(std::complex<double>) * (4*grid_size*grid_size));

	multiply_fresnel_pattern(wtransfer,skyp,1);
	std::cout << wstacks(2048,2048,i) << "\n";
	std::cout << predict_visibility_quantized(points,theta,lam,0.0,0.0,(i-std::floor(w_planes/2))*dw) << "\n";
	plane.clear();	


    }

    // std::cout << "############\n";
    // std::cout << std::setprecision(15);
    // std::cout << wstacks(2046,2048,5) << "\n";
    // std::cout << wstacks(2000,2096,5) << "\n";
    // std::cout << wstacks(2096,2096,5) << "\n";
    // std::cout << wstacks(2096,2000,5) << "\n";
    // std::cout << wstacks(2000,2000,5) << "\n";
    // std::cout << "############\n";
    // std::cout << predict_visibility_quantized(points,theta,lam,-10.0,0.0,dw) << "\n";
    // std::cout << predict_visibility_quantized(points,theta,lam,-240.0,240.0,dw) << "\n";
    // std::cout << predict_visibility_quantized(points,theta,lam,240.0,240.0,dw) << "\n";
    // std::cout << predict_visibility_quantized(points,theta,lam,240.0,-240.0,dw) << "\n";
    // std::cout << predict_visibility_quantized(points,theta,lam,-240.0,-240.0,dw) << "\n";
    

    // Begin De-convolution process using Sze-Tan Kernels.
    std::complex<double> vis_sze = {0.0,0.0};
    int oversampling = grid_conv_uv->oversampling;
    int oversampling_w = grid_conv_w->oversampling;
    int aa_h = std::floor(aa_support_uv/2);
    int aaw_h = std::floor(aa_support_w/2);
    for(int dui = -aa_h; dui < aa_h; ++dui){

	int dus = std::round(u/du) + grid_size + dui; 
	int aas_u = (dui+aa_h) * oversampling;
	
	for(int dvi = -aa_h; dvi < aa_h; ++dvi){

	    int dvs = std::round(v/du) + grid_size + dvi;
	    int aas_v = (dvi+aa_h) * oversampling;
	    
	    for(int dwi = -aaw_h; dwi < aaw_h; ++dwi){

		int dws = std::round(w/dw) + floor(w_planes/2) + dwi;
		int aas_w = (dwi+aaw_h) * oversampling_w;	
		
		double grid_convolution = 1.0 * 
		    grid_conv_uv->data[aas_u] *
		    grid_conv_uv->data[aas_v] *
		    grid_conv_w->data[aas_w];
		
		//std::cout << "Grid Convolution: " << grid_convolution << "\n";
		
		vis_sze += (wstacks(dus,dvs,dws) * grid_convolution);
		
	    }
	}
    }

    return vis_sze;
   
}


// // Why do my FFT's not match the DFT to 1 part in 10^6???
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
// 					 double x0,
// 					 struct sep_kernel_data *grid_conv_uv,
// 					 struct sep_kernel_data *grid_conv_w,
// 					 struct sep_kernel_data *grid_corr_lm,
// 					 struct sep_kernel_data *grid_corr_n){


//     int grid_size = std::floor(theta * lam);

//     vector2D<std::complex<double>> sky = generate_sky(points,
// 						      theta,
// 						      lam,
// 						      du,
// 						      dw,
// 						      x0,
// 						      grid_corr_lm,
// 						      grid_corr_n);

//     vector2D<std::complex<double>> planep(grid_size,grid_size,{0.0,0.0});
//     vector2D<std::complex<double>> skyp(grid_size,grid_size,{0.0,0.0});
//     fftw_plan plan2;
    
//     plan2 = fftw_plan_dft_2d(grid_size,grid_size,
//     			    reinterpret_cast<fftw_complex*>(skyp.dp()),
//     			     reinterpret_cast<fftw_complex*>(planep.dp()),
//      			    FFTW_FORWARD,
//      			    FFTW_ESTIMATE);
//     planep.clear();
//     skyp.clear();
//     fft_shift_2Darray(sky);
//     std::memcpy(skyp.dp(),sky.dp(),sizeof(std::complex<double>) * (grid_size*grid_size));
//     // fft_2d_DIT_radix2(skyp,
//     // 		      grid_size,
//     // 		      grid_size);

//     fftw_execute(plan2);
//     fft_shift_2Darray(skyp);
//     fft_shift_2Darray(planep);
    
//     std::cout << "############\n";
//     std::cout << std::setprecision(15);
//     std::cout << planep(1023,1024) << "\n";
//     std::cout << planep(1000,1048) << "\n";
//     std::cout << planep(1048,1048) << "\n";
//     std::cout << planep(1048,1000) << "\n";
//     std::cout << planep(1000,1000) << "\n";
//     std::cout << "############\n";
//     std::cout << predict_visibility_quantized(points,theta,lam,-10.0,0.0,0.0) << "\n";
//     std::cout << predict_visibility_quantized(points,theta,lam,-240.0,240.0,0.0) << "\n";
//     std::cout << predict_visibility_quantized(points,theta,lam,240.0,240.0,0.0) << "\n";
//     std::cout << predict_visibility_quantized(points,theta,lam,240.0,-240.0,0.0) << "\n";
//     std::cout << predict_visibility_quantized(points,theta,lam,-240.0,-240.0,0.0) << "\n";
    
//     return {0.0,0.0};
// }
