#include <iostream>
#include <iomanip>
#include <complex>
#include <cassert>
#include <cstring>
#include <vector>
#include <cmath>
#include <random>
#include <fftw3.h>

#include "wstack_common.cuh"
/*
  Predicts a visibility at a particular point using the direct fourier transform.
 */

const double PI  =3.141592653589793238463;
const float  PI_F=3.14159265358979f;

template <typename T>
class vector2D {
public:

    vector2D(size_t d1=0, size_t d2=0, T const & t=T()) :
        d1(d1), d2(d2), data(d1*d2, t)
    {}

    T & operator()(size_t i, size_t j) {
        return data[j*d1 + i];
    }

    T const & operator()(size_t i, size_t j) const {
        return data[j*d1 + i];
    }
    
    T & operator()(size_t i) {
        return data[i];
    }

    T const & operator()(size_t i) const {
        return data[i];
    }

    size_t size(){ return d1*d2; }
    size_t d1s(){ return d1; }
    size_t d2s(){ return d2; }
    T* dp(){ return data.data();}

    void clear(){
	std::fill(data.begin(), data.end(), 0);
    }

private:
    size_t d1,d2;
    std::vector<T> data;
};


template <typename T>
class vector3D {
public:
    vector3D(size_t d1=0, size_t d2=0, size_t d3=0, T const & t=T()) :
        d1(d1), d2(d2), d3(d3), data(d1*d2*d3, t)
    {}

    T & operator()(size_t i, size_t j, size_t k) {
        return data[k*d1*d2 + j*d1 + i];
    }

    T const & operator()(size_t i, size_t j, size_t k) const {
        return data[k*d1*d2 + j*d1 + i];
    }

    T & operator()(size_t i) {
        return data[i];
    }

    T const & operator()(size_t i) const {
        return data[i];
    }

    size_t size(){ return d1*d2*d3; }
    size_t d1s(){ return d1; }
    size_t d2s(){ return d2; }
    size_t d3s(){ return d3; }

    T* dp(){ return data.data();}

    void clear(){
	std::fill(data.begin(), data.end(), 0);
    }


private:
    size_t d1,d2,d3;
    std::vector<T> data;
};

std::complex<double> predict_visibility(std::vector<double> points,
					double u,
					double v,
					double w){

    std::complex<double> vis = (0.0,0.0);
    int npts = points.size()/2;
    for(int i = 0; i < npts; ++i){
	double l = points[2*i];
	double m = points[2*i + 1];
	double n = std::sqrt(1 - l*l - m*m) - 1.0;	
	
	std::complex<double> phase = {0,-2 * PI * (u*l + v*m + w*n)};
	std::complex<double> amp = {1.0,0.0};
	vis += amp * std::exp(phase);
	
    }
    return vis;
}

std::complex<double> predict_visibility_quantized(std::vector<double> points,
						  double theta,
						  double lam,
						  double u,
						  double v,
						  double w){

    int grid_size = std::floor(theta * lam);
    
    std::complex<double> vis = {0.0,0.0};
    double npts = points.size()/2;
    for(int i = 0; i < npts; ++i){
	double l = points[2*i];
	double m = points[2*i + 1];

	//Snap the l/m to a grid point
	int lc = std::floor((l/theta + 0.5) * grid_size);
	int mc = std::floor((m/theta + 0.5) * grid_size);

	double lq = (((double)lc - grid_size/2)/grid_size) * theta;
	double mq = (((double)mc - grid_size/2)/grid_size) * theta;	
	double n = std::sqrt(1.0 - lq*lq - mq*mq) - 1.0;
	
	std::complex<double> phase = {0,-2 * PI * (u*lq + v*mq + w*n)};
	std::complex<double> amp = {1.0,0.0};
	vis += amp * std::exp(phase);
	
    }
    return vis;
}


std::vector<double> generate_random_points(int npts,
					   double theta,
					   double lam){

    std::vector<double> points(2 * npts,0.0);
    
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-theta/2,theta/2);

    for(int i = 0; i < npts; ++i){
	points[2*i] = distribution(generator); // l
	points[2*i + 1] = distribution(generator); // m
    }

    return points;
}

//Quantises our sources onto the sky.
vector2D<std::complex<double>> generate_sky(std::vector<double> points,
					    double theta,
					    double lam,
					    double du,
					    double dw,
					    struct sep_kernel_data *grid_corr_lm,
					    struct sep_kernel_data *grid_corr_n){
    
    int grid_size = std::floor(theta * lam);
    vector2D<std::complex<double>> sky(grid_size, grid_size,{0.0,0.0});

    int npts = points.size()/2;

    for (int i = 0; i < npts; ++i){

	// Calculate co-ordinates
	double l = points[2*i];
	double m = points[2*i + 1];
	
	int lc = std::floor((l / theta + 0.5) * grid_size);
	int mc = std::floor((m / theta + 0.5) * grid_size);

	double lq = (((double)lc - grid_size/2)/grid_size) * theta;
	double mq = (((double)mc - grid_size/2)/grid_size) * theta;
	double n = std::sqrt(1.0 - lq*lq - mq*mq) - 1.0;

	// Calculate grid correction function

	int lm_size_t = grid_corr_lm->size * grid_corr_lm->oversampling;
	int n_size_t = grid_corr_n->size * grid_corr_n->oversampling;
	double lm_step = 1.0/(double)lm_size_t;
	double n_step = 1.0/(double)n_size_t; // Not too sure about this


	int aau = std::floor((du*lq)/lm_step + lm_size_t/2);
	int aav = std::floor((du*mq)/lm_step + lm_size_t/2);
	int aaw = std::floor((dw*n)/n_step + n_size_t/2);

	    
	double a = 1.0;
	a *= grid_corr_lm->data[aau];
	a *= grid_corr_lm->data[aav];
	a *= grid_corr_n->data[aaw];	

	std::complex<double> source = {1.0,0.0};
	source = source / a;
	sky(lc,mc) += source; // Sky needs to be transposed, not quite sure why.
    }
    
    return sky;
}

void multiply_fresnel_pattern(vector2D<std::complex<double>>& fresnel,
			      vector2D<std::complex<double>>& sky,
			      int t){
    
    std::complex<double> ft = {0.0,0.0};
    std::complex<double> st = {0.0,0.0};
    for (int i = 0; i < sky.size(); ++i){
	ft = fresnel(i);
	st = sky(i);
	if (t == 1){
	    sky(i) = st * ft;
	} else {
	    sky(i) = st  * std::pow(ft,t);
	}
    }
}


void zero_pad_2Darray(vector2D<std::complex<double>>& array,
		      vector2D<std::complex<double>>& padded_array){

    int i0 = padded_array.d1s()/4;
    int i1 = 3*(padded_array.d1s()/4);
    int j0 = padded_array.d2s()/4;
    int j1 = 3*(padded_array.d2s()/4);
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

//TODO: Put in x0 values instead of assuming x0=0.25
//TODO: Other forms of anti-aliasing functions.
//TODO: Oversampling.
//TODO: There is a subtle bug raising error. Where is it...
std::complex<double> wstack_predict(double theta,
				    double lam,
				    std::vector<double> points,
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
				    struct sep_kernel_data *grid_corr_n){

    int grid_size = floor(theta * lam);

    // Fresnel Pattern
    
    vector2D<std::complex<double>> wtransfer(grid_size,grid_size,{0.0,0.0});
    
    for (int y=0; y < grid_size; ++y){
	for (int x=0; x < grid_size; ++x){
	    double l = theta * (double)(x - grid_size / 2) / grid_size;
	    double m = theta * (double)(y - grid_size / 2) / grid_size;
	    double ph = dw * (1.0 - std::sqrt(1.0 - l*l - m*m));
	    
	    std::complex<double> wtrans = {0.0, 2 * PI * ph};
	    wtransfer(x,y) = std::exp(wtrans);
	}

	
    }
    // Work out range of W-Planes
    
    double max_w = std::sin(theta/2) * std::sqrt(2*(grid_size*grid_size));
    int w_planes = std::ceil(max_w/dw) + aa_support_w + 1;

    std::cout << "Max W: " << max_w << "\n";
    std::cout << "W Planes: " << w_planes << "\n";
    
    vector2D<std::complex<double>> sky = generate_sky(points,
						      theta,
						      lam,
						      du,
						      dw,
						      grid_corr_lm,
						      grid_corr_n);

    // We double our grid size to get the optimal spacing.
    vector3D<std::complex<double>> wstacks(2*grid_size,2*grid_size,w_planes,(0.0,0.0));
    vector2D<std::complex<double>> skyp(2*grid_size,2*grid_size,(0.0,0.0));
    vector2D<std::complex<double>> plane(2*grid_size,2*grid_size,(0.0,0.0));
    fftw_plan plan;
    
    plan = fftw_plan_dft_2d(2*grid_size,2*grid_size,
    			    reinterpret_cast<fftw_complex*>(skyp.dp()),
     			    reinterpret_cast<fftw_complex*>(plane.dp()),
     			    FFTW_FORWARD,
     			    FFTW_MEASURE);


    multiply_fresnel_pattern(wtransfer,sky,(std::floor(-w_planes/2)));

    std::cout << std::setprecision(15);
    for(int i = 0; i < w_planes; ++i){

	std::cout << "Processing Plane: " << i << "\n";;
	zero_pad_2Darray(sky,skyp);
	fft_shift_2Darray(skyp);
	fftw_execute(plan);
	fft_shift_2Darray(plane);
	
	//Copy Plane into our stacks
	std::complex<double> *wp = wstacks.dp() + i*(4*grid_size*grid_size);
	std::complex<double> *pp = plane.dp();
	std::memcpy(wp,pp,sizeof(std::complex<double>) * (4*grid_size*grid_size));

	multiply_fresnel_pattern(wtransfer,sky,1);
     
	skyp.clear();
	plane.clear();	


    }

    std::cout << "############\n";
    std::cout << std::setprecision(15);
    std::cout << wstacks(2000,2096,4) << "\n";
    std::cout << wstacks(2096,2096,4) << "\n";
    std::cout << wstacks(2096,2000,4) << "\n";
    std::cout << wstacks(2000,2000,4) << "\n";
    std::cout << predict_visibility_quantized(points,theta,lam,-240,240,0) << "\n";
    std::cout << predict_visibility_quantized(points,theta,lam,240,240,0) << "\n";
    std::cout << predict_visibility_quantized(points,theta,lam,240,-240,0) << "\n";
    std::cout << predict_visibility_quantized(points,theta,lam,-240,-240,0) << "\n";
    std::cout << wstacks(2048,2048,0) << "\n";
    std::cout << wstacks(2048,2048,1) << "\n";
    std::cout << wstacks(2048,2048,2) << "\n";
    std::cout << wstacks(2048,2048,3) << "\n";
    std::cout << wstacks(2048,2048,4) << "\n";
    std::cout << wstacks(2048,2048,5) << "\n";
    
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
