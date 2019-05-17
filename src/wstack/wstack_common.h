
#include "hdf5.cuh"

#ifndef WSTACK_H
#define WSTACK_H

#define THREADS_BLOCK 16

#include <vector>
#include <complex>
#include <chrono>
#include <algorithm>
#include <random>

template<class T>
constexpr T PI = T(3.1415926535897932385L);

template <typename T>
class vector2D {
public:

    vector2D(size_t d1=0, size_t d2=0, T const & t=T(), size_t s1 = 0, size_t s2 = 0) :
        d1(d1+s2), d2(d2), s1(s1), s2(s2), data(d1*d2 + d1*s2, t)
    {}

    ~vector2D(){
	
    }
    
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

    void transpose(){

	std::vector<T> datat(d1*d2,0);
	for(int j = 0; j < d1; ++j){
	    for(int i = 0; i < d2; ++i){
		datat[i*d2 + j] = data[j*d1 + i];
	    }
	}

	data = datat;
    }

private:
    size_t d1,d2; // Data Size
    size_t s1,s2; // Stride
    std::vector<T> data;
};


template <typename T>
class vector3D {
public:
    vector3D(size_t d1=0, size_t d2=0, size_t d3=0,
	     T const & t=T(), size_t s1=0, size_t s2=0, size_t s3=0) :
        d1(d1+s2), d2(d2+s3), d3(d3),
	s1(s1), s2(s2), s3(s3), data(d1*d2*d3 + d1*s2 + d2*s3, t)
    {}

    ~vector3D(){
	
    }

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

    void fill_random(){
	auto seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator;
	generator.seed(seed);
	std::uniform_real_distribution<double> distribution(0,1);
	auto gen = [&distribution, &generator](){
                   return distribution(generator);
               };
	std::generate(std::begin(data), std::end(data), gen);

    }

    
private:
    size_t d1,d2,d3;
    size_t s1,s2,s3;
    std::vector<T> data;
};


// Prototypes for generating visibilities

static inline std::vector<std::vector<double>> generate_random_visibilities(double theta,
							      double lambda,
							      double dw,
							      int npts){

    
    std::vector<std::vector<double>> vis(npts, std::vector<double>(3,0.0));
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator;
    generator.seed(seed);
    std::uniform_real_distribution<double> distribution(-theta*lambda/2,theta*lambda/2);
    std::uniform_real_distribution<double> distribution_w(-dw,dw);
   
    for(int i = 0; i < npts; ++i){	
	vis[i][0] = distribution(generator);
	vis[i][1] = distribution(generator);
	vis[i][2] = distribution_w(generator);
    }

    return vis;
}

static inline std::vector<double> generate_random_visibilities_(double theta,
							      double lambda,
							      double dw,
							      int npts){

    
    std::vector<double> vis(3*npts, 0.0);
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator;
    generator.seed(seed);
    std::uniform_real_distribution<double> distribution(-theta*lambda/2,theta*lambda/2);
    std::uniform_real_distribution<double> distribution_w(-dw,dw);
   
    for(int i = 0; i < npts; ++i){	
	vis[3*i + 0] = distribution(generator);
	vis[3*i + 1] = distribution(generator);
	vis[3*i + 2] = distribution_w(generator);
    }

    return vis;
}



static inline std::vector<std::vector<double>> generate_line_visibilities(double theta,
							    double lambda,
							    double v,
							    double dw,
							      int npts){

    
    std::vector<std::vector<double>> vis(npts, std::vector<double>(3,0.0));
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator;
    generator.seed(seed);
    std::uniform_real_distribution<double> distribution(-theta*lambda/2,theta*lambda/2);
    std::uniform_real_distribution<double> distribution_w(-dw,dw);


    double npts_step = (theta*lambda)/npts;
    for(int i = 0; i < npts; ++i){	
	vis[i][0] = npts_step*i - theta*lambda/2;
	vis[i][1] = v;
	vis[i][2] = 0;
    }

    return vis;
}

static inline std::vector<double> generate_line_visibilities_(double theta,
							    double lambda,
							    double v,
							    double dw,
							      int npts){

    
    std::vector<double> vis(3*npts, 0.0);
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator;
    generator.seed(seed);
    std::uniform_real_distribution<double> distribution(-theta*lambda/2,theta*lambda/2);
    std::uniform_real_distribution<double> distribution_w(-dw,dw);

    double npts_step = (theta*lambda)/npts;
    for(int i = 0; i < npts; ++i){	
	vis[3*i + 0] = npts_step*i - theta*lambda/2;
	vis[3*i + 1] = v;
	vis[3*i + 2] = 0;
    }

    return vis;
}





// W Stac


std::complex<double> predict_visibility(const std::vector<double>& points,
					double u,
					double v,
					double w);

std::complex<double> predict_visibility_quantized_vec(const std::vector<double>& points,
						      double theta,
						      double lam,
						      double u,
						      double v,
						      double w);


std::vector<std::complex<double> > predict_visibility_quantized_vec(const std::vector<double>& points,
								double theta,
								double lam,
								std::vector<double> uvwvec);

vector2D<std::complex<double>> generate_fresnel(double theta,
						double lam,
						double dw,
						double x0);

//Quantises our sources onto the sky.
void generate_sky(const std::vector<double>& points,
		  vector2D<std::complex<double>>& sky, // We do this by reference because of FFTW planner.
		  double theta,
		  double lam,
		  double du,
		  double dw,
		  double x0,
		  struct sep_kernel_data *grid_corr_lm,
		  struct sep_kernel_data *grid_corr_n);

std::vector<double> generate_random_points(int npts, double theta);

std::vector<double> generate_testcard_dataset(double theta);


static inline std::complex<double> deconvolve_visibility(std::vector<double> uvw,
					   double du,
					   double dw,
					   int aa_support_uv,
					   int aa_support_w,
					   int oversampling,
					   int oversampling_w,
					   int w_planes,
					   int grid_size,
					   const vector3D<std::complex<double> >& wstacks,
					   struct sep_kernel_data *grid_conv_uv,
					   struct sep_kernel_data *grid_conv_w){
    // Co-ordinates
    double u = uvw[0];
    double v = uvw[1];
    double w = uvw[2];
    
    // Begin De-convolution process using Sze-Tan Kernels.
    std::complex<double> vis_sze = {0.0,0.0};

    // U/V/W oversample values
    double flu = u - std::ceil(u/du)*du;
    double flv = v - std::ceil(v/du)*du;
    double flw = w - std::ceil(w/dw)*dw;
    
    int ovu = static_cast<int>(std::floor(std::abs(flu)/du * oversampling));
    int ovv = static_cast<int>(std::floor(std::abs(flv)/du * oversampling));
    int ovw = static_cast<int>(std::floor(std::abs(flw)/dw * oversampling_w));   
    
    int aa_h = std::floor(aa_support_uv/2);
    int aaw_h = std::floor(aa_support_w/2);

    for(int dwi = -aaw_h; dwi < aaw_h; ++dwi){
	
    	int dws = static_cast<int>(std::ceil(w/dw)) + static_cast<int>(std::floor(w_planes/2)) + dwi;
    	int aas_w = aa_support_w * ovw + (dwi+aaw_h);
    	double gridconv_w = grid_conv_w->data[aas_w];
	
    	for(int dvi = -aa_h; dvi < aa_h; ++dvi){
	    
    	    int dvs = static_cast<int>(std::ceil(v/du)) + grid_size + dvi;
    	    int aas_v = aa_support_uv * ovv + (dvi+aa_h);
    	    double gridconv_uv = gridconv_w * grid_conv_uv->data[aas_v];
	    
    	    for(int dui = -aa_h; dui < aa_h; ++dui){
		
    		int dus = static_cast<int>(std::ceil(u/du)) + grid_size + dui; 
    		int aas_u = aa_support_uv * ovu + (dui+aa_h);
    		double gridconv_u = grid_conv_uv->data[aas_u];
    		double gridconv_uvw = gridconv_uv * gridconv_u;
    		vis_sze += (wstacks(dus,dvs,dws) * gridconv_uvw );
    	    }
    	}
    }


    return vis_sze;
}

static inline std::complex<double> deconvolve_visibility_(double u,
					    double v,
					    double w,
					   double du,
					   double dw,
					   int aa_support_uv,
					   int aa_support_w,
					   int oversampling,
					   int oversampling_w,
					   int w_planes,
					   int grid_size,
					   const vector3D<std::complex<double> >& wstacks,
					   struct sep_kernel_data *grid_conv_uv,
					   struct sep_kernel_data *grid_conv_w){
    
    // Begin De-convolution process using Sze-Tan Kernels.
    std::complex<double> vis_sze = {0.0,0.0};

    // U/V/W oversample values
    double flu = u - std::ceil(u/du)*du;
    double flv = v - std::ceil(v/du)*du;
    double flw = w - std::ceil(w/dw)*dw;
    
    // int ovu = static_cast<int>(std::floor(std::abs(flu)/du)) * oversampling;
    // int ovv = static_cast<int>(std::floor(std::abs(flv)/du)) * oversampling;
    // int ovw = static_cast<int>(std::floor(std::abs(flw)/dw)) * oversampling_w;   

    int ovu = static_cast<int>(std::floor(std::abs(flu)/du * oversampling));
    int ovv = static_cast<int>(std::floor(std::abs(flv)/du * oversampling));
    int ovw = static_cast<int>(std::floor(std::abs(flw)/dw * oversampling_w));   
    

    
    int aa_h = std::floor(aa_support_uv/2);
    int aaw_h = std::floor(aa_support_w/2);


    for(int dwi = -aaw_h; dwi < aaw_h; ++dwi){
	
	int dws = static_cast<int>(std::ceil(w/dw)) + static_cast<int>(std::floor(w_planes/2)) + dwi;
	int aas_w = aa_support_w * ovw + (dwi+aaw_h);
	double gridconv_w = grid_conv_w->data[aas_w];
	
	for(int dvi = -aa_h; dvi < aa_h; ++dvi){
	    
	    int dvs = static_cast<int>(std::ceil(v/du)) + grid_size + dvi;
	    int aas_v = aa_support_uv * ovv + (dvi+aa_h);
	    double gridconv_uv = gridconv_w * grid_conv_uv->data[aas_v];
	    
	    for(int dui = -aa_h; dui < aa_h; ++dui){
		
		int dus = static_cast<int>(std::ceil(u/du)) + grid_size + dui; 
		int aas_u = aa_support_uv * ovu + (dui+aa_h);
		double gridconv_u = grid_conv_uv->data[aas_u];
		double gridconv_uvw = gridconv_uv * gridconv_u;
		vis_sze += (wstacks(dus,dvs,dws) * gridconv_uvw );
	    }
	}
    }

    return vis_sze;
}



std::vector<std::complex<double> > wstack_predict(double theta,
						  double lam,
						  const std::vector<double>& points,
						  std::vector<double> uvwvec,
						  double du, // Sze-Tan Optimum Spacing in U/V
						  double dw, // Sze-Tan Optimum Spacing in W
						  int aa_support_uv,
						  int aa_support_w,
						  double x0,
						  struct sep_kernel_data *grid_conv_uv,
						  struct sep_kernel_data *grid_conv_w,
						  struct sep_kernel_data *grid_corr_lm,
						  struct sep_kernel_data *grid_corr_n);

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
								    struct sep_kernel_data *grid_corr_n);


#endif
