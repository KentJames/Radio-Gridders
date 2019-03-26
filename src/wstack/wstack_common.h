
#include "hdf5.cuh"

#ifndef WSTACK_H
#define WSTACK_H

#define THREADS_BLOCK 16

#include <vector>
#include <complex>

template<class T>
constexpr T PI = T(3.1415926535897932385L);

template <typename T>
class vector2D {
public:

    vector2D(size_t d1=0, size_t d2=0, T const & t=T()) :
        d1(d1), d2(d2), data(d1*d2, t)
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
    size_t d1,d2;    
    std::vector<T> data;
};


template <typename T>
class vector3D {
public:
    vector3D(size_t d1=0, size_t d2=0, size_t d3=0, T const & t=T()) :
        d1(d1), d2(d2), d3(d3), data(d1*d2*d3, t)
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


private:
    size_t d1,d2,d3;
    std::vector<T> data;
};




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
