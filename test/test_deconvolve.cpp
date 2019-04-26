#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <ctime>
#include <chrono>

#include "wstack_common.h"
#include "helper_string.h"

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
    // Co-ordinates
    // double u = uvw[0];
    // double v = uvw[1];
    // double w = uvw[2];
    
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
	
	int dws = static_cast<int>(std::ceil(w/dw) + std::floor(w_planes/2) + dwi);
	//int aas_w = (dwi+aaw_h) * oversampling_w + ovw;
	int aas_w = aa_support_w * ovw + (dwi+aaw_h);
	//double gridconv_w = 1.0;
	double gridconv_w = grid_conv_w->data[aas_w];
	
	for(int dvi = -aa_h; dvi < aa_h; ++dvi){
	    
	    int dvs = static_cast<int>(std::ceil(v/du) + grid_size + dvi);
	    //int aas_v = (dvi+aa_h) * oversampling + ovv;
	    int aas_v = aa_support_uv * ovv + (dvi+aa_h);
	    //double gridconv_uv = 1.0;
	    double gridconv_uv = gridconv_w * grid_conv_uv->data[aas_v];
	    
	    for(int dui = -aa_h; dui < aa_h; ++dui){
		
		int dus = static_cast<int>(std::ceil(u/du) + grid_size + dui); 
		//int aas_u = (dui+aa_h) * oversampling + ovu;
		int aas_u = aa_support_uv * ovu + (dui+aa_h);
		double gridconv_u = grid_conv_uv->data[aas_u];
		//double gridconv_u = 1.0;
		double gridconv_uvw = gridconv_uv * gridconv_u;
		vis_sze += (wstacks(dus,dvs,dws) * gridconv_uvw );
		//vis_sze += 1.0 * gridconv_uvw;
	    }
	}
    }

    return vis_sze;
}

std::vector<std::vector<double>> generate_random_visibilities(double theta,
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

std::vector<double> generate_random_visibilities_(double theta,
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



std::vector<std::vector<double>> generate_line_visibilities(double theta,
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

std::vector<double> generate_line_visibilities_(double theta,
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


int main(int argc, char **argv){


    double theta = 0.1;
    double lambda=20480;
    int grid_size = static_cast<int>(std::floor(theta * lambda));
    int oversampg = static_cast<int>(2 * grid_size);
    int w_planes = 8;
    int npts = 120000000;    


    struct sep_kernel_data *sepkern_uv = (struct sep_kernel_data *)malloc(sizeof(struct sep_kernel_data));
    struct sep_kernel_data *sepkern_w = (struct sep_kernel_data*)malloc(sizeof(struct sep_kernel_data));
    struct sep_kernel_data *sepkern_lm = (struct sep_kernel_data*)malloc(sizeof(struct sep_kernel_data));
    struct sep_kernel_data *sepkern_n = (struct sep_kernel_data*)malloc(sizeof(struct sep_kernel_data));
    
    std::cout << "Loading Kernel...";
    load_sep_kern_T("./kernels/sze/sepkern_uv_transpose.hdf5",sepkern_uv);
    std::cout << "Loading W Kernel...";
    load_sep_kern_T("./kernels/sze/sepkern_w_transpose.hdf5",sepkern_w);

    std::cout << "Loading AA Kernel...";
    load_sep_kern("./kernels/sze/sepkern_lm.hdf5", sepkern_lm);
    std::cout << "Loading AA Kernel...";
    load_sep_kern("./kernels/sze/sepkern_n.hdf5", sepkern_n);


    double du = sepkern_uv->du;
    double dw = sepkern_w->dw;
    double x0 = sepkern_lm->x0;

    int support_uv = sepkern_uv->size;
    int support_w = sepkern_w->size;
    
    
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator;
    generator.seed(seed);
    std::uniform_real_distribution<double> distribution(0,1);
    vector3D<std::complex<double> > wstacks_s (oversampg, oversampg, w_planes, {0.0,0.0});
    wstacks_s.fill_random();

    long flops_per_vis = 6 * support_uv * support_uv * support_w;
    long total_flops = flops_per_vis * npts;
    std::cout << "Total Flops: " << total_flops << "\n";

    
    std::vector<double> uvwvec = generate_line_visibilities_(theta,lambda,0.0,4.3,npts);
    //std::vector<double> uvwvec = generate_random_visibilities_(theta,lambda,4.3,npts);
    std::vector<std::complex<double> > visibilities(uvwvec.size(),{0.0,0.0});
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
#pragma omp parallel
#pragma omp for schedule(static,1000000)
    for (std::size_t i = 0; i < npts; ++i){

    	visibilities[i] = deconvolve_visibility_(uvwvec[3*i + 0],
						 uvwvec[3*i + 1],
						 uvwvec[3*i + 2],
						 du,
						 dw,
						 support_uv,
						 support_w,
						 sepkern_uv->oversampling,
						 sepkern_w->oversampling,
						 w_planes,
						 grid_size,
						 wstacks_s,
						 sepkern_uv,
						 sepkern_w);
    }

    
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
    float duration_s = static_cast<float>(duration)/1000;
    float gflops = static_cast<float>(total_flops) / duration_s;
   
    std::cout << "Deconvolve Time: " << duration << "ms \n";;
    std::cout << "GFLOP/s: " << gflops << "\n"; 
    
    

    
}


