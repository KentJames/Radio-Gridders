#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <ctime>
#include <chrono>

#include "wstack_common.h"
#include "helper_string.h"




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
    vector3D<std::complex<double> > wstacks_s (oversampg, oversampg, w_planes, {0.0,0.0},1,9,5);
    wstacks_s.fill_random();

    long flops_per_vis = 6 * support_uv * support_uv * support_w; // 3D Deconvolve
    flops_per_vis += 3 * support_uv * support_uv * support_w; // Compute seperable kernel
    long total_flops = flops_per_vis * npts;
    std::cout << "Total Flops: " << total_flops << "\n";

    
    std::vector<double> uvwvec = generate_line_visibilities_(theta,lambda,0.0,4.3,npts);
    //std::vector<double> uvwvec = generate_random_visibilities_(theta,lambda,4.3,npts);
    std::vector<std::complex<double> > visibilities(uvwvec.size(),{0.0,0.0});
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
#pragma omp parallel
#pragma omp for schedule(static,1000)
    //#pragma omp for schedule(dynamic)
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


