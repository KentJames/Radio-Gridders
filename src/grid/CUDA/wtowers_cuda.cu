#include <iostream>
#include <cmath>
#include <algorithm>
#include <random>
#include <cuComplex.h>
#include "cuda.h"
#include "cuda_runtime_api.h"

#define GRID_SIZE 4096
#define N GRID_SIZE * GRID_SIZE
#define THREADS_BLOCK 1024

__host__ void populate_grid (double *a){

    std::random_device rdv;
    std::default_random_engine dre(rdv());
    std::uniform_real_distribution<double> uid(0,9);
    std::generate(a,a + (GRID_SIZE*GRID_SIZE), [&] () { return uid(dre); }); 
   /* 
    for(int i = 0; i<GRID_SIZE;i++){
        for(int j=0;j<GRID_SIZE;j++){
            *(a+i*GRID_SIZE + j) = uid(dre);
        }
    }
*/
}


__global__ void reduce_grid (double *c, double *b, double *a)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    c[i] =  a[i] + b[i];
} 

int main (void) {




    int numberofgpus;

    cudaGetDeviceCount(&numberofgpus);
    std::cout << "Number of GPUs Detected: " << numberofgpus << "\n"; 
    for(int i=0; i<numberofgpus;i++){

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop,i);

        std::cout << "Device Number: " << i <<" \n";
        std::cout << "Device Name: " << prop.name <<"\n";
        std::cout << "Total Memory: " << prop.totalGlobalMem << "\n";
        std::cout << "Clock Rate: " << prop.clockRate << "\n";
        std::cout << "Threads Per MP: " << prop.maxThreadsPerMultiProcessor << "\n";
        std::cout << "Threads Per Block: " << prop.maxThreadsPerBlock << "\n";
        std::cout << "Threads Per Dim: " << prop.maxThreadsDim << "\n";
        std::cout << "\n";
         

    }



    double *a_host, *b_host, *a_dev, *b_dev, *c_dev;

    a_host = (double *)malloc(GRID_SIZE * GRID_SIZE * sizeof(double));
    b_host = (double *)malloc(GRID_SIZE * GRID_SIZE * sizeof(double));
    //cudaMallocHost((void **)&a_host,GRID_SIZE * GRID_SIZE * sizeof(double));
    //cudaMallocHost((void **)&b_host,GRID_SIZE * GRID_SIZE * sizeof(double));

    
    cudaMalloc((void **)&a_dev,GRID_SIZE * GRID_SIZE * sizeof(double));
    cudaMalloc((void **)&b_dev,GRID_SIZE * GRID_SIZE * sizeof(double));
    cudaMalloc((void **)&c_dev,GRID_SIZE * GRID_SIZE * sizeof(double));
    populate_grid(a_host);
    populate_grid(b_host); 

    std::cout << a_host[1] << "\n";
    std::cout << b_host[1] << "\n";
    


    cudaMemcpy(a_dev,a_host,sizeof(double)*GRID_SIZE*GRID_SIZE,cudaMemcpyHostToDevice);
    cudaMemcpy(b_dev,b_host,sizeof(double)*GRID_SIZE*GRID_SIZE,cudaMemcpyHostToDevice);

    

//    for(int i=0;i=10;i++){
        reduce_grid<<<N/THREADS_BLOCK,THREADS_BLOCK>>>(c_dev,b_dev,a_dev);
  //  }

    cudaMemcpy(a_host,c_dev,sizeof(double)*GRID_SIZE*GRID_SIZE,cudaMemcpyDeviceToHost);
    std::cout << a_host[1] << "\n";
}
