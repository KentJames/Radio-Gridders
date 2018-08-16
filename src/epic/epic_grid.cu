//C++ Includes
#include <iostream>
#include <cassert>
#include <sys/time.h>

//CUDA Includes
#include <cuComplex.h>
#include "cuda.h"
#include "math.h"
#include "cuda_runtime_api.h"
#include <thrust/random.h>

//Radio Gridders Includes
#include "radio.cuh"

// Saves me writing a CLI. Sorry Jayce.
#define GRID_SIZE 128 // UV Grid Size in 1-D
#define ANTENNAS 256 // Antennas to grid
#define ILLUM_X 2 // Aperture pattern extent. Assuming top hat,
#define ILLUM_Y 2 // '' for Y
#define NBATCH 8192
#define CHANNELS 4


// SoA format for mock fengine_data
struct fengine_data{
    int *x, *y, *z; //Just put exactly on grid points for sake of simplicity.
    cuComplex* fdata;
    int number_of_f_points;
    int batch_len; // Stride between batch[i] and batch[i+1]
};

/*****************************
        Device Functions
 *****************************/

//From Kyrills implementation in SKA/RC
__device__ void scatter_grid_add(cuComplex *uvgrid,
				 int grid_size,
				 int grid_pitch,
				 int grid_point_u,
				 int grid_point_v,
				 cuComplex sum){

    if (grid_point_u < 0 || grid_point_u >= grid_size ||
      grid_point_v < 0 || grid_point_v >= grid_size)
    return;

    // Add to grid. This is the bottleneck of the entire kernel
    atomicAdd(&uvgrid[grid_point_u + grid_pitch*grid_point_v].x, sum.x); // Re
    atomicAdd(&uvgrid[grid_point_u + grid_pitch*grid_point_v].y, sum.y); // Im
}

#ifdef __COUNT_VIS__
__device__ void scatter_grid_point(cuComplex* fdata,
				   cuComplex* uvgrid, // Our main UV Grid
				   cuComplex* illum, //Our W-Kernel
				   int* x,
				   int* y,
				   int* z,
				   int max_supp, // Max size of W-Kernel
				   int myU, //Our assigned u/v points.
				   int myV, // ^^^
				   int grid_size, //The size of our w-towers subgrid.
				   int data_size,
				   int batch_no,
				   unsigned long long int *visc_reg){ 
#else
 __device__ void scatter_grid_point(cuComplex* fdata,
				    cuComplex* uvgrid, // Our main UV Grid
				    cuComplex* illum, //Our Illumination-Kernel
				    int* x,
				    int* y,
				    int* z,
				    int max_supp, // Max size of W-Kernel
				    int myU, //Our assigned u/v points.
				    int myV, // ^^^
				    int grid_size,
				    int data_size,
				    int batch_no){ 
#endif
  
  int grid_point_u = myU, grid_point_v = myV;
  cuComplex sum  = make_cuComplex(0.0,0.0);
  short supp = ILLUM_X;
  int vi_s = batch_no * data_size;
  int grid_s = grid_size * grid_size * batch_no;
  int vi = 0;
  for (vi = vi_s; vi < (vi_s+data_size); ++vi){

    int u = x[vi]; 
    int v = y[vi];

    // Determine convolution point. This is basically just an
    // optimised way to calculate.
    //int myConvU = myU - u;
    //int myConvV = myV - v;
    int myConvU = (u - myU) % max_supp;
    int myConvV = (v - myV) % max_supp;    
    if (myConvU < 0) myConvU += max_supp;
    if (myConvV < 0) myConvV += max_supp;

    // Determine grid point. Because of the above we know here that
    //   myGridU % max_supp = myU
    //   myGridV % max_supp = myV
    int myGridU = u + myConvU
      , myGridV = v + myConvV;

    // Grid point changed?
    if (myGridU != grid_point_u || myGridV != grid_point_v) {
      // Atomically add to grid. This is the bottleneck of this kernel.
      scatter_grid_add(uvgrid+grid_s, grid_size, grid_size, grid_point_u, grid_point_v, sum);
      // Switch to new point
      sum = make_cuComplex(0.0, 0.0);
      grid_point_u = myGridU;
      grid_point_v = myGridV;
    }
    //TODO: Re-do the w-kernel/gcf for our data.
    //	cuDoubleComplex px;
    cuComplex px = illum[myConvV * supp + myConvU];// ??
    //cuComplex px = *(cuComplex*)&wkern->kern_by_w[w_plane].data[sub_offset + myConvV * supp + myConvU];	
    // Sum up
    cuComplex vi_v = fdata[vi];
    sum = cuCfmaf(cuConjf(px), vi_v, sum);

  }
  // Add remaining sum to grid
  #ifdef __COUNT_VIS__
  atomicAdd(visc_reg,vi);
  #endif
  scatter_grid_add(uvgrid+grid_s, grid_size, grid_size, grid_point_u, grid_point_v, sum);
}



/*******************
   Romein Kernel
 ******************/
 
#ifdef __COUNT_VIS__
 __global__ void scatter_grid_kernel(cuComplex* fdata,
				     cuComplex* illum, // Illumination Pattern
				     cuComplex* uvgrid, //Our UV-Grid
				     int* x,
				     int* y,
				     int* z
				     int max_support, //  Convolution size
				     int grid_size, // Subgrid size
				     int data_size,
				     int batch_no,
				     unsigned long long int* visc_reg){
#else
__global__ void scatter_grid_kernel(cuComplex* fdata,
				    cuComplex* illum, // Illumination Pattern
				    cuComplex* uvgrid, //Our UV-Grid
				    int* x,
				    int* y,
				    int* z,
				    int max_support, //  Convolution size
				    int grid_size,
				    int data_size){
				
#endif
  //Assign some visibilities to grid;
    int batch_no = blockIdx.x;
    for(int i = threadIdx.x; i < max_support * max_support; i += blockDim.x){
	//  int i = threadIdx.x + blockIdx.x * blockDim.x;
	int myU = i % max_support;
	int myV = i / max_support;
    
#ifdef __COUNT_VIS__
	scatter_grid_point(fdata, uvgrid, illum, x, y, z, max_support, myU, myV, grid_size, data_size, batch_no, visc_reg);
#else
	scatter_grid_point(fdata, uvgrid, illum, x, y, z, max_support, myU, myV, grid_size, data_size, batch_no);
#endif		       
  }
}


/*************************
   EPIC Romein Interface
**************************/

 
__host__ cudaError_t epic_romein(cuComplex* fdata,
				 cuComplex* illum, // Our illumination pattern.
				 cuComplex* uvgrid, // Our grid
				 int* x,
				 int* y,
				 int* z,
				 int support_size, // Convolution size
				 int grid_size, // Size of grid along each dimension. (Square matrix)
				 int data_size){

    struct timeval tcuda_start, tcuda_end;
    
    std::cout << "Initialising CUDA streams... ";
    cudaStream_t* streams = (cudaStream_t *) malloc(NBATCH * sizeof(cudaStream_t));
    for (int i = 0; i < NBATCH; ++i) cudaError_check(cudaStreamCreate(&streams[i]));
    std::cout << "done\n";
    std::cout << "Executing batch of Romein Kernels... ";
    // Launch a batch of Romein-Kernels.
    gettimeofday(&tcuda_start,NULL);
    //for (int i = 0; i < NBATCH; ++i){
	//std::cout << i << "\n";
	// Run each one asynchronously to maximise GPU occupancy.
    scatter_grid_kernel <<< NBATCH, 32, 0, streams[0] >>> (fdata, illum, uvgrid, x, y, z, support_size, grid_size, data_size);
	//cudaError_check(cudaDeviceSynchronize()); // Debug
	//}
    cudaError_check(cudaDeviceSynchronize());
    gettimeofday(&tcuda_end,NULL);
    std::cout << "done\n";
    std::cout << "Romein batch time... " << ((tcuda_end.tv_sec - tcuda_start.tv_sec)*1000000 + 
					     (tcuda_end.tv_usec - tcuda_start.tv_usec));
    std::cout << "us\n";
	  
    return cudaSuccess;
}

/****************************************
   Naive Implementation for Validation
*****************************************/

 __host__ cudaError_t epic_naive(cuComplex* fdata,
				 cuComplex* illum_host,
				 cuComplex* uvgrid_host,
				 int* x,
				 int* y,
				 int* z,
				 int support_size,
				 int grid_size,
				 int data_size,
				 int batch_no){

     for (int g = 0; g < batch_no; ++g){
	 for (int fd = 0; fd < data_size; ++fd){
	     int x_s = x[g * data_size + fd];
	     int y_s = y[g * data_size + fd];
	     cuComplex fdp = fdata[g * data_size + fd];	     
	     for (int y = y_s; y < y_s + support_size; ++y){
		 for (int x = x_s; x < x_s + support_size; ++x){
		     cuComplex illump = illum_host[(y-y_s) * support_size + (x-x_s)];
		     uvgrid_host[g * grid_size * grid_size + y * grid_size + x] =
			 cuCfmaf(illump, fdp, uvgrid_host[g * grid_size * grid_size + y * grid_size + x]);
		 }
	     }
	 }
     }     
     return cudaSuccess;
}

int main(){

    //Get information on GPU's in system.
    std::cout << "CUDA System Information: \n\n";
    int numberofgpus;
    
    
    cudaGetDeviceCount(&numberofgpus);
    std::cout << " Number of GPUs Detected: " << numberofgpus << "\n\n";
    
    cudaDeviceProp *prop = (cudaDeviceProp*)malloc(sizeof(cudaDeviceProp) * numberofgpus);
    
    for(int i=0; i<numberofgpus;i++){
	
	
	cudaGetDeviceProperties(&prop[i],i);
	
	std::cout << "\tDevice Number: " << i <<" \n";
	std::cout << "\t\tDevice Name: " << prop->name <<"\n";
	std::cout << "\t\tTotal Memory: " << (double)prop->totalGlobalMem / (1024 * 1024) << " MB \n";
	std::cout << "\t\tShared Memory (per block): " << (double)prop->sharedMemPerBlock / 1024 << " kB \n";
	std::cout << "\t\tClock Rate: " << prop->clockRate << "\n";
	std::cout << "\t\tMultiprocessors: " << prop->multiProcessorCount << "\n";
	std::cout << "\t\tThreads Per MP: " << prop->maxThreadsPerMultiProcessor << "\n";
	std::cout << "\t\tThreads Per Block: " << prop->maxThreadsPerBlock << "\n";
	std::cout << "\t\tThreads Per Dim: " << prop->maxThreadsDim << "\n";
	std::cout << "\t\tThreads Per Warp: " << prop->warpSize << "\n";
	std::cout << "\t\tUnified Addressing: " << prop->unifiedAddressing << "\n";
	std::cout << "\n";	
    }
    
    cuComplex* uvgrid; // Our U-V Grid
    cuComplex* illumination; //Our illumination pattern
    cuComplex* uvgrid_host;
    cuComplex* fdata;
    int* x;
    int* y;
    int* z;
        
    int gsize = GRID_SIZE * GRID_SIZE * NBATCH * sizeof(cuComplex);
    int vecsize_loc = ANTENNAS * NBATCH * sizeof(double);
    int vecsize_data = ANTENNAS * NBATCH * sizeof(cuComplex);
    int illumsize = ILLUM_X * ILLUM_Y * sizeof(cuComplex);
    
    // Mallocs
    cudaError_check(cudaMallocManaged((void **)&uvgrid, gsize,cudaMemAttachGlobal));
    cudaError_check(cudaMallocManaged((void **)&illumination, illumsize, cudaMemAttachGlobal));
    cudaError_check(cudaMallocManaged((void **)&uvgrid_host, gsize, cudaMemAttachGlobal));
    //Allocate elements of fdata.
    cudaError_check(cudaMallocManaged((void **)&x, vecsize_loc, cudaMemAttachGlobal));
    cudaError_check(cudaMallocManaged((void **)&y, vecsize_loc, cudaMemAttachGlobal));
    cudaError_check(cudaMallocManaged((void **)&z, vecsize_loc, cudaMemAttachGlobal));
    cudaError_check(cudaMallocManaged((void **)&fdata, vecsize_data, cudaMemAttachGlobal));
    
    //Memset uvgrid
    cudaError_check(cudaMemset(uvgrid, 0.0, gsize));
    cudaError_check(cudaMemset(uvgrid_host, 0.0, gsize));

    std::cout << "Initialising random input data... " << std::flush;
    // Initialise input vector with random fourier values.
    thrust::default_random_engine generator;
    thrust::normal_distribution<float> fdata_distribution(0.0,5.0);
    thrust::uniform_int_distribution<int> loc_distribution(ILLUM_X,GRID_SIZE-ILLUM_X);
   
    for (int i = 0; i < ANTENNAS * NBATCH; ++i){
	fdata[i] = make_cuComplex(fdata_distribution(generator),fdata_distribution(generator));
	x[i] = loc_distribution(generator);
	y[i] = loc_distribution(generator);
    }
    cudaError_check(cudaMemset(z,0,vecsize_loc)); // Don't care about z yet..
    std::cout << "done\n";

    std::cout << "Initialising illumination pattern (square top-hat function)... " << std::flush;
    // Initialise illumination pattern. Square top-hat function.
    for (int i = 0; i < ILLUM_X * ILLUM_Y; ++i) illumination[i] = make_cuComplex(1.0,0.0);
    std::cout << "done\n";   

    // Initialise and run CUDA.
    std::cout << "Antennas... " << ANTENNAS << "\n";
    std::cout << "No. of timestamps... " << NBATCH << "\n";
    std::cout << "Grid Size... " << GRID_SIZE << "x" << GRID_SIZE << "\n";
    std::cout << "Illumination Pattern Size... " << ILLUM_X << "x" << ILLUM_Y << "\n";
    std::cout << "Amount of real-time data in fake batch... " << ((NBATCH*0.00001)*1000000) << "us\n";
    cudaError_check(epic_romein(fdata, illumination, uvgrid, x, y, z, ILLUM_X, GRID_SIZE, ANTENNAS));
    cudaError_check(cudaDeviceSynchronize());

    //Run validation test case
    std::cout << "Running validation case... " << std::flush;
    epic_naive(fdata, illumination, uvgrid_host, x, y, z, ILLUM_X, GRID_SIZE, ANTENNAS, NBATCH);
    cudaError_check(cudaDeviceSynchronize());
    std::cout << "done\n";

    std::cout << "Computing delta between Romein/Naive... " << std::flush; 
    cuDoubleComplex delta;
    for (int d = 0; d < (GRID_SIZE * GRID_SIZE * NBATCH); ++d){
	cuComplex f1 = uvgrid[d];
	cuComplex f2 = uvgrid_host[d];
	cuComplex deltad = cuCsubf(f1,f2);
	cuDoubleComplex deltadd = make_cuDoubleComplex(cuCrealf(deltad),cuCimagf(deltad));
	delta = cuCadd(delta,deltadd);
    }
    std::cout << "done\n";
    std::cout << "Real delta... " << fabs(cuCreal(delta)) << "\n";
    std::cout << "Imag delta... " << fabs(cuCimag(delta)) << "\n";

    std::cout << "Validation passed check... " << std::flush;
    if ((fabs(cuCreal(delta)) < 0.1) && (fabs(cuCimag(delta)) < 0.1)){
	std::cout << "PASS\n";
    } else {
	std::cout << "FAIL\n";
    }

    //Lets look at the grid...
    double *row = (double *)malloc(sizeof(double) * GRID_SIZE);
    std::ofstream image_f ("image.out", std::ofstream::out | std::ofstream::binary);
    std::cout << "Writing Romein Image to File... " << std::flush;
    int grid_index = 2400 * GRID_SIZE * GRID_SIZE; //Arbitrary grid...
    for(int i = 0; i < GRID_SIZE; ++i){
	for(int j = 0; j < GRID_SIZE; ++j){
	    row[j] = cuCrealf(uvgrid[grid_index + i*GRID_SIZE + j]);
	}
	image_f.write((char*)row, sizeof(double) * GRID_SIZE);
    }
    image_f.close();
    std::cout << "done\n";

    std::ofstream image_n ("image_naive.out", std::ofstream::out | std::ofstream::binary);
    std::cout << "Writing Naive Image to File... " << std::flush;
    //int grid_index = 2400 * GRID_SIZE * GRID_SIZE; //Arbitrary grid...
    for(int i = 0; i < GRID_SIZE; ++i){
	for(int j = 0; j < GRID_SIZE; ++j){
	    row[j] = cuCrealf(uvgrid_host[grid_index + i*GRID_SIZE + j]);
	}
	image_n.write((char*)row, sizeof(double) * GRID_SIZE);
    }
    image_n.close();
    std::cout << "done\n";

    cudaError_t err = cudaGetLastError();
    std::cout << "Error: " << cudaGetErrorString(err) << "\n";
    cudaError_check(cudaDeviceReset());
    

}