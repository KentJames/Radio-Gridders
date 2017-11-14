/*
CUDA Discrete Fourier Transform. Radio Astronomy.

Author: James Kent <jck42@cam.ac.uk>
Institution: University of Cambridge

*/

#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <random>

//CUDA Includes
#include <cuComplex.h>
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "helper_string.h"

//Our include

#include "dft_common.cuh"



void showHelp(){

  std::cout<<"cuDFT v1.0\n";
  std::cout<<"\nCalculates direct iDFT (not iFFT) of visibilities\n"
                "representing the spatial coherence function of a\n"
                "radio astronomy interferometer baseline.\n\n";
  std::cout<<"James Kent <jck42@cam.ac.uk>\n";
  std::cout<<"University of Cambridge\n\n";
  std::cout<<"\t-theta               Field of View (Radians)\n";
  std::cout<<"\t-lambda              Number of Wavelengths\n";
  std::cout<<"\t-image               Image File\n";
  std::cout<<"\t-vis                 Input Visibilities\n";
  std::cout<<"\t-blocks              Number of CUDA Blocks\n";
  std::cout<<"\t-threadblock         Number of CUDA Threads per Block\n";
  std::cout<<"\t-device              Device to Execute On (Int)\n";
  std::cout<<"\n\n\n";
}

int main (int argc, char **argv) {


  
  std::cout << "CUDA System Information: \n\n";
  int numberofgpus;

  cudaGetDeviceCount(&numberofgpus);
  std::cout << " Number of GPUs Detected: " << numberofgpus << "\n\n"; 
  for(int i=0; i<numberofgpus;i++){

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop,i);
    
    std::cout << "\tDevice Number: " << i <<" \n";
    std::cout << "\t\tDevice Name: " << prop.name <<"\n";
    std::cout << "\t\tTotal Memory: " << prop.totalGlobalMem << "\n";
    std::cout << "\t\tClock Rate: " << prop.clockRate << "\n";
    std::cout << "\t\tThreads Per MP: " << prop.maxThreadsPerMultiProcessor << "\n";
    std::cout << "\t\tThreads Per Block: " << prop.maxThreadsPerBlock << "\n";
    std::cout << "\t\tThreads Per Dim: " << prop.maxThreadsDim << "\n";
    std::cout << "\n";
         

  }

  

  init_dtype_cpx();
  
  double theta = false;
  double lambda = false;
  char *image = NULL;
  bool vis = false;

  char *visfile = NULL;

  double bl_min = 0;
  double bl_max = 1.7976931348623158e+308 ;

  int dev_no = 0, cuda_blocks, cuda_threads_block;

  
  
  // Parameters
  if (checkCmdLineFlag(argc, (const char **)argv, "help")) {

    showHelp();
    return 0;
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "device")){
    dev_no = getCmdLineArgumentInt(argc, (const char **) argv, "device");
  } else { dev_no = 0; }

  cudaSetDevice(dev_no);
  
  if (checkCmdLineFlag(argc, (const char **)argv, "vis") == 0){

    std::cout << "No Visibility File Specified!! \n";
    showHelp();
    return 0;
  }
  else {

    getCmdLineArgumentString(argc, (const char **) argv, "vis", &visfile);
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "theta") == 0){

    std::cout << "No theta given!! \n";
    showHelp();
    return 0;
  }
  else {

    theta = getCmdLineArgumentFloat(argc, (const char **) argv, "theta");

  }

  if (checkCmdLineFlag(argc, (const char **) argv, "lambda") == 0){

    std::cout << "No lambda given!! \n";
    showHelp();
    return 0;
  }
  else {

    lambda = getCmdLineArgumentFloat(argc, (const char **) argv, "lambda");

  }


  if (checkCmdLineFlag(argc, (const char **) argv, "blocks") == 0){

    std::cout << "Number of CUDA Blocks not specified!! \n";
    showHelp();
    return 0;
  }
  else {

    cuda_blocks = getCmdLineArgumentInt(argc, (const char **) argv, "blocks");

  }

  if (checkCmdLineFlag(argc, (const char **) argv, "threadblock") == 0){

    std::cout << "Number of CUDA Threads per Block  not specified!! \n";
    showHelp();
    return 0;
  }
  else {

    cuda_threads_block = getCmdLineArgumentInt(argc, (const char **) argv, "threadblock");

  }
  
  
  //File I/O

  
  int grid_size = floor(lambda * theta);

  const char* visfile_c = visfile;

  cuDoubleComplex *image_host, *grid_dev;

  //May as well allocate our host image now for when we move it back.
  image_host = (cuDoubleComplex*)malloc(grid_size * grid_size * sizeof(cuDoubleComplex));

  image_dft_host(visfile_c, grid_size, theta, lambda, bl_min, bl_max,
		 cuda_blocks, cuda_threads_block);

  


  

  return 0;

}
