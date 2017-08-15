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

#include "dft_common.h"



void showHelp(){

  std::cout<<"cuDFT v1.0\n";
  std::cout<<"James Kent <jck42@cam.ac.uk>\n";
  std::cout<<"University of Cambridge\n\n";
  std::cout<<"\t-theta               Field of View (Radians)\n";
  std::cout<<"\t-lambda              Number of Wavelengths\n";
  std::cout<<"\t-image               Image File\n";
  std::cout<<"\t-vis                 Input Visibilities\n";
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
    
    std::cout << "  Device Number: " << i <<" \n";
    std::cout << "   Device Name: " << prop.name <<"\n";
    std::cout << "   Total Memory: " << prop.totalGlobalMem << "\n";
    std::cout << "   Clock Rate: " << prop.clockRate << "\n";
    std::cout << "   Threads Per MP: " << prop.maxThreadsPerMultiProcessor << "\n";
    std::cout << "   Threads Per Block: " << prop.maxThreadsPerBlock << "\n";
    std::cout << "   Threads Per Dim: " << prop.maxThreadsDim << "\n";
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
  
  // Parameters
  if (checkCmdLineFlag(argc, (const char **)argv, "help")) {

    showHelp();
    return 0;
  }

  
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

  
  //File I/O

  
  struct vis_data vis_dat;

  const char* visfile_c = visfile;
  if (load_vis(visfile_c,&vis_dat,bl_min,bl_max)) return 1; 


  //Declare our grid.
  int grid_size = floor(lambda * theta);

  std::cout << "Grid Size: " << grid_size << " x " << grid_size << "\n";
  std::cout << "Grid Memory: " << (grid_size * grid_size * sizeof(double _Complex))/1e9;

  double _Complex *grid = (double _Complex*)malloc(grid_size * grid_size * sizeof(double _Complex));



  
  return 0;

}
