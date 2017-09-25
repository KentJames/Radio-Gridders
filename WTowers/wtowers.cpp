/*
W-Towers CUDA Implementation. Radio Astronomy.

This is an implementation of W-Towers, an algorithmic framework
for image-generation in radio astronomy. This framework was 
first proposed by Peter Wortmann. 



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

#include "wtowers_common.h"



void showHelp(){

  std::cout<<"cuTowers v1.0\n";
  std::cout<<"James Kent <jck42@cam.ac.uk>\n";
  std::cout<<"A CUDA Implementation of W-Towers \n";
  std::cout<<"University of Cambridge\n\n";
  std::cout<<"\t-theta               Field of View (Radians)\n";
  std::cout<<"\t-lambda              Number of Wavelengths\n";
  std::cout<<"\t-image               Image File\n";
  std::cout<<"\t-wkernel             Input W-Kernels\n";
  //  std::cout<<"\t-akernel             Input A-Kernels\n";
  std::cout<<"\t-vis                 Input Visibilities\n";
  std::cout<<"\t-subgrid             W-Towers Subgrid Size\n";
  std::cout<<"\t-margin              W-Towers Margin Size\n";
  std::cout<<"\t-wincrement          W-Plane Increment Size\n";
  std::cout<<"\n\n\n";
}

int main (int argc, char **argv) {


  //Get information on GPU's in system.
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

  int subgrid_size;
  int subgrid_margin;
  double wincrement;
  
  char *visfile = NULL;
  char *wkernfile = NULL;
  char *akernfile = NULL;

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

  if (checkCmdLineFlag(argc, (const char **)argv, "wkern") == 0){

    std::cout << "No W-Kernel File Specified!! \n";
    showHelp();
    return 0;
  }
  else {

    getCmdLineArgumentString(argc, (const char **) argv, "wkern", &wkernfile);

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

  if(checkCmdLineFlag(argc, (const char **) argv, "subgrid") == 0 ||
     checkCmdLineFlag(argc, (const char **) argv, "margin") == 0 ||
     checkCmdLineFlag(argc, (const char **) argv, "winc") == 0) {

    std::cout << "Missing W-Towers Parameters. Check subgrid/margin/winc\n";
    showHelp();
    return 0;
  }
  else {

    subgrid_size = getCmdLineArgumentInt(argc, (const char **) argv, "subgrid");
    subgrid_margin = getCmdLineArgumentInt(argc, (const char **) argv, "margin");
    wincrement = getCmdLineArgumentFloat(argc, (const char **) argv, "wincrement");

  }

  //File I/O

  
  int grid_size = floor(lambda * theta);

  const char* visfile_c = visfile;
  const char* wkernfile_c = wkernfile;

  cuDoubleComplex *image_host, *grid_dev;

  //May as well allocate our host image now for when we move it back.
  image_host = (cuDoubleComplex*)malloc(grid_size * grid_size * sizeof(cuDoubleComplex));


  //Call our W-Towers wrapper.
  wtowers_host(visfile_c, wkernfile_c, grid_size, theta, lambda, bl_min, bl_max,1);

  

  


  

  return 0;

}
