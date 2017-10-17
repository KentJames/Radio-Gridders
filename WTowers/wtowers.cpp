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

  std::cout<<"\ncuTowers v1.0\n";
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
  std::cout<<"\t-winc                W-Plane Increment Size\n";
  std::cout<<"\t-bl_min              Minimum Baseline Distance (wavelengths)\n";
  std::cout<<"\t-bl_max              Maximum Baseline Distance (wavelengths)\n";
  std::cout<<"\t-wproj               W-Projection Mode\n";
  std::cout<<"\n\n\n";
}

int main (int argc, char **argv) {


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

  init_dtype_cpx();
  
  double theta = false;
  double lambda = false;
  char *image = NULL;
  bool vis = false;

  int subgrid_size;
  int subgrid_margin;
  double winc;
  
  char *visfile = NULL;
  char *wkernfile = NULL;
  char *akernfile = NULL;

  double bl_min = 0;
  double bl_max = 1.7976931348623158e+308 ; // By default set to double limit.
  
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

  if (checkCmdLineFlag(argc, (const char **)argv, "wkernel") == 0){

    std::cout << "No W-Kernel File Specified!! \n";
    showHelp();
    return 0;
  }
  else {

    getCmdLineArgumentString(argc, (const char **) argv, "wkernel", &wkernfile);

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

  if (checkCmdLineFlag(argc, (const char **) argv, "bl_min"))
    bl_min = getCmdLineArgumentInt(argc, (const char **) argv, "bl_min");

  if (checkCmdLineFlag(argc, (const char **) argv, "bl_max"))
    bl_max = getCmdLineArgumentInt(argc, (const char **) argv, "bl_max");


  if((checkCmdLineFlag(argc, (const char **) argv, "subgrid") == 0 ||
     checkCmdLineFlag(argc, (const char **) argv, "margin") == 0 ||
     checkCmdLineFlag(argc, (const char **) argv, "winc") == 0) &&
     !checkCmdLineFlag(argc, (const char **) argv, "wproj")) {

    std::cout << "Missing W-Towers Parameters. Check subgrid/margin/winc\n";
    showHelp();
    return 0;
  }
  else {

    subgrid_size = getCmdLineArgumentInt(argc, (const char **) argv, "subgrid");
    subgrid_margin = getCmdLineArgumentInt(argc, (const char **) argv, "margin");
    winc = getCmdLineArgumentFloat(argc, (const char **) argv, "winc");

  }

  //File I/O

  
  int grid_size = floor(lambda * theta);

  const char* visfile_c = visfile;
  const char* wkernfile_c = wkernfile;

  cuDoubleComplex *image_host, *grid_dev;

  //May as well allocate our host image now for when we move it back.
  image_host = (cuDoubleComplex*)malloc(grid_size * grid_size * sizeof(cuDoubleComplex));

  //  int threads_block = prop[1].maxThreadsPerBlock
  
  if(checkCmdLineFlag(argc, (const char **) argv, "wproj")){
    //Call our W-Projection wrapper.
    if(checkCmdLineFlag(argc, (const char **) argv, "flat")){

      std::cout << "W-Projection Gridder(on flattened dataset)... \n";
      std::cout << "BL Max: " << bl_max << " BL Min: " << bl_min << " \n";
      wprojection_CUDA_flat(visfile_c, wkernfile_c, grid_size, theta, lambda,
		       bl_min, bl_max, prop[0].maxThreadsPerBlock);


    } else {
      std::cout << "W-Projection Gridder... \n";
      std::cout << "BL Max: " << bl_max << " BL Min: " << bl_min << " \n";
      wprojection_CUDA(visfile_c, wkernfile_c, grid_size, theta, lambda,
		       bl_min, bl_max, prop[0].maxThreadsPerBlock);
    }
  }
  else {
    //Call our W-Towers wrapper.
    std::cout << "W-Towers Gridder... \n";
    wtowers_CUDA(visfile_c, wkernfile_c, grid_size, theta, lambda, bl_min, bl_max,
	       subgrid_size, subgrid_margin, winc);

  }
  
  


  

  return 0;

}
