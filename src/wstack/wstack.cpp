#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <complex>
#include <random>

//CUDA Includes

#include "helper_string.h"
#include "wstack_common.h"
#ifdef CUDA_ACCELERATION
#include "predict.cuh"
#endif

void showHelp(){

    std::cout<<"\ncuStacker v0.1\n";
    std::cout<<"James Kent <jck42@cam.ac.uk>\n";
    std::cout<<"A W-Stacking Implementation originally devised by Sze-Tan \n";
    std::cout<<"University of Cambridge, 2019\n\n";

    std::cout<<"General Parameters: \n";
    std::cout<<"\t-theta               Field of View (Radians)\n";
    std::cout<<"\t-lambda              Number of Wavelengths\n";
    std::cout<<"\t-npts                Number of random point sources(Default:10)\n";
    std::cout<<"\t-mode                Predict(0) or Image (1)\n\n";

    std::cout<<"Sze Tan Parameters: \n";
    std::cout<<"\t-sepkern_uv          Seperable Sze-Tan Grid Convolution Kernels for U/V\n";
    std::cout<<"\t-sepkern_w           Seperable Sze-Tan Grid Convolution Kernels for W\n";
    std::cout<<"\t-sepkern_lm          Seperable Sze-Tan Grid Correction Kernels for l/m\n";
    std::cout<<"\t-sepkern_n           Seperable Sze-Tan Grid Correction Kernels for n\n\n";

    std::cout<<"Predict Parameters: \n";
    std::cout<<"\t-pu                  Predict u value\n";
    std::cout<<"\t-pv                  Predict v value\n";
    std::cout<<"\t-pw                  Predict w value\n\n";

    std::cout<<"Compute Parameters: \n";
    std::cout<<"\t-cuda                CPU(0) or CUDA(1)\n";
    std::cout<<"\t-device              GPU Device to use (Default: 0)\n";

}

int main(int argc, char **argv) {

    if (checkCmdLineFlag(argc, (const char **)argv, "help")) {

	showHelp();
	return 0;
    }
    
#ifdef CUDA_ACCELERATION
    //Get information on GPU's in system.
    std::cout << "CUDA System Information: \n\n";
    int cuda_acceleration = 0;
    int numberofgpus;
    int dev_no = 0;
    

  
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


    if (checkCmdLineFlag(argc, (const char **) argv, "cuda")){
	cuda_acceleration = getCmdLineArgumentInt(argc, (const char **) argv, "cuda");
    }

    if (checkCmdLineFlag(argc, (const char **) argv, "device")){
	dev_no = getCmdLineArgumentInt(argc, (const char **) argv, "device");
    }

    cudaSetDevice(dev_no);
#endif

    
    double theta = false;
    double lambda = false;
    int mode = 0;
    int npts = 0; 
    double pu, pv, pw;
    pu = pv = pw = 0;

  

    char *sepkern_uv_file = NULL;
    char *sepkern_w_file = NULL;
    char *sepkern_lm_file = NULL;
    char *sepkern_n_file = NULL;

    
  
    if (checkCmdLineFlag(argc, (const char **) argv, "theta") == 0 ||
	checkCmdLineFlag(argc, (const char **) argv, "lambda") == 0) {
	std::cout << "No theta or lambda specified!\n";
	showHelp();
	return 0;
    }
    else {
	theta =  getCmdLineArgumentDouble(argc, (const char **) argv, "theta");
	lambda = getCmdLineArgumentDouble(argc, (const char **) argv, "lambda");
	if (theta < 0) {
	    std::cout << "Invalid theta value specified\n";
	    return 0;
	}
	if (lambda < 0) {
	    std::cout << "Invalid lambda value specified\n";
	    return 0;
	}
    }    

    if(checkCmdLineFlag(argc, (const char**) argv, "npts")){
	npts = getCmdLineArgumentInt(argc, (const char **) argv, "npts");
    }

    if (checkCmdLineFlag(argc, (const char **) argv, "sepkern_uv") == 0){

	std::cout << "No U/V Sze-Tan Grid Convolution Kernel File Specified!! \n";
	showHelp();
	return 0;
    }
    else {

	getCmdLineArgumentString(argc, (const char **) argv, "sepkern_uv", &sepkern_uv_file);

    } 


    if (checkCmdLineFlag(argc, (const char **) argv, "sepkern_w") == 0){

	std::cout << "No W Sze-Tan Grid Convolution Kernel File Specified!! \n";
	showHelp();
	return 0;
    }
    else {

	getCmdLineArgumentString(argc, (const char **) argv, "sepkern_w", &sepkern_w_file);

    } 

    if (checkCmdLineFlag(argc, (const char **) argv, "sepkern_lm") == 0){

	std::cout << "No l/m Sze-Tan Grid Correction Kernel File Specified!! \n";
	showHelp();
	return 0;
    }
    else {

	getCmdLineArgumentString(argc, (const char **) argv, "sepkern_lm", &sepkern_lm_file);

    } 


    if (checkCmdLineFlag(argc, (const char **) argv, "sepkern_n") == 0){

	std::cout << "No n Sze-Tan Grid Correction Kernel File Specified!! \n";
	showHelp();
	return 0;
    }
    else {

	getCmdLineArgumentString(argc, (const char **) argv, "sepkern_n", &sepkern_n_file);

    } 
  

    if (checkCmdLineFlag(argc, (const char **) argv, "mode") == 0){
	std::cout << "No mode specified!\n";
	showHelp();
	return 0;
    }
    else {
	mode = getCmdLineArgumentDouble(argc, (const char **) argv, "mode");
	if (mode < 0) {
	    std::cout << "Invalid theta value specified\n";
	    return 0;
	}      
    }


    struct sep_kernel_data *sepkern_uv = (struct sep_kernel_data*)malloc(sizeof(struct sep_kernel_data));
    struct sep_kernel_data *sepkern_w = (struct sep_kernel_data*)malloc(sizeof(struct sep_kernel_data));
    struct sep_kernel_data *sepkern_lm = (struct sep_kernel_data*)malloc(sizeof(struct sep_kernel_data));
    struct sep_kernel_data *sepkern_n = (struct sep_kernel_data*)malloc(sizeof(struct sep_kernel_data));

#ifdef CUDA_ACCELERATION
    if(cuda_acceleration){

	std::cout << "Loading Kernel...";
	load_sep_kern_CUDA(sepkern_uv_file,sepkern_uv);
	std::cout << "Loading W Kernel...";
	load_sep_kern_CUDA(sepkern_w_file,sepkern_w);
	
	std::cout << "Loading AA Kernel...";
	load_sep_kern_CUDA(sepkern_lm_file, sepkern_lm);
	std::cout << "Loading AA Kernel...";
	load_sep_kern_CUDA(sepkern_n_file, sepkern_n);
	
    } else {
#endif
	std::cout << "Loading Kernel...";
	load_sep_kern(sepkern_uv_file,sepkern_uv);
	std::cout << "Loading W Kernel...";
	load_sep_kern(sepkern_w_file,sepkern_w);
	
	std::cout << "Loading AA Kernel...";
	load_sep_kern(sepkern_lm_file, sepkern_lm);
	std::cout << "Loading AA Kernel...";
	load_sep_kern(sepkern_n_file, sepkern_n);
#ifdef CUDA_ACCELERATION
    } 
#endif
    double du = sepkern_uv->du;
    double dw = sepkern_w->dw;
    double x0 = sepkern_lm->x0;

    int support_uv = sepkern_uv->size;
    int support_w = sepkern_w->size;

    std::cout << "Optimum Parameters: " << "\n";
    std::cout << "dw: " << dw << "\n";
    std::cout << "du: " << du << "\n";
    
    
    /* 
       PREDICT
    */   
    if (mode == 0){


	std::complex<double> visq;
	std::complex<double> vis;
	std::vector<double> points = generate_random_points(npts, theta);
#ifdef CUDA_ACCELERATION
	if(cuda_acceleration){

	    // Placeholder for now.
	    // TODO: Add import from file.

	    std::vector<double> uvec(3,14.3);
	    std::vector<double> vvec(3,12.2);
	    std::vector<double> wvec(3,5.6);
	    std::vector<std::complex<double> > vis(3,{0.0,0.0});
	    vis = wstack_predict_cu_3D(theta,
				    lambda,
				    points,
				    uvec, vvec, wvec,
				    du, dw,
				    support_uv,
				    support_w,
				    x0,
				    sepkern_uv,
				    sepkern_w,
				    sepkern_lm,
				    sepkern_w);
				    
				    
	    
	    

	} else {
#endif	    
	    if (checkCmdLineFlag(argc, (const char **) argv, "pu") == 0 ||
		checkCmdLineFlag(argc, (const char **) argv, "pv") == 0 ||
		checkCmdLineFlag(argc, (const char **) argv, "pw") == 0) {
		std::cout << "No u/v/w co-ordinates specified for predict!\n";
		showHelp();	  
		return 0;
	    }
	    else {
		pu =  getCmdLineArgumentDouble(argc, (const char **) argv, "pu");
		pv =  getCmdLineArgumentDouble(argc, (const char **) argv, "pv");
		pw =  getCmdLineArgumentDouble(argc, (const char **) argv, "pw");
	    }
	
	    std::cout << "##### W Stacking #####\n"; 
	    std::cout << "Visibility at: " << pu << " " << pv << " " << pw << "\n";
	    visq = predict_visibility_quantized(points,theta,lambda,pu,pv,pw);
	    std::cout << "DFT Prediction: " << visq << "\n";
	    vis = wstack_predict(theta,
				 lambda,
				 points,
				 pu,
				 pv,
				 pw,
				 du,
				 dw,
				 support_uv,
				 support_w,
				 x0,
				 sepkern_uv,
				 sepkern_w,
				 sepkern_lm,
				 sepkern_n);
#ifdef CUDA_ACCELERATION
	}
#endif

	std::cout << "W-Stacks Prediction: " << vis << "\n";

	std::cout << "Error: " << std::abs(vis - visq) / npts << "\n";

      
      

	

    }
    /* 
       IMAGE
    */
    else {
	// Does nothing for now.

	



	
	return 0;
    }
    return 0;
}
