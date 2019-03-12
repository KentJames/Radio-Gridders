#include <iostream>
#include <vector>

#include "wstack_common.cuh"
#include "helper_string.h"


std::complex<double> run_test_predict(const std::vector<double>& points,
				      double theta,
				      double lambda,
				      double u,
				      double v,
				      double w,
				      double du,
				      double dw,
				      int support_uv,
				      int support_w,
				      double x0,
				      struct sep_kernel_data *sepkern_uv,
				      struct sep_kernel_data *sepkern_w,
				      struct sep_kernel_data *sepkern_lm,
				      struct sep_kernel_data *sepkern_n){

    
    return wstack_predict(theta,lambda,points,u,v,w,du,dw,support_uv,support_w,x0,
			  sepkern_uv,sepkern_w,sepkern_lm,sepkern_n);
    

}





int main(int argc, char **argv){


    double theta;
    double lambda;


    if (checkCmdLineFlag(argc, (const char **) argv, "theta") == 0 ||
	checkCmdLineFlag(argc, (const char **) argv, "lambda") == 0) {
	std::cout << "No theta or lambda specified!\n";
	//showHelp();
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


    struct sep_kernel_data *sepkern_uv = (struct sep_kernel_data *)malloc(sizeof(struct sep_kernel_data));
    struct sep_kernel_data *sepkern_w = (struct sep_kernel_data*)malloc(sizeof(struct sep_kernel_data));
    struct sep_kernel_data *sepkern_lm = (struct sep_kernel_data*)malloc(sizeof(struct sep_kernel_data));
    struct sep_kernel_data *sepkern_n = (struct sep_kernel_data*)malloc(sizeof(struct sep_kernel_data));
    
    std::cout << "Loading Kernel...";
    load_sep_kern("./kernels/sze/sepkern_uv.hdf5",sepkern_uv);
    std::cout << "Loading W Kernel...";
    load_sep_kern("./kernels/sze/sepkern_w.hdf5",sepkern_w);

    std::cout << "Loading AA Kernel...";
    load_sep_kern("./kernels/sze/sepkern_lm.hdf5", sepkern_lm);
    std::cout << "Loading AA Kernel...";
    load_sep_kern("./kernels/sze/sepkern_n.hdf5", sepkern_n);


    double du = sepkern_uv->du;
    double dw = sepkern_w->dw;
    double x0 = sepkern_lm->x0;

    int support_uv = sepkern_uv->size;
    int support_w = sepkern_w->size;
    
    
    std::vector<double> testcard_points = generate_testcard_dataset(theta);
    // std::vector<double> testcard_points = {0.0,0.0};

    std::complex<double> viswstack = run_test_predict(testcard_points, theta, lambda,
						      0.0,0.0,0.0,
						      du, dw,
						      support_uv, support_w,
						      x0,
						      sepkern_uv,
						      sepkern_w,
						      sepkern_lm,
						      sepkern_n
						      );
    std::complex<double> vis_dft = predict_visibility_quantized(testcard_points,
								theta,
								lambda,
								0.0, 0.0, 0.0);

    std::complex<double> error = std::abs(viswstack - vis_dft) / (testcard_points.size()/2);
    std::cout << "W-Stack Prediction: " << viswstack << "\n";
    std::cout << "DFT Prediction: " << vis_dft << "\n";
    std::cout << "Error: " << error << "\n";
   
    viswstack = run_test_predict(testcard_points, theta, lambda,
				 50.0,130.0,dw+0.00002,
				 du, dw,
				 support_uv, support_w,
				 x0,
				 sepkern_uv,
				 sepkern_w,
				 sepkern_lm,
				 sepkern_n
				 );
    vis_dft = predict_visibility_quantized(testcard_points,
					   theta,
					   lambda,
					   50.0, 130.0, dw+0.00002);
    error = std::abs(viswstack - vis_dft) / (testcard_points.size()/2);
    std::cout << "W-Stack Prediction: " << viswstack << "\n";
    std::cout << "DFT Prediction: " << vis_dft << "\n";
    std::cout << "Error: " << error << "\n";
   


    return 0;

}


