#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <ctime>
#include <chrono>

#include "wstack_common.h"
#include "helper_string.h"


std::vector<std::complex<double>> run_test_predict(const std::vector<double>& points,
						   double theta,
						   double lambda,
						   std::vector<double> uvw,
						   double du,
						   double dw,
						   int support_uv,
						   int support_w,
						   double x0,
						   struct sep_kernel_data *sepkern_uv,
						   struct sep_kernel_data *sepkern_w,
						   struct sep_kernel_data *sepkern_lm,
						   struct sep_kernel_data *sepkern_n){

    
    return wstack_predict(theta,lambda,points, uvw, du, dw, support_uv, support_w, x0,
			  sepkern_uv,sepkern_w,sepkern_lm,sepkern_n);
    

}

std::vector<std::vector<std::complex<double>>> run_test_predict_lines(const std::vector<double>& points,
								double theta,
								double lambda,
								std::vector<std::vector<double>> uvw,
								double du,
								double dw,
								int support_uv,
								int support_w,
								double x0,
								struct sep_kernel_data *sepkern_uv,
								struct sep_kernel_data *sepkern_w,
								struct sep_kernel_data *sepkern_lm,
								struct sep_kernel_data *sepkern_n){
    
    
    return wstack_predict_lines(theta,lambda,points, uvw, du, dw, support_uv, support_w, x0,
			  sepkern_uv,sepkern_w,sepkern_lm,sepkern_n);
    

}



int main(int argc, char **argv){


    double theta;
    double lambda;
    int npts = 40000000;

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
    
    std::cout << " TESTING WITH TEST-CARD: \n";
    std::vector<double> random_visibilities = generate_random_visibilities_(theta,lambda,dw,npts);
    std::vector<double> testcard_points = generate_testcard_dataset(theta);
    // std::vector<double> testcard_points = {0.0,0.0};
    
    std::vector<std::complex<double> > viswstack = run_test_predict(testcard_points, theta, lambda,
								   random_visibilities,
								   du, dw,
								   support_uv, support_w,
								   x0,
								   sepkern_uv,
								   sepkern_w,
								   sepkern_lm,
								   sepkern_n
								   );

    std::cout << " Calculating via DFT: \n" << std::flush;
    std::vector<std::complex<double>> vis_dft = predict_visibility_quantized_vec(testcard_points,
    								theta,
    								lambda,
    								random_visibilities);

    std::vector<double> error (viswstack.size(),0.0);
    std::transform(viswstack.begin(), viswstack.end(), vis_dft.begin(), error.begin(),
    		   [npts](const std::complex<double> wstack,
    			  const std::complex<double> dft)
    		   -> double { return std::abs(dft-wstack)/npts;});


    double agg_error = std::accumulate(error.begin(), error.end(), 0.0);
    std::cout << "Example error is: " << viswstack[186573] << " " << vis_dft[186573] << "\n";


    
    std::cout << "Aggregate Error: " << agg_error/error.size() << "\n\n";
 

    std::cout << " Testing using Lines of Visibilities: \n\n";

    std::vector<std::vector<double>> uvw_lines (12);
    uvw_lines[0] = generate_line_visibilities_(theta,lambda,5.0,dw,10000000);
    uvw_lines[1] = generate_line_visibilities_(theta,lambda,-10.0,dw,10000000);
    uvw_lines[2] = generate_line_visibilities_(theta,lambda,15.0,dw,10000000);
    uvw_lines[3] = generate_line_visibilities_(theta,lambda,24.5,dw,10000000);
    uvw_lines[4] = generate_line_visibilities_(theta,lambda,-178.0,dw,10000000);
    uvw_lines[5] = generate_line_visibilities_(theta,lambda,657.0,dw,10000000);
    uvw_lines[6] = generate_line_visibilities_(theta,lambda,-67.0,dw,10000000);
    uvw_lines[7] = generate_line_visibilities_(theta,lambda,-87.0,dw,10000000);
    uvw_lines[8] = generate_line_visibilities_(theta,lambda,65.0,dw,10000000);
    uvw_lines[9] = generate_line_visibilities_(theta,lambda,123.0,dw,10000000);
    uvw_lines[10] = generate_line_visibilities_(theta,lambda,98.0,dw,10000000);
    uvw_lines[11] = generate_line_visibilities_(theta,lambda,0.0,dw,10000000);
    
    std::vector<std::vector<std::complex<double>>> vis_lines = run_test_predict_lines(testcard_points, theta, lambda,
    									      uvw_lines, du, dw,
    									      support_uv, support_w,
    									      x0,
    									      sepkern_uv,
    									      sepkern_w,
    									      sepkern_lm,
    									      sepkern_n);

    agg_error = 0.0;
    std::size_t line_size = 0;
    for (int i = 0; i < uvw_lines.size(); ++i){
    	line_size += vis_lines[i].size();
    	std::vector<double> errorl (vis_lines[i].size(),0.0);
    	vis_dft = predict_visibility_quantized_vec(testcard_points,
    						   theta,
    						   lambda,
    						   uvw_lines[i]);
	
    	std::transform(vis_lines[i].begin(), vis_lines[i].end(), vis_dft.begin(), errorl.begin(),
    		       [npts](const std::complex<double> wstack,
    			      const std::complex<double> dft)
    		       -> double { return std::abs(dft-wstack)/npts;});
    	double agg_errori = std::accumulate(errorl.begin(), errorl.end(),0.0);
    	agg_error += agg_errori;
    }
    std::cout << "Aggregate Error(lines): " << agg_error/line_size << "\n";
    
    std::cout << "\nOdd corner cases: \n";

    std::vector<double> cornercase = {0,1.0/lambda};
    std::vector<double> locations (3,0.0);
    
    viswstack = run_test_predict(cornercase, theta, lambda,
    				 locations,
    				 du, dw,
    				 support_uv, support_w,
    				 x0,
    				 sepkern_uv,
    				 sepkern_w,
    				 sepkern_lm,
    				 sepkern_n
    				 );
    vis_dft = predict_visibility_quantized_vec(cornercase,
    					   theta,
    					   lambda,
    					   locations);

    std::vector<double> error_cc1(viswstack.size(),0.0);
    std::transform(viswstack.begin(), viswstack.end(), vis_dft.begin(), error_cc1.begin(),
    		   [npts](const std::complex<double> wstack,
    			  const std::complex<double> dft)
    		   -> double { return std::abs(dft-wstack);});

    agg_error = 0.0;
    agg_error = std::accumulate(error_cc1.begin(), error_cc1.end(), 0.0);
    
    std::cout << "Aggregate Error: " << agg_error/error.size() << "\n";

    locations[0] = 50.0;
    locations[1] = 130.0;
    locations[2] = dw-0.00002;

    
    viswstack = run_test_predict(cornercase, theta, lambda,
    				 locations,
    				 du, dw,
    				 support_uv, support_w,
    				 x0,
    				 sepkern_uv,
    				 sepkern_w,
    				 sepkern_lm,
    				 sepkern_n
    				 );
    vis_dft = predict_visibility_quantized_vec(cornercase,
    					       theta,
    					       lambda,
    					       locations);

    std::vector<double> error_cc2(viswstack.size(),0.0); 
    std::transform(viswstack.begin(), viswstack.end(), vis_dft.begin(), error_cc2.begin(),
    		   [npts](const std::complex<double> wstack,
    			  const std::complex<double> dft)
    		   -> double { return std::abs(dft-wstack);});

    double agg_err2;
    agg_error = std::accumulate(error_cc2.begin(), error_cc2.end(), 0.0);
    
    std::cout << "Aggregate Error: " << agg_error/error.size() << "\n";
    
    return 0;

}


