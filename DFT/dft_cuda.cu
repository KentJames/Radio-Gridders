#include <iostream>
#include <cstdlib>

//CUDA Includes
#include <cuComplex.h>
#include "cuda.h"
#include "cuda_runtime_api.h"

//Our include
#include "dft_common.h"



// This calculates the DFT at a specific point on the grid.
// It adds to a local register and then does an atomic add to device memory.
__device__ cuDoubleComplex calculate_dft_sum(struct vis_data *vis, double l, double m){

  //nvcc should put this in a register.
  cuDoubleComplex grid_point = make_cuDoubleComplex(0.0,0.0);
  
  for (int bl = 0; bl < vis->bl_count; ++bl){

    for (int time = 0; time < vis->bl[bl].time_count; ++time){

      for (int freq = 0; freq < vis->bl[bl].freq_count; ++freq){

	//This step is quite convoluted due to mixing C and CUDA compelx datatypes..
	cuDoubleComplex visibility;
	double __complex__ visibility_c = vis->bl[bl].vis[time*vis->bl[bl].freq_count + freq];
	memcpy(&visibility, &visibility_c, sizeof(double __complex__));
	

	//nvcc should optimise this section.
	double subang1 = m * vis->bl[bl].uvw[time*vis->bl[bl].freq_count + freq];
	double subang2 = l * vis->bl[bl].uvw[time*vis->bl[bl].freq_count + freq + 1];
	double subang3 = (sqrtf(1-l*l-m*m)-1) * vis->bl[bl].uvw[time*vis->bl[bl].freq_count +
								freq + 2];

	double angle = 2 * M_PI * subang1 + subang2 + subang3;

	double real_p = cuCreal(visibility) * cos(angle) + cuCimag(visibility) * sin(angle);
	double complex_p = -cuCreal(visibility) * sin(angle) + cuCimag(visibility) * cos(angle);

	//Add these to our grid_point so far.
	grid_point = cuCadd(grid_point, make_cuDoubleComplex(real_p, complex_p));
							       

      }
    }
  }
  

  return grid_point; //Placeholder
}
