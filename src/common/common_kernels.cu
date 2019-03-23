//Our Includes
#include "radio.cuh"
#include "common_kernels.cuh"
/*
  
  This file defines various kernels that I have found useful enough to use
  across multiple projects.

*/


//Elementwise multiplication of subimg with fresnel. 
__global__ void fresnel_subimg_mul(cuDoubleComplex *subgrid,
				   cuDoubleComplex *fresnel,
				   cuDoubleComplex *subimg,
				   int n,
				   int wp){

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x < n && y < n){
    cuDoubleComplex wtrans = cu_cpow(fresnel[y * n + x], make_cuDoubleComplex(wp,0.0));
    subimg[y * n + x] = cuCmul(fresnel[y * n + x], subimg[y * n + x]);
    subimg[y * n + x] = cuCadd(subimg[y * n + x], subgrid[y * n + x]);
    subgrid[y * n + x] = make_cuDoubleComplex(0.0,0.0);
  }
}

//Transforms grid to w==0 plane.
__global__ void w0_transfer_kernel(cuDoubleComplex *grid, cuDoubleComplex *base, int exp, int size){

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x<size && y<size){
    cuDoubleComplex wt0 = cu_cpow(base[y * size + x], make_cuDoubleComplex(exp, 0.0));
    grid[y * size + x] = cuCdiv(grid[y * size + x],wt0);
  }
}

//Set the total grid size to cover every pixel in the main grid.
__global__ void add_subs2main_kernel(cuDoubleComplex *main, cuDoubleComplex *subs,
				     int main_size, int sub_size, int sub_margin,
				     int chunk_count, int chunk_size){


  int x = (blockDim.x * blockIdx.x + threadIdx.x) - main_size/2;
  int y = (blockDim.y * blockIdx.y + threadIdx.y) - main_size/2;
  
  for(int cy = 0; cy < chunk_count; ++cy){
    for(int cx = 0; cx < chunk_count; ++cx){
      
      int x_min = chunk_size*cx - main_size/2; //- sub_size/2;
      int y_min = chunk_size*cy - main_size/2; //- sub_size/2;
      
      int x0 = x_min - sub_margin/2;
      int y0 = y_min - sub_margin/2;

      int x1 = x0 + sub_size;
      int y1 = y0 + sub_size;

      if (x0 < -main_size/2) { x0 = -main_size/2; }
      if (y0 < -main_size/2) { y0 = -main_size/2; }
      if (x1 > main_size/2) { x1 = main_size/2; }
      if (y1 > main_size/2) { y1 = main_size/2; }
      cuDoubleComplex *main_mid = main + (main_size + 1)*main_size/2;
      if(y>= y0 && y < y1 && x>= x0 && x < x1){
	int y_s = y - y_min + sub_margin/2;
	int x_s = x - x_min + sub_margin/2;
	cuDoubleComplex *sub_offset = subs + (((cy * chunk_count) + cx) * sub_size * sub_size);
	cuDoubleComplex normalised_number = cuCdiv(sub_offset[y_s*sub_size + x_s],
						   make_cuDoubleComplex(sub_size * sub_size, 0.0));
	main_mid[y * main_size + x] = cuCadd(main_mid[y * main_size + x], normalised_number);
	//Not sure if this is good style. 1) Calculate offset. 2) Dereference via array notation
      }
    }
  }  
}


// //Shifts a 2D grid to be in the right place for an FFT.
// template <typename T>
// __global__ void fft_shift_kernel(T *grid, int size){

//   int x = blockIdx.x * blockDim.x + threadIdx.x;
//   int y = blockIdx.y * blockDim.y + threadIdx.y;

//   if(x<size/2 && y <size){

//     int ix0 = y * size + x;
//     int ix1 = (ix0 + (size + 1) * (size/2)) % (size*size);

//     T temp = grid[ix0];
//     grid[ix0] = grid[ix1];
//     grid[ix1] = temp;
//   }
// }
