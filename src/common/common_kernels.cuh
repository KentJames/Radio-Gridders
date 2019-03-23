#ifndef COMMON_H
#define COMMON_H

__global__ void fresnel_subimg_mul(cuDoubleComplex *subgrid,
				   cuDoubleComplex *fresnel,
				   cuDoubleComplex *subimg,
				   int n,
				   int wp);

__global__ void w0_transfer_kernel(cuDoubleComplex *grid, cuDoubleComplex *base, int exp, int size);

__global__ void add_subs2main_kernel(cuDoubleComplex *main, cuDoubleComplex *subs,
				     int main_size, int sub_size, int sub_margin,
				     int chunk_count, int chunk_size);


template <typename T>
__global__ void fft_shift_kernel(T *grid, int size){

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x<size/2 && y <size){

    int ix0 = y * size + x;
    int ix1 = (ix0 + (size + 1) * (size/2)) % (size*size);

    T temp = grid[ix0];
    grid[ix0] = grid[ix1];
    grid[ix1] = temp;
  }
}

#endif