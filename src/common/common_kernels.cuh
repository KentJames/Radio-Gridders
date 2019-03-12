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

__global__ void fft_shift_kernel(cuDoubleComplex *grid, int size);


#endif