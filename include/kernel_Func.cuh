#ifndef KERNEL_FUNC_CUH
#define KERNEL_FUNC_CUH
#include<cuda.h>
#include<curand_kernel.h>
#include <cuda_runtime.h>

#define BLOCK_DIM_VEC 256

const float Tolerance = 1e-2;
// global variables
__device__ float dev_Beta;
__device__ float dev_Alpha;
__device__ float dev_r_norm;
__device__ float dev_r_norm_old;

__global__ void cg_set_r_norm(float *a, float *b, int size);
__global__ void cg_sdot(float *a, float *b, float *out, int size);
__global__ void cg_devide(float *num, float *den, float *out, int size);
__global__ void cg_update_ALXR(float *den, float *d_p, float *d_x, float *d_r, float *d_Ax, int size);
__global__ void cg_update_BeP(float *d_p, float *d_r, int size);

#endif