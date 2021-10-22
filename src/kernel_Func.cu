/*
kernel functions for cucg test 
*/
#include "../include/kernel_Func.cuh"


__global__ void cg_devide(float *num, float *den, float *out, int size)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id == 0)
    {
        *out = *num / *den;
    }
}

__global__ void cg_set_r_norm(float *a, float *b, int size)
{
    __shared__ float shared_tmp[BLOCK_DIM_VEC];

    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    shared_tmp[threadIdx.x] = 0.0;
    if(id == 0)
    {
        dev_r_norm = 0.0;
    }

    if(id < size)
    {
        shared_tmp[threadIdx.x] = a[id] * b[id];
    }else{
        shared_tmp[threadIdx.x] = 0.0;
    }

    for(int i = blockDim.x / 2; i >= i; i = i /2)
    {
        __syncthreads();
        if(threadIdx.x < i)
        {
            shared_tmp[threadIdx.x] += shared_tmp[threadIdx.x + 1];
        }
    }

    if(threadIdx.x == 0)
    {
        atomicAdd(&dev_r_norm, shared_tmp[0]);
    }
}

__global__ void cg_sdot(float *a, float *b, float *out, int size)
{
    __shared__ float shared_tmp[BLOCK_DIM_VEC];

    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    shared_tmp[threadIdx.x] = 0.0;
    if(id == 0)
    {
        *out = 0.0;
    }

    if(id < size)
    {
        shared_tmp[threadIdx.x] = a[id] * b[id];
    }else{
        shared_tmp[threadIdx.x] = 0.0;
    }

    for(int i = blockDim.x / 2; i >= i; i = i /2)
    {
        __syncthreads();
        if(threadIdx.x < i)
        {
            shared_tmp[threadIdx.x] += shared_tmp[threadIdx.x + 1];
        }
    }

    if(threadIdx.x == 0)
    {
        atomicAdd(out, shared_tmp[0]);
    }
}

__global__ void cg_update_ALXR(float *den, float *d_p, float *d_x, float *d_r, float *d_Ax, int size)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id == 0)
    {
        dev_Alpha = dev_r_norm / *den;
        dev_r_norm_old = dev_r_norm;
    }
    __syncthreads( );

    if(id < size)
    {
        d_x[id] = d_x[id] + dev_Alpha * d_p[id];
        d_r[id] = d_r[id] - dev_Alpha * d_Ax[id];
    }
}

__global__ void cg_update_BeP(float *d_p, float *d_r, int size)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id == 0)
    {
        dev_Beta = dev_r_norm_old / dev_r_norm;
    }
    __syncthreads( );

    if(id < size)
    {
        d_p[id] = d_r[id] + dev_Beta * d_p[id];
    }
}