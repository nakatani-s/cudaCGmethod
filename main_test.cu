/*
  自作　cuda共役勾配法関数のテスト用
*/ 
#include<iostream>
#include <stdio.h>
#include <fstream>
#include <math.h>
#include <cuda.h>
#include <time.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <errno.h>
#include <string.h>

#include "include/kernel_Func.cuh"
#include "include/matrix.cuh"
#define CHECK(call)                                                  \
{                                                                    \
    const cudaError_t error = call;                                  \
    if (error != cudaSuccess)                                        \
    {                                                                \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                \
        printf("code:%d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                          \
        exit(1);                                                     \
    }                                                                \
}
#define CHECK_CUBLAS(call,str)                                                        \
{                                                                                     \
    if ( call != CUBLAS_STATUS_SUCCESS)                                               \
    {                                                                                 \
        printf("CUBLAS Error: %s : %s %d\n", str, __FILE__, __LINE__);                \
        exit(1);                                                                      \
    }                                                                                 \
}
#define CHECK_CUSOLVER(call,str)                                                      \
{                                                                                     \
    if ( call != CUSOLVER_STATUS_SUCCESS)                                             \
    {                                                                                 \
        printf("CUBLAS Error: %s : %s %d\n", str, __FILE__, __LINE__);                \
        exit(1);                                                                      \
    }                                                                                 \
}

int main(int argc, char **argv)
{
    unsigned int MAX_ITER = 100;
    unsigned int dim_X = 3;
    unsigned int dim_A = dim_X * dim_X;

    dim3 vec_block_dim(BLOCK_DIM_VEC);
    dim3 vec_grid_dim((dim_X + BLOCK_DIM_VEC -1) / BLOCK_DIM_VEC);

    float *hstA, *devA; // hstA , devA <-- Hees相当
    float *hstX, *devX; // hstX , devX <-- 初期値
    float *hstB, *devB; // hstB , devB <-- 勾配ベクトル相当

    hstA = (float*)malloc(sizeof(float) * dim_A);
    hstX = (float*)malloc(sizeof(float) * dim_X);
    hstB = (float*)malloc(sizeof(float) * dim_X);

    CHECK( cudaMalloc(&devA, sizeof(float) * dim_A) );
    CHECK( cudaMalloc(&devX, sizeof(float) * dim_X) );
    CHECK( cudaMalloc(&devB, sizeof(float) * dim_X) );

    float matA[dim_A] = {1.0, 0.0, 0.0,
                         0.0, 2.0, 0.0,
                         0.0, 0.0, 1.0};
    float VecX[dim_X] = { };
    float VecB[dim_X] = {4.0, 5.0, 6.0};
    hstA = matA;
    hstB = VecB;
    hstX = VecX;

    CHECK( cudaMemcpy(devA, hstA, sizeof(float) * dim_A, cudaMemcpyHostToDevice) );
    CHECK( cudaMemcpy(devX, hstX, sizeof(float) * dim_X, cudaMemcpyHostToDevice) );
    CHECK( cudaMemcpy(devB, hstB, sizeof(float) * dim_X, cudaMemcpyHostToDevice) );
    
    printMat(hstA, dim_X);
    printVect(hstA, dim_X);
    printVect(hstX, dim_X);

    float *devP;
    float *devR;
    float *devAx;

    CHECK( cudaMalloc(&devP, sizeof(float) * dim_X) );
    CHECK( cudaMalloc(&devR, sizeof(float) * dim_X) );
    CHECK( cudaMalloc(&devAx, sizeof(float) * dim_A) );

    float *dev_temp_scal;
    CHECK( cudaMalloc(&dev_temp_scal, sizeof(float)) );

    
    // cuBlasの定義
    cublasHandle_t cublasH = 0;
    cublasCreate(&cublasH);
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    cublasSideMode_t side = CUBLAS_SIDE_LEFT;
    cublasOperation_t trans = CUBLAS_OP_T;
    cublasOperation_t trans_N = CUBLAS_OP_N;
    cublasFillMode_t uplo_QR = CUBLAS_FILL_MODE_UPPER;
    cublasDiagType_t cub_diag = CUBLAS_DIAG_NON_UNIT;
    
    const float alpha = 1.0;
    const float beta = 0.0;

    cusolverDnHandle_t cusolverH = NULL;
    cusolverDnCreate(&cusolverH);

    free(hstA);
    free(hstB);
    free(hstX);
    // ここから、CU−Newton-Like methodで必要な処理
    // 残差ベクトルの処理，本例題では初期値0ベクトルのため，r = b - Ax = b となるので，r = bとしたが、本来は計算が必要
    clock_t start_t, stop_t;
    float hstTol = 999.0;
    float operation_time;
    start_t = clock( );
    CHECK( cudaMemcpy(devR, hstB, sizeof(float)*dim_X, cudaMemcpyHostToDevice) );
    CHECK( cudaMemcpy(devP, hstB, sizeof(float)*dim_X, cudaMemcpyHostToDevice) );

    cg_set_r_norm<<<vec_grid_dim, vec_block_dim>>>(devR, devP, dim_X);
    for(int iter = 0; iter < MAX_ITER; iter++)
    {
        // devAx = devA * devP
        CHECK_CUBLAS(cublasSgemm(cublasH, trans_N, trans_N, dim_X, 1, dim_X, &alpha, devA, dim_X, devP, 1, &beta, devAx, dim_X), "TensorOperation Failed");
        
        // den = devP^T * devA * devP
        cg_sdot<<<vec_grid_dim, vec_block_dim>>>(devP,  devAx, dev_temp_scal, dim_X);
        
        // alpha_k = ...., x_{k+1} = ...., r_{k+1} = .....
        cg_update_ALXR<<<vec_grid_dim, vec_block_dim>>>(dev_temp_scal, devP, devX, devR, devAx, dim_X);
        
        // dev_r_norm <-- r_{k+1}^T * r_{k+1}
        cg_set_r_norm<<<vec_grid_dim, vec_block_dim>>>(devR, devR, dim_X);

        CHECK( cudaMemcpyFromSymbol(&hstTol, dev_r_norm, sizeof(float)) );
        if(hstTol < Tolerance)
        {
            stop_t = clock( );
            operation_time = stop_t - start_t;
            printf("Final Iterations :: %d  Computation Time :: %f\n", iter, operation_time / CLOCKS_PER_SEC);
            printResult<<<1,1>>>(devX, dim_X, iter);
            break;
        }else{
            // printf("Iterations :: %d\n", iter);
            printResult<<<1,1>>>(devX, dim_X, iter);
        }

        // Beta_k = ....., p_{k+1} = .......
        cg_update_BeP<<<vec_grid_dim, vec_block_dim>>>(devP, devR, dim_X);

        // dev_r_norm <--- r_{k+1}~T * p_{k+1} 
        cg_set_r_norm<<<vec_grid_dim, vec_block_dim>>>(devR, devP, dim_X);
    }
    if(cublasH) cublasDestroy(cublasH);
    cudaDeviceReset( );
    printf("%s\n", cudaGetErrorString(cudaGetLastError()));
    return 0;
}