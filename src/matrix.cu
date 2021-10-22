/*
    Matrix.cu
*/
#include "../include/matrix.cuh"

void printMat(float *A, int n)
{
    printf("A := \n");
    for(int row = 0; row < n; row++)
    {
        for(int col = 0; col < n; col++)
        {
            float Areg = A[row + col * n];
            if(col == 0)
            {
                printf("| %lf ", Areg);
            }else if(col == n-1){
                printf("%lf |\n", Areg);
            }else{
                printf("%lf ", Areg);
            }
        }
    }
}

void printVect(float *Vec, int n)
{
    printf("b := \n");
    for(int col = 0; col < n; col++)
    {
        if(col == 0)
        {
            printf("[ %lf ", Vec[col]);
        }else if(col == n-1){
            printf("%lf ]\n",Vec[col]);
        }else{
            printf("%lf ", Vec[col]);
        }
    }
}

__global__ void printResult(float *Vec, int size, int iter)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id == 0)
    {
        printf("x(%d) :=", iter);
        for(int col = 0; col < size; col++)
        {
            if(col == 0)
            {
                printf("[ %lf ", Vec[col]);
            }else if(col == size-1){
                printf("%lf ]\n",Vec[col]);
            }else{
                printf("%lf ", Vec[col]);
            }
        }
    }
}