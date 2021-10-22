/*
 Matrix.cuh
*/
#include <stdio.h>
#include <cuda.h>
#include <math.h>

void printMat(float *A, int n);
void printVect(float *Vec, int n);

__global__ void printResult(float *Vec, int size, int iter);


