#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <limits>
#include <cmath>
#include <cuda.h>
#include "base_info.h"
#include "get_random.cu"
#include "matrix_cal.cu"

#define N (1024 * 4)

int main(){
    float gpu_elapsed_time_ms;
    cudaStream_t cuda_stream[2];
    cudaStreamCreateWithFlags(&cuda_stream[1], cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cuda_stream[0], cudaStreamNonBlocking);

    // make event
    cudaEvent_t start, stop, start1, stop1;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);

    // device matrix malloc
    int *d_A, *d_B, *d_C, *d_D, *d_E;
    cudaMalloc((void **) &d_A, sizeof(int)*N*N);
    cudaMalloc((void **) &d_B, sizeof(int)*N*N);
    cudaMalloc((void **) &d_C, sizeof(int)*N*N);
    cudaMalloc((void **) &d_D, sizeof(int)*N*N);
    cudaMalloc((void **) &d_E, sizeof(int)*N*N);

    dim3 dimGrid(GRID_SIZE, GRID_SIZE);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    int seed;
    seed = get_seed();
    d_rand_matrix<<<1, 1>>>(seed, d_A, N);
    seed = get_seed();
    d_rand_matrix<<<1, 1>>>(seed, d_B, N);
    seed = get_seed();
    d_rand_matrix<<<1, 1>>>(seed, d_D, N);
    seed = get_seed();
    d_rand_matrix<<<1, 1>>>(seed, d_E, N);
    cudaDeviceSynchronize();

    char c = 0;
    printf("set done");
    while((c = getchar()) != 'y');

    // record matrix multiple
    cudaEventRecord(start, cuda_stream[0]);
    cudaEventRecord(start1, cuda_stream[1]);
    
    
    d_mm_normal<<<dimGrid, dimBlock, 0, cuda_stream[0]>>>(d_A, d_B, d_C, N);
    // d_mm_normal<<<dimGrid, dimBlock, 0, cuda_stream[1]>>>(d_D, d_E, d_C, N);
    cudaDeviceSynchronize();

    cudaEventRecord(stop, cuda_stream[0]);
    cudaEventRecord(stop1, cuda_stream[1]);
    cudaEventSynchronize(stop);
    cudaEventSynchronize(stop1);

    // calculate elapsed time
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    
    printf("normal mm : %f ms\n", gpu_elapsed_time_ms);
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start1, stop1);
    
    printf("normal mm : %f ms\n", gpu_elapsed_time_ms);
    
    // free
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}