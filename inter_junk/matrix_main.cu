#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <random>
#include <limits>
#include <cmath>
#include <cuda.h>
#include "base_info.h"
#include "get_random.cu"
#include "matrix_cal.cu"

#include <sys/time.h>
#include <time.h>

uint64_t monotonic_time() {
    struct timespec timespec;
    clock_gettime(CLOCK_MONOTONIC, &timespec);
    return timespec.tv_sec * 1000 * 1000 * 1000 + timespec.tv_nsec;
}

// thread run as same time
bool flag = false;
bool cache_control = false;


void *d_mt_thread(void *arg){
    cudaStream_t cuda_stream;
    cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, INT_MAX);
    while(flag){
        // make random seed
        int seed = dis(gen);

        int *d_A, *d_B;
        cudaMalloc((void **) &d_A, sizeof(int)*N*N);
        cudaMalloc((void **) &d_B, sizeof(int)*N*N);

        dim3 dimGrid((N - 1) / GRID_SIZE + 1, (N - 1) / GRID_SIZE + 1, 1);
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

        d_rand_matrix<<<1, 1>>>(seed, d_A);
        d_rand_matrix<<<1, 1>>>(seed, d_B);
        cudaDeviceSynchronize();

        d_matrix_transpose<<<dimGrid, dimBlock, 0, cuda_stream>>>(d_A, d_B);
        cudaDeviceSynchronize();

        cudaFree(d_A);
        cudaFree(d_B);
    }
    return 0;
}

void *d_mm_thread(void *arg){
    float gpu_elapsed_time_ms;
    cudaStream_t cuda_stream;
    cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking);

    // make random seed
    int seed = get_seed();

    // make event
    uint64_t st, ed;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // device matrix malloc
    int *d_A, *d_B, *d_C;
    cudaMalloc((void **) &d_A, sizeof(int)*N*N);
    cudaMalloc((void **) &d_B, sizeof(int)*N*N);
    cudaMalloc((void **) &d_C, sizeof(int)*N*N);

    dim3 dimGrid(GRID_SIZE, GRID_SIZE);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    d_rand_matrix<<<1, 1>>>(seed, d_A);
    d_rand_matrix<<<1, 1>>>(seed, d_B);
    cudaDeviceSynchronize();

    while(!flag);

    // record matrix multiple
    cudaEventRecord(start, cuda_stream);
    st = monotonic_time();
    
    d_mm_normal<<<dimGrid, dimBlock, 0, cuda_stream>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    cudaEventRecord(stop, cuda_stream);
    ed = monotonic_time();
    cudaEventSynchronize(stop);

    // calculate elapsed time
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    printf("normal mm : %f ms\n", gpu_elapsed_time_ms);
    // printf("mono %d normal mm : %lu ms\n", seed, ed-st);
    
    // free
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}


void *d_mm_shared_thread(void *arg){
    float gpu_elapsed_time_ms;
    cudaStream_t cuda_stream;
    cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking);

    // make random seed
    int seed = get_seed();

    // make event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // device matrix malloc
    int *d_A, *d_B, *d_C;
    cudaMalloc((void **) &d_A, sizeof(int)*N*N);
    cudaMalloc((void **) &d_B, sizeof(int)*N*N);
    cudaMalloc((void **) &d_C, sizeof(int)*N*N);

    dim3 dimGrid(GRID_SIZE, GRID_SIZE);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    d_rand_matrix<<<1, 1>>>(seed, d_A);
    d_rand_matrix<<<1, 1>>>(seed, d_B);
    cudaDeviceSynchronize();

    while(!flag);

    cudaEventRecord(start, cuda_stream);
    d_mm_shared_mem<<<dimGrid, dimBlock, 0, cuda_stream>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();
    
    cudaEventRecord(stop, cuda_stream);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    printf("shared memory mm : %f ms\n", gpu_elapsed_time_ms);

    // free
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}


void *h_mm_thread(void *arg){
    int seed = get_seed();

    int *h_A, *h_B, *h_C;
    cudaMallocHost((void **) &h_A, sizeof(int)*N*N);
    cudaMallocHost((void **) &h_B, sizeof(int)*N*N);
    cudaMallocHost((void **) &h_C, sizeof(int)*N*N);
    h_A = h_rand_matrix(h_A);
    h_B = h_rand_matrix(h_B);

    float cpu_elapsed_time_ms;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    h_C = h_mm(h_A, h_B, h_C);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&cpu_elapsed_time_ms, start, stop);
    printf("%d host mm : %f ms\n", seed, cpu_elapsed_time_ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(h_A);
    cudaFree(h_B);
    cudaFree(h_C);

    return 0;
}

int main(void) {
    int arg = 0;
    int thread_id;
    int status;

    pthread_t kernel_n[KERNEL_CNT];
    pthread_t dirty;

    for (int i = 0; i < KERNEL_CNT; i++){
        thread_id = pthread_create(&kernel_n[i], NULL, d_mm_thread, (void*)&arg);
        if (thread_id != 0){
            printf("pthread create error : %d", thread_id);
            exit(0);
        }
    }

    flag = true;    

    for (int i = 0; i < KERNEL_CNT; i++){
        thread_id = pthread_join(kernel_n[i], (void **)&status);
        if (thread_id != 0){
            printf("pthread join error : %d", thread_id);
            exit(0);
        }
    }

    flag = false;

// ========================================================================

    for (int i = 0; i < KERNEL_CNT; i++){
        thread_id = pthread_create(&kernel_n[i], NULL, d_mm_shared_thread, (void*)&arg);
        if (thread_id != 0){
            printf("pthread create error : %d", thread_id);
            exit(0);
        }
    }

    flag = true;
    if(cache_control) pthread_create(&dirty, NULL, d_mt_thread, (void*)&arg);

    for (int i = 0; i < KERNEL_CNT; i++){
        thread_id = pthread_join(kernel_n[i], (void **)&status);
        if (thread_id != 0){
            printf("pthread join error : %d", thread_id);
            exit(0);
        }
    }
    flag = false;

    // ======================================================
    for (int i = 0; i < KERNEL_CNT; i++){
        thread_id = pthread_create(&kernel_n[i], NULL, d_mm_thread, (void*)&arg);
        if (thread_id != 0){
            printf("pthread create error : %d", thread_id);
            exit(0);
        }
    }

    flag = true;    

    for (int i = 0; i < KERNEL_CNT; i++){
        thread_id = pthread_join(kernel_n[i], (void **)&status);
        if (thread_id != 0){
            printf("pthread join error : %d", thread_id);
            exit(0);
        }
    }

    flag = false;

    // // =========================== cpu?

    // for (int i = 0; i < KERNEL_CNT; i++){
    //     thread_id = pthread_create(&kernel_n[i], NULL, h_mm_thread, (void*)&arg);
    //     if (thread_id){
    //         printf("pthread create error : %d", thread_id);
    //         exit(0);
    //     }
    // }

    // flag = true;
    // for (int i = 0; i < KERNEL_CNT; i++){
    //     thread_id = pthread_join(kernel_n[i], (void **)&status);
    //     if (thread_id){
    //         printf("pthread create error : %d", thread_id);
    //         exit(0);
    //     }
    // }
    // flag = false;
}