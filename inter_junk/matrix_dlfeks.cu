#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <random>
#include <limits>
#include <curand_kernel.h> 
#include <cmath>
#include <base_info.h

#define GRID_SIZE 16
#define BLOCK_SIZE 16
// 32 * 32
#define KERNEL_CNT 4
#define SQUARE_MATRIX_SIZE 1024*4
#define N SQUARE_MATRIX_SIZE

// thread tun as same time
bool flag = false;
bool cache_control = false;

__global__ void d_matrix_transpose(int *A, int *B){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    A[col * N + row] = B[row * N + col];
}



__global__ void d_mm_normal(int *A, int *B, int*C)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int sum = 0;
    for(int i = 0; i < N; i++) 
    {
        sum += A[row * N + i] * B[i * N + col];
    }
    C[row * N + col] = sum;
}

__global__ void d_mm_shared_mem(int *A, int *B, int*C)
{
    __shared__ int aTile[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int bTile[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int sum = 0;
    for (int tileN = 0; tileN < gridDim.x; tileN++){
        aTile[threadIdx.y][threadIdx.x] = A[row * N + threadIdx.x + tileN*BLOCK_SIZE];
        bTile[threadIdx.y][threadIdx.x] = B[(threadIdx.y + tileN*BLOCK_SIZE)*N + col];

        __syncthreads();
        for(int i = 0; i < BLOCK_SIZE; i++) 
        {
            sum += aTile[threadIdx.y][i] * bTile[i][threadIdx.x];
        }
    }
    C[row * N + col] = sum;
}

int* h_mm(int *A, int *B, int *C){
    for (int i = 0; i<N; i++){
        for (int j = 0; j<N; j++){
            int sum = 0;
            for (int k = 0; k<N; k++){
                sum += A[i*N + k] * B[k*N + j];
            }
            C[i*N + j] = sum;
        }
    }
    return C;
}

__device__ int get_rand_num(curandState_t *state, int A, int B){
    float rand_int = curand_uniform(state);
    rand_int = rand_int * (B - A) + A;
    return rand_int;
}

__global__ void d_rand_matrix(int seed, int* result){
    curandState_t state;
    curand_init(seed, 0, 0, &state);

	for(int i = 0; i < N*N; i++){
		result[i] = get_rand_num(&state, 0, 1024); 
	}
}

int* h_rand_matrix(int *matrix){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, 1048576);
    for(int i = 0; i < N*N; i++){
        matrix[i] = dis(gen);   
    }
    return matrix;
}

void *d_mt_thread(void *arg){
    cudaStream_t cuda_stream;
    cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, INT_MAX);
    while(flag && cache_control){
        // make random seed
        int seed = dis(gen);

        // device matrix malloc
        int *d_A, *d_B;
        cudaMalloc((void **) &d_A, sizeof(int)*N*N);
        cudaMalloc((void **) &d_B, sizeof(int)*N*N);

        dim3 dimGrid(GRID_SIZE, GRID_SIZE);
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

        d_rand_matrix<<<1, 1>>>(seed, d_A);
        d_rand_matrix<<<1, 1>>>(seed, d_B);

        d_matrix_transpose<<<dimGrid, dimBlock, 0, cuda_stream>>>(d_A, d_B);

        
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
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, INT_MAX);
    int seed = dis(gen);

    // make event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // device matrix malloc
    int *d_A, *d_B, *d_C;
    cudaMalloc((void **) &d_A, sizeof(int)*N*N);
    cudaMalloc((void **) &d_B, sizeof(int)*N*N);
    cudaMalloc((void **) &d_C, sizeof(int)*N*N);

    dim3 dimGrid((N - 1) / GRID_SIZE + 1, (N - 1) / GRID_SIZE + 1, 1);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    d_rand_matrix<<<1, 1>>>(seed, d_A);
    d_rand_matrix<<<1, 1>>>(seed, d_B);

    while(!flag);
    
    // ===========================================
    // record matrix multiple - normal
    cudaEventRecord(start, cuda_stream);
    

    d_mm_normal<<<dimGrid, dimBlock, 0, cuda_stream>>>(d_A, d_B, d_C);
    // cudaDeviceSynchronize();

    cudaEventRecord(stop, cuda_stream);
    cudaEventSynchronize(stop);

    // calculate elapsed time
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    printf("%d normal mm : %f ms\n", seed, gpu_elapsed_time_ms);


    // ============================================
    // record matrix multiple - shared memory
    cudaEventRecord(start, cuda_stream);

    d_mm_shared_mem<<<dimGrid, dimBlock, 0, cuda_stream>>>(d_A, d_B, d_C);
    // cudaDeviceSynchronize();

    cudaEventRecord(stop, cuda_stream);
    cudaEventSynchronize(stop);

    // calculate elapsed time
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    printf("%d shared memory mm : %f ms\n", seed, gpu_elapsed_time_ms);

    // clean
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // copy result to host(cpu) for check
    // int *h_C;
    // cudaMemcpy(h_C, d_C, sizeof(int)*N*N, cudaMemcpyDeviceToHost);

    // free
    // free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

void *h_mm_thread(void *arg){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, INT_MAX);
    int seed = dis(gen);

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

    // calculate elapsed time
    cudaEventElapsedTime(&cpu_elapsed_time_ms, start, stop);
    printf("%d host mm : %f ms\n", seed, cpu_elapsed_time_ms);

    // clean
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(h_A);
    cudaFree(h_B);
    cudaFree(h_C);

    return 0;
}

int main(void) {
    // thread per block 1024
    // block per sm 1792??
    // N by N
    int arg = 0;
    // int thread_return;
    int thread_id;
    int status;

    pthread_t kernel_n[KERNEL_CNT];
    pthread_t dirty;
    
    pthread_create(&dirty, NULL, d_mt_thread, (void*)&arg);
    
    for (int i = 0; i < KERNEL_CNT; i++){
        thread_id = pthread_create(&kernel_n[i], NULL, d_mm_thread, (void*)&arg);
        if (thread_id){
            printf("pthread create error : %d", thread_id);
            exit(0);
        }
    }
    pthread_join(dirty, (void **)&status);


    flag = true;
    for (int i = 0; i < KERNEL_CNT; i++){
        thread_id = pthread_join(kernel_n[i], (void **)&status);
        if (thread_id){
            printf("pthread create error : %d", thread_id);
            exit(0);
        }
    }

    flag = false;
    // =========================== cpu?

    for (int i = 0; i < KERNEL_CNT; i++){
        thread_id = pthread_create(&kernel_n[i], NULL, h_mm_thread, (void*)&arg);
        if (thread_id){
            printf("pthread create error : %d", thread_id);
            exit(0);
        }
    }

    flag = true;
    for (int i = 0; i < KERNEL_CNT; i++){
        thread_id = pthread_join(kernel_n[i], (void **)&status);
        if (thread_id){
            printf("pthread create error : %d", thread_id);
            exit(0);
        }
    }
}