//
// Created by aidan on 17/02/2026.
//

#include "matmul.cuh"

#include <algorithm>
#include <cuda.h>
#include <iostream>

// Please see task1.cu for comments on AI usage.

__global__ void matmul_kernel(const float *A, const float *B, float *C, size_t n) {
    size_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_idx >= n * n) {
        return;
    }

    // Integer division by n allows for index increments by n
    size_t row_idx = thread_idx / n;
    // Modulo by n allows for finer-grained access
    size_t column_idx = thread_idx % n;
    // Access via row_idx * n + column_idx allow you to access all elements
    // As n allows for jumps by row, and the offset allows for access to a specific cell in the row

    C[row_idx * n + column_idx] = 0;
    for (size_t k = 0; k < n; k++) {
        C[row_idx * n + column_idx] += A[row_idx * n + k] * B[k * n + column_idx];
    }
}

void matmul(const float *A, const float *B, float *C, size_t n, unsigned int threads_per_block) {
    size_t number_of_blocks = (n * n + threads_per_block - 1) / threads_per_block;

    matmul_kernel<<<number_of_blocks, threads_per_block>>>(A, B, C, n);
}
