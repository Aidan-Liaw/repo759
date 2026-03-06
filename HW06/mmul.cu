//
// Created by aidan on 5/03/2026.
//

#include "mmul.cuh"

#include <iterator>

// ACKNOWLEDGEMENT: 95% of the work is my own, and 5% of the work is ChatGPTs.
// For this task, I asked ChatGPT "Is my code correct?" and provided it with my code
// It picked up that the correct calculation is C := A*B + C, not C := A*B
// which meant that my beta value had to change.
// It also picked up that I had used the wrong computeType
// Otherwise, it was satisfied with my code.

void mmul(cublasHandle_t handle, const float* A, const float* B, float* C, int n) {
	float alpha = 1;
	float beta = 1;

    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
    	&alpha, A, CUDA_R_32F, n,
    	B, CUDA_R_32F, n, &beta,
    	C, CUDA_R_32F, n,
    	CUBLAS_COMPUTE_32F , CUBLAS_GEMM_DEFAULT);

	cudaDeviceSynchronize();
}
