//
// Created by aidan on 19/03/2026.
//

#include "matmul.h"

#include <algorithm>

// ACKNOWLEDGEMENT: 95% of the work is my own, and 5% of the work is ChatGPTs.
// For this task, I asked ChatGPT "Is my code correct?" and provided it with my code
// It picked up no issues

void mmul(const float* A, const float* B, float* C, const std::size_t n) {
#pragma omp parallel for
    for (std::size_t i = 0; i < n; i++) {
        for (std::size_t k = 0; k < n; k++) {
#pragma omp simd
            for (std::size_t j = 0; j < n; j++) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}
