//
// Created by aidan on 10/04/2026.
//

#include "reduce.h"

// ACKNOWLEDGEMENT: 5% of the work is my own, and 95% of the work is
// from Advanced OpenMP Host Performance and 5.0 Features.pdf
// All optimisations came from there

// REF: Advanced OpenMP Host Performance and 5.0 Features.pdf
float reduce(const float *arr, const size_t l, const size_t r) {
    float sum = 0.0f;
#pragma omp parallel for simd reduction(+:sum) schedule(simd: static, 5)
    for (size_t idx = l; idx < r; ++idx) {
        sum += arr[idx];
    }
    return sum;
}