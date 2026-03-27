//
// Created by aidan on 26/03/2026.
//

#include "montecarlo.h"

#include <cmath>


// ACKNOWLEDGEMENT: 95% of the work is my own, and 5% of the work is ChatGPT's
// ChatGPT was provided the code and was asked "Is the code correct?".
// It did not find any issues

// REF: https://www.geeksforgeeks.org/dsa/estimating-value-pi-using-monte-carlo/
int montecarlo(const size_t n, const float *x, const float *y, const float radius) {
    int point_count = 0;

#pragma omp parallel for simd reduction(+:point_count)
    for (size_t idx = 0; idx < n; idx++) {
        float distance_squared = (*(x + idx) * *(x + idx)) + (*(y + idx) * *(y + idx));
        point_count += (distance_squared <= (radius * radius));
    }

    return point_count;
}
