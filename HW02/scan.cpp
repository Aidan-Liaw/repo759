//
// Created by aidan on 3/02/2026.
//

#include "scan.h"

#include <cstddef>

// Performs an inclusive scan on input array arr and stores
// the result in the output array
// arr and output are arrays of n elements
void scan(const float *arr, float *output, std::size_t n) {
    output[0] = arr[0];
    for (int idx = 1; idx < n; ++idx) {
        output[idx] = arr[idx-1] + arr[idx];
    }
}
