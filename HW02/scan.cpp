//
// Created by aidan on 3/02/2026.
//

#include "scan.h"

#include <cstddef>

// ACKNOWLEDGEMENT: The problem was solved by me 95% and ChatGPT 5%
// The AI model spotted that arr[idx-1] was being used instead of output[idx-1]

// Performs an inclusive scan on input array arr and stores
// the result in the output array
// arr and output are arrays of n elements
void scan(const float *arr, float *output, std::size_t n) {
    output[0] = arr[0];
    for (std::size_t idx = 1; idx < n; ++idx) {
        output[idx] = output[idx-1] + arr[idx];
    }
}
