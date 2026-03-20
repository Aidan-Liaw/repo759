//
// Created by aidan on 19/03/2026.
//

#include "convolution.h"

// ACKNOWLEDGEMENT: 95% of the work is my own, and 5% of the work is ChatGPTs.
// For this task, I asked ChatGPT "Is my code correct?" and provided it with my code
// It picked up differences between traditional convolution and the HW02 specification

static inline float get_image_element(const float *image, std::size_t n, long long output_idx1, long long output_idx2) {
    bool is_output_idx1_valid = output_idx1 < n && output_idx1 >= 0;
    bool is_output_idx2_valid = output_idx2 < n && output_idx2 >= 0;


    if (is_output_idx1_valid && is_output_idx2_valid) {
        return image[output_idx1 * n + output_idx2];
    }

    // Both are outside bounds, so corner
    if (!is_output_idx1_valid && !is_output_idx2_valid) {
        return 0;
    }

    return 1;
}

void convolve(const float *image, float *output, std::size_t n, const float *mask, std::size_t m) {
    // Guaranteed to be an integer, as m is an odd number
    std::size_t k = (m - 1) / 2;
#pragma omp parallel for collapse(2)
    for (std::size_t x = 0; x < n; x++) {
        for (std::size_t y = 0; y < n; y++) {
            float sum = 0;
            for (std::size_t i = 0; i < m; i++) {
                std::size_t output_idx1 = static_cast<std::size_t>(x + i) - k;
#pragma omp simd reduction(+:sum)
                for (std::size_t j = 0; j < m; j++) {
                    std::size_t output_idx2 = static_cast<std::size_t >(y + j) - k;
                    float modifier = get_image_element(image, n, output_idx1, output_idx2);
                    sum += mask[i * m + j] * modifier;
                }
            }
            output[x * n + y] = sum;
        }
    }
}